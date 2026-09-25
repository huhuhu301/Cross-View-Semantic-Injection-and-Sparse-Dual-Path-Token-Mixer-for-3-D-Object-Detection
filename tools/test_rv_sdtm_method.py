#!/usr/bin/env python3
# Modified by RV-SDTM contributors for this public release; see NOTICE.
"""Dataset-free tests of RV-SDTM formulas, routing and sparse feature alignment."""

import copy
import contextlib
import io
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import _init_path
import torch
from torch import nn

from pcdet.models.backbones_3d.rv_sdtm import LocalMixer, RouterBranch, SDTMBlock
from pcdet.models.backbones_3d.vfe.dynamic_voxel_vfe import RangeViewVFE
from pcdet.models.model_utils.cross_view_dca import (
    AngularFusion, CrossViewDCA, fuse_rv_features,
    normalized_range_bias, uniform_subsample_indices, wrapped_angular_distance,
)
from pcdet.utils.spconv_utils import spconv
from validate_rv_sdtm_release import (
    REPO_ROOT, build_dataset_stub, load_runtime_config, normalize_mirror_value,
    working_directory,
)


class ConfigTests(unittest.TestCase):
    def test_waymo_settings(self):
        config = load_runtime_config(REPO_ROOT / 'tools/cfgs/waymo_models/rv_sdtm.yaml')
        self.assertEqual(config.MODEL.VFE.RV_FEATURE_SHAPE, [64, 1024])
        backbone = config.MODEL.BACKBONE_3D
        self.assertEqual((backbone.RV_ATTN_DIM, backbone.RV_HEADS), (80, 4))
        self.assertEqual((backbone.RV_DCA_K, backbone.RV_MAX_TOKENS), (6, 12000))
        self.assertFalse(backbone.RV_ALPHA_LEARNABLE)
        self.assertEqual(backbone.RV_ALPHA_INIT, 0.3)
        self.assertEqual(backbone.SDTM.ROUTER_KEEP_RATIO, 0.3)
        self.assertEqual(backbone.SDTM.ROUTER_SELECTION_CFG.HYBRID_RANDOM_RATIO, 0.25)
        self.assertTrue(backbone.SDTM.ROUTER_SELECTION_CFG.STRAIGHT_THROUGH_SCORE)
        self.assertEqual((config.SEED, config.OPTIMIZATION.NUM_EPOCHS), (666, 12))
        self.assertTrue(config.SYNC_BN)
        self.assertFalse(config.OPTIMIZATION.USE_AMP)

    def test_training_command_overrides(self):
        import train
        path = str(REPO_ROOT / 'tools/cfgs/waymo_models/rv_sdtm.yaml')
        with working_directory(REPO_ROOT):
            with patch('sys.argv', ['train.py', '--cfg_file', path]):
                args, _ = train.parse_config()
            self.assertEqual(args.seed, 666)
            self.assertTrue(args.fix_random_seed and args.sync_bn)
            self.assertFalse(args.use_amp)
            with patch('sys.argv', ['train.py', '--cfg_file', path, '--seed', '123',
                                   '--set', 'OPTIMIZATION.USE_AMP', 'True', 'SYNC_BN', 'False']):
                args, _ = train.parse_config()
            self.assertEqual(args.seed, 123)
            self.assertTrue(args.use_amp)
            self.assertFalse(args.sync_bn)

    def test_ablation_inheritance_from_both_launch_directories(self):
        from easydict import EasyDict
        from pcdet.config import cfg_from_yaml_file
        variants = ('voxel_only', 'add', 'concat', 'geocva_range', 'wo_geocva',
                    'local_only', 'router_only', 'wo_gate', 'z_maxpool', 'wo_afd')
        for variant in variants:
            path = Path('cfgs/waymo_models/ablations/rv_sdtm_{}.yaml'.format(variant))
            with working_directory(REPO_ROOT):
                root_cfg = cfg_from_yaml_file(path, EasyDict())
            with working_directory(REPO_ROOT / 'tools'):
                tools_cfg = cfg_from_yaml_file(path, EasyDict())
            self.assertEqual(normalize_mirror_value(root_cfg), normalize_mirror_value(tools_cfg))
            self.assertEqual(root_cfg.MODEL.BACKBONE_3D.RV_ATTN_DIM, 80)
            self.assertIn('DATASET', root_cfg.DATA_CONFIG)
            self.assertEqual(root_cfg.MODEL.DENSE_HEAD.NAME, 'SparseDynamicHead')
            backbone = root_cfg.MODEL.BACKBONE_3D
            if variant in ('add', 'concat', 'geocva_range'):
                self.assertEqual(backbone.RV_FUSION, variant)
            if variant in ('voxel_only', 'wo_geocva'):
                self.assertEqual(backbone.RV_FUSION, 'none')
                self.assertEqual(root_cfg.MODEL.VFE.RV_ENABLED, variant != 'voxel_only')
            if variant == 'local_only':
                self.assertFalse(backbone.SDTM.USE_ROUTER_STAGE2)
            if variant == 'router_only':
                self.assertFalse(backbone.SDTM.USE_LOCAL)
                self.assertFalse(backbone.SDTM.USE_GATE)
            if variant == 'wo_gate':
                self.assertFalse(backbone.SDTM.USE_GATE)
            if variant == 'z_maxpool':
                self.assertEqual(backbone.APPOOL_MODE, 'max')
            if variant == 'wo_afd':
                self.assertFalse(backbone.AFD)


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(666)

    def test_cap_and_seam(self):
        self.assertEqual(uniform_subsample_indices(13, 5, 'cpu').tolist(), [0, 3, 6, 9, 12])
        self.assertEqual(uniform_subsample_indices(6, 5, 'cpu').tolist(), [0, 2, 4])
        self.assertEqual(uniform_subsample_indices(0, 5, 'cpu').numel(), 0)
        distance = wrapped_angular_distance(torch.tensor([[0.99, 0.4]]),
                                            torch.tensor([[0.01, 0.4]]), 0.05, 0.08)
        torch.testing.assert_close(distance, torch.tensor([[0.4]]))

    def test_add_concat(self):
        x, rv = torch.randn(2, 3), torch.randn(4, 5)
        uv = torch.tensor([[0.99, 0.5], [0.45, 0.5]])
        rv_uv = torch.tensor([[0.01, 0.5], [0.8, 0.2], [0.4, 0.5], [0.7, 0.5]])
        for mode in ('add', 'concat'):
            module = AngularFusion(3, 5, mode, max_rv_tokens=2, vox_chunk=1)
            selected = rv[torch.tensor([0, 2])]
            expected = x + module.projection(selected) if mode == 'add' else module.projection(torch.cat([x, selected], 1))
            module.train()
            torch.testing.assert_close(module(x, uv, rv, rv_uv), expected)
            module.eval()
            torch.testing.assert_close(module(x, uv, rv, rv_uv), expected)
            self.assertEqual(len(list(module.children())), 1)
            self.assertIsNone(module.projection.bias)

    def test_range_bias(self):
        actual = normalized_range_bias(torch.tensor([10., 0.]), torch.tensor([20., 0.]))
        torch.testing.assert_close(actual, torch.tensor([-3.125, 0.]))

    def test_attention_against_formula(self):
        for use_range in (False, True):
            module = CrossViewDCA(3, 5, 8, 4, k=2, max_rv_tokens=3, vox_chunk=1,
                                  tau=0.6, learnable_alpha=False, use_range_bias=use_range)
            x, rv = torch.randn(2, 3), torch.randn(7, 5)
            uv, rv_uv = torch.rand(2, 2), torch.rand(7, 2)
            ranges, rv_ranges = torch.tensor([4., 9.]), torch.arange(1., 8.)
            keep = uniform_subsample_indices(7, 3, 'cpu')
            distance = wrapped_angular_distance(uv, rv_uv[keep], module.su, module.sv)
            idx = distance.topk(2, largest=False, dim=1).indices
            q = module.wq(module.ln_q(x)).reshape(2, 4, 2)
            kv = module.ln_kv(rv[keep])
            k = module.wk(kv)[idx].reshape(2, 2, 4, 2).transpose(1, 2)
            v = module.wv(kv)[idx].reshape(2, 2, 4, 2).transpose(1, 2)
            bias = -distance.gather(1, idx).square() / (2 * module.sigma ** 2)
            if use_range:
                bias += normalized_range_bias(ranges[:, None], rv_ranges[keep][idx])
            weights = ((q[:, :, None] * k).sum(-1) / (math.sqrt(2) * 0.6) + bias[:, None]).softmax(-1)
            expected = module.norm(x + 0.3 * module.wo((weights[..., None] * v).sum(2).reshape(2, 8)))
            module.train()
            actual = module(x, uv, rv, rv_uv, ranges, rv_ranges)
            torch.testing.assert_close(actual, expected)
            module.eval()
            torch.testing.assert_close(module(x, uv, rv, rv_uv, ranges, rv_ranges), actual)
            self.assertNotIn('alpha', dict(module.named_parameters()))
            self.assertIn('alpha', dict(module.named_buffers()))
            actual.square().sum().backward()
            self.assertTrue(torch.isfinite(module.wq.weight.grad).all())

    def test_missing_and_misaligned_anchors(self):
        x = torch.randn(2, 3)
        fusion = AngularFusion(3, 5, 'add')
        self.assertIs(fuse_rv_features(x, None, {}, None, None), x)
        with self.assertRaises(KeyError):
            fuse_rv_features(x, torch.zeros(2), {}, nn.Identity(), fusion)
        with self.assertRaises(ValueError):
            fuse_rv_features(x, torch.zeros(2), {'vox_uv': torch.zeros(3, 2),
                             'rv_tokens': [], 'rv_uv': []}, nn.Identity(), fusion)


class DistanceTests(unittest.TestCase):
    def test_three_dimensional_distance_and_boundaries(self):
        from eval_waymo_ranges import center_distance, filter_objects
        from waymo_open_dataset.protos import metrics_pb2
        self.assertEqual(center_distance(SimpleNamespace(center_x=3, center_y=4, center_z=12)), 13)
        objects = metrics_pb2.Objects()
        for x in (0, 14.999, 15, 29.999, 30, 74.999, 75):
            obj = objects.objects.add()
            obj.object.box.center_x = x
        self.assertEqual(len(filter_objects(objects, 0, 15).objects), 2)
        self.assertEqual(len(filter_objects(objects, 15, 30).objects), 2)
        self.assertEqual(len(filter_objects(objects, 60, 75).objects), 1)

    def test_official_output_parsing(self):
        from eval_waymo_ranges import CLASSES, parse_metrics
        output = '\n'.join('OBJECT_TYPE_TYPE_{}_LEVEL_{}: [mAP 0.5] [mAPH 0.4]'.format(c, l)
                           for c in CLASSES for l in (1, 2))
        metrics = parse_metrics(output)
        self.assertAlmostEqual(metrics['AVERAGE/L2']['mAPH'], 0.4)
        with self.assertRaises(ValueError):
            parse_metrics('')


class RouterTests(unittest.TestCase):
    def make_router(self, **kwargs):
        return RouterBranch(8, 8, 2, 2, 8, 'test', keep_ratio=kwargs.get('keep_ratio', 0.3),
                            selection_strategy='learned_hybrid', selection_cfg={
                                'HYBRID_RANDOM_RATIO': 0.25, 'MIN_TOKENS': 2,
                                'SCORE_TEMPERATURE': 0.8, 'SCORE_NOISE_STD': 0.1,
                                'STRAIGHT_THROUGH_SCORE': True})

    def test_eval_is_full_budget_top_and_rng_free(self):
        router = self.make_router().eval()
        x = torch.randn(40, 8)
        indices = torch.arange(40)
        state = torch.random.get_rng_state()
        selected = router._select_indices(x, indices, 12)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        expected = router.score_head(x).squeeze(-1).topk(12).indices
        self.assertTrue(torch.equal(selected, expected))
        self.assertEqual(router._select_indices(x, indices[:1], 2).tolist(), [0])

    def test_training_random_budget(self):
        router = self.make_router().train()
        x, indices = torch.randn(40, 8), torch.arange(40)
        with patch.object(router, '_compute_scores', return_value=torch.arange(40.)):
            with patch.object(router, '_random_select', wraps=router._random_select) as sample:
                selected = router._select_indices(x, indices, 12)
        self.assertEqual(sample.call_args[0][1], 3)
        self.assertEqual(selected.unique().numel(), 12)
        self.assertEqual(set(selected[:9].tolist()), set(range(31, 40)))
        self.assertTrue(all(i not in range(31, 40) for i in selected[9:].tolist()))

    def test_st_identity_and_score_gradient(self):
        for keep in (0.3, 1.0):
            router = self.make_router(keep_ratio=keep)
            x = torch.randn(5, 8, requires_grad=True)
            actual = router._apply_score_gradient(x)
            self.assertTrue(torch.equal(actual, x))
            actual.square().sum().backward()
            gradients = [p.grad for p in router.score_head.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_local_operators(self):
        local = list(LocalMixer(8, 'test').block.children())
        self.assertEqual(len(local), 8)
        self.assertEqual([tuple(local[i].kernel_size) for i in (0, 3, 6)], [(3, 3, 3), (1, 3, 3), (1, 1, 1)])
        self.assertTrue(all(isinstance(local[i], nn.BatchNorm1d) for i in (1, 4, 7)))
        self.assertTrue(all(isinstance(local[i], nn.SiLU) for i in (2, 5)))
        self.assertEqual(len({id(local[i].weight) for i in (1, 4, 7)}), 3)


class ProfilerTests(unittest.TestCase):
    def test_timing_boundary_and_warmup_exclusion(self):
        import profile_inference as profiler
        from easydict import EasyDict
        events = []

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))

            def cuda(self):
                return self

            def load_params_from_file(self, *args, **kwargs):
                self.strict = kwargs['strict']

            def forward(self, batch):
                events.append('forward')
                if 'gt_boxes' in batch:
                    raise AssertionError('GT-dependent recall must be outside the timing boundary')
                return self.weight

        class Loader:
            def __len__(self):
                return 3

            def __iter__(self):
                for _ in range(3):
                    events.append('load')
                    yield {'gt_boxes': torch.zeros(1)}

        config = EasyDict(DATA_CONFIG={'DATA_PROCESSOR': []}, CLASS_NAMES=['Vehicle'], MODEL={})
        args = SimpleNamespace(cfg_file='model.yaml', ckpt=Path('model.pth'), workers=0,
                               warmup=1, frames=2, runs=2, seed=666, output=None, set_cfgs=None)
        model, stream = Model(), io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(profiler, 'parse_args', return_value=args))
            stack.enter_context(patch.object(profiler, 'cfg_from_yaml_file', return_value=config))
            stack.enter_context(patch.object(profiler, 'build_dataloader', return_value=(object(), Loader(), None)))
            stack.enter_context(patch.object(profiler, 'build_network', return_value=model))
            stack.enter_context(patch.object(profiler, 'load_data_to_gpu', side_effect=lambda b: events.append('transfer')))
            stack.enter_context(patch.object(profiler.common_utils, 'create_logger', return_value=MagicMock()))
            stack.enter_context(patch.object(profiler.common_utils, 'set_random_seed'))
            stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
            stack.enter_context(patch.object(torch.cuda, 'get_device_name', return_value='test-device'))
            stack.enter_context(patch.object(torch.cuda, 'max_memory_allocated', return_value=1024 ** 3))
            reset = stack.enter_context(patch.object(torch.cuda, 'reset_peak_memory_stats'))
            stack.enter_context(patch.object(torch.cuda, 'synchronize', side_effect=lambda: events.append('sync')))
            stack.callback(setattr, torch.backends.cuda.matmul, 'allow_tf32', torch.backends.cuda.matmul.allow_tf32)
            stack.callback(setattr, torch.backends.cudnn, 'allow_tf32', torch.backends.cudnn.allow_tf32)
            ticks = iter([0, 1, 2, 2.01, 3, 3.02] * 2)

            def clock():
                events.append('clock')
                return next(ticks)

            stack.enter_context(patch.object(profiler.time, 'perf_counter', side_effect=clock))
            stack.enter_context(contextlib.redirect_stdout(stream))
            profiler.main()
        report = json.loads(stream.getvalue())
        self.assertAlmostEqual(report['mean_latency_ms'], 15)
        self.assertEqual(report['peak_allocated_gib'], 1)
        self.assertEqual(len(report['runs']), 2)
        self.assertTrue(model.strict)
        self.assertEqual(reset.call_count, 2)
        self.assertEqual(events, ['load', 'transfer', 'sync', 'clock', 'forward', 'sync', 'clock'] * 6)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA and compiled sparse operations required')
class SparseTests(unittest.TestCase):
    def sparse_input(self):
        coordinates = torch.cartesian_prod(torch.arange(2), torch.arange(3), torch.arange(4), torch.arange(4))
        coordinates[:, 2:] *= 2
        features = torch.randn(coordinates.shape[0], 8, device='cuda', requires_grad=True)
        return spconv.SparseConvTensor(features, coordinates.int().cuda(), [8, 16, 16], 2)

    def test_coarse_residual_and_inverse_alignment(self):
        router = RouterTests().make_router().cuda().eval()
        x = self.sparse_input()
        captured = {}
        router.router_down.register_forward_hook(lambda m, a, y: captured.update(coarse=y.features.detach().clone()))
        router.router_up.register_forward_pre_hook(lambda m, a: captured.update(up_input=a[0].features.detach().clone()))
        with patch.object(router.attn, 'forward', side_effect=lambda features, *args: torch.zeros_like(features)):
            output = router(x)
        torch.testing.assert_close(captured['up_input'], captured['coarse'])
        self.assertTrue(torch.equal(output.indices, x.indices))
        self.assertEqual(output.features.shape, x.features.shape)

    def test_block_inputs_fusion_and_backward(self):
        block = SDTMBlock(8, 8, 2, 2, 8, 'fusion', keep_ratio=0.3,
                          selection_strategy='learned_hybrid').cuda().train()
        x = self.sparse_input()
        original = x.features.detach().clone()
        captured = {}
        block.local_mixer.register_forward_hook(lambda m, a, y: captured.update(local=y.features))
        block.router_branch.register_forward_pre_hook(lambda m, a: captured.update(router_input=a[0].features))
        block.router_branch.register_forward_hook(lambda m, a, y: captured.update(global_=y.features))
        block.router_gate.register_forward_pre_hook(lambda m, a: captured.update(gate_input=a[0]))
        block.router_gate.register_forward_hook(lambda m, a, y: captured.update(gate=y))
        output = block(x)
        torch.testing.assert_close(captured['router_input'], original)
        torch.testing.assert_close(captured['gate_input'], original)
        torch.testing.assert_close(output.features, original + captured['local'] + captured['gate'] * captured['global_'])
        output.features.square().mean().backward()
        invalid = [name for name, p in block.named_parameters()
                   if p.grad is None or not torch.isfinite(p.grad).all()]
        self.assertEqual(invalid, [])

    def test_waymo_variants_end_to_end(self):
        from pcdet.models import build_network
        torch.manual_seed(666)
        angles = torch.linspace(-math.pi, math.pi, 2048, device='cuda')
        radii = 8 + 42 * torch.rand(2048, device='cuda')
        points = torch.stack([torch.zeros_like(radii), radii * angles.cos(),
                              radii * angles.sin(), 2 * torch.rand_like(radii) - 1,
                              torch.rand_like(radii), torch.rand_like(radii)], 1)
        variants = ('geocva', 'voxel_only', 'add', 'concat', 'geocva_range',
                    'wo_geocva', 'local_only', 'router_only', 'wo_gate', 'z_maxpool', 'wo_afd')
        for variant in variants:
            with self.subTest(variant=variant):
                relative = ('tools/cfgs/waymo_models/rv_sdtm.yaml' if variant == 'geocva' else
                            'tools/cfgs/waymo_models/ablations/rv_sdtm_{}.yaml'.format(variant))
                config = load_runtime_config(REPO_ROOT / relative)
                model = build_network(config.MODEL, 3, build_dataset_stub(config)).cuda().eval()
                if variant == 'geocva':
                    self.assertNotIn('backbone_3d.cross_view_dca.alpha', dict(model.named_parameters()))
                    self.assertEqual(len(model.backbone_3d.stage1[0]), 1)
                    self.assertEqual(len(model.backbone_3d.stage2), 1)
                    self.assertFalse(model.backbone_3d.stage1[0][0].use_router)
                    self.assertTrue(model.backbone_3d.stage2[0].use_router)
                    self.assertFalse(hasattr(model.backbone_3d, 'rv_mid_dca'))
                    state = model.state_dict()
                    model._load_state_dict(state, strict=True)
                    incomplete = dict(state)
                    incomplete.pop('backbone_3d.cross_view_dca.wq.weight')
                    with self.assertRaises(RuntimeError):
                        model._load_state_dict(incomplete, strict=True)
                with torch.no_grad():
                    predictions, _ = model({'batch_size': 1, 'points': points})
                self.assertEqual(len(predictions), 1)
                for key in ('pred_boxes', 'pred_scores'):
                    self.assertTrue(torch.isfinite(predictions[0][key]).all())
                del model
                torch.cuda.empty_cache()

    def test_vfe_point_fusion_anchors_ranges_and_empty_sample(self):
        config = load_runtime_config(REPO_ROOT / 'tools/cfgs/waymo_models/rv_sdtm.yaml')
        data = build_dataset_stub(config)
        vfe_config = copy.deepcopy(config.MODEL.VFE)
        vfe_config.RV_FEATURE_SHAPE = [16, 32]
        vfe_config.RV_NUM_LAYERS = 1
        vfe_config.RV_LAYER_BLOCKS = [1, 1, 1]
        points = torch.tensor([[0., 5.00, 0.02, 0.01, .2, .3],
                               [0., 5.01, 0.03, 0.02, .3, .4]], device='cuda')
        for enabled in (True, False):
            vfe_config.RV_ENABLED = enabled
            vfe = RangeViewVFE(vfe_config, 5, data.voxel_size, data.grid_size, data.point_cloud_range).cuda().eval()
            self.assertEqual(vfe.pfn_layers[0].linear[0].in_features, 75 if enabled else 11)
            self.assertEqual(hasattr(vfe, 'rv_backbone'), enabled)
            observed = {}
            vfe.pfn_layers[0].register_forward_pre_hook(lambda m, a: observed.update(points=a[0]))
            if enabled:
                vfe.rv_backbone.register_forward_hook(lambda m, a, y: observed.update(rv=y))
            output = vfe({'points': points, 'batch_size': 2})
            self.assertEqual(output['voxel_features'].shape, (1, 64))
            torch.testing.assert_close(observed['points'][:, :5], points[:, 1:])
            torch.testing.assert_close(observed['points'][:, 5:8], points[:, 1:4] - points[:, 1:4].mean(0))
            if enabled:
                _, indices, _, u, v = vfe.create_range_view_feature(points, 16, 32, 45, batch_size=2)
                expected_uv = torch.stack([u / 31, v / 15], 1).mean(0, keepdim=True)
                torch.testing.assert_close(output['vox_uv'], expected_uv)
                torch.testing.assert_close(output['vox_ranges'], points[:, 1:4].norm(dim=1).mean().reshape(1))
                gathered = observed['rv'].permute(0, 2, 3, 1).reshape(-1, 64)[indices]
                torch.testing.assert_close(observed['points'][:, 11:], gathered)
                self.assertEqual(output['rv_tokens'][1].shape, (0, 64))
                self.assertEqual(output['rv_ranges'][0].shape[0], output['rv_tokens'][0].shape[0])
            else:
                self.assertNotIn('rv_tokens', output)
                self.assertNotIn('vox_uv', output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
