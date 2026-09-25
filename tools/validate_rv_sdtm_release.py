#!/usr/bin/env python3
# Modified by RV-SDTM contributors for this public release; see NOTICE.
"""Validate the public RV-SDTM Waymo, nuScenes, and Argoverse2 surface.

The default checks are read-only and do not require dataset files.  ``--build``
adds a GPU model-construction smoke test with a synthetic dataset descriptor;
``--argo2-forward`` adds synthetic inference and training backward.  Neither
mode loads a checkpoint, reads dataset files, or writes artifacts.
"""

import argparse
import ast
import collections
import contextlib
import json
import math
import os
import random
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

from easydict import EasyDict


warnings.filterwarnings('ignore', category=DeprecationWarning)


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / 'tools'
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TOOLS_DIR))


AV2_CLASS_NAMES = [
    'Regular_vehicle', 'Pedestrian', 'Bicyclist', 'Motorcyclist',
    'Wheeled_rider', 'Bollard', 'Construction_cone', 'Sign',
    'Construction_barrel', 'Stop_sign',
    'Mobile_pedestrian_crossing_sign', 'Large_vehicle', 'Bus', 'Box_truck',
    'Truck', 'Vehicular_trailer', 'Truck_cab', 'School_bus',
    'Articulated_bus', 'Message_board_trailer', 'Bicycle', 'Motorcycle',
    'Wheeled_device', 'Wheelchair', 'Stroller', 'Dog',
]


CONFIG_SPECS = {
    'waymo': {
        'default': 'tools/cfgs/waymo_models/rv_sdtm.yaml',
        'dataset': 'WaymoDataset',
        'detector': 'CenterPoint',
        'vfe': 'RangeViewVFE',
        'backbone': 'RVSDTM',
        'dense_head': 'SparseDynamicHead',
        'eval_metric': 'waymo',
        'rv_vox_chunk': 2048,
    },
    'nuscenes': {
        'default': 'tools/cfgs/nuscenes_models/rv_sdtm.yaml',
        'dataset': 'NuScenesDataset',
        'detector': 'TransFusion',
        'vfe': 'RangeViewVFE',
        'backbone': 'RVSDTMNuScenes',
        'dense_head': 'TransFusionHead',
        'eval_metric': 'nuscenes',
        'rv_vox_chunk': 2048,
    },
    'argo2': {
        'default': 'tools/cfgs/argoverse_models/rv_sdtm.yaml',
        'dataset': 'Argo2Dataset',
        'detector': 'CenterPoint',
        'vfe': 'RangeViewVFE',
        'backbone': 'RVSDTMAV2',
        'dense_head': 'SparseDynamicHead',
        'eval_metric': 'argo2',
        'rv_vox_chunk': 512,
    },
}


REGISTRY_SOURCES = {
    'dataset': 'pcdet/datasets/__init__.py',
    'detector': 'pcdet/models/detectors/__init__.py',
    'vfe': 'pcdet/models/backbones_3d/vfe/__init__.py',
    'backbone': 'pcdet/models/backbones_3d/__init__.py',
    'dense_head': 'pcdet/models/dense_heads/__init__.py',
}


class Results:
    def __init__(self):
        self.failures = []
        self.passes = 0

    def pass_(self, message):
        self.passes += 1
        print('[PASS] {}'.format(message))

    def fail(self, message):
        self.failures.append(message)
        print('[FAIL] {}'.format(message))

    def check(self, condition, message, failure=None):
        if condition:
            self.pass_(message)
            return True
        self.fail(failure or message)
        return False


@contextlib.contextmanager
def working_directory(path):
    previous = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(previous)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Read-only validation of the three public RV-SDTM configs')
    parser.add_argument(
        '--waymo-config', default=CONFIG_SPECS['waymo']['default'])
    parser.add_argument(
        '--nuscenes-config', default=CONFIG_SPECS['nuscenes']['default'])
    parser.add_argument(
        '--argo2-config', default=CONFIG_SPECS['argo2']['default'])
    parser.add_argument(
        '--build', action='store_true',
        help='construct each model and optimizer on a GPU without dataset files')
    parser.add_argument(
        '--argo2-forward', action='store_true',
        help='run synthetic AV2 inference and training backward on a GPU')
    parser.add_argument(
        '--device', type=int, default=0,
        help='CUDA device index after CUDA_VISIBLE_DEVICES remapping')
    parser.add_argument(
        '--seed', type=int, default=20260827,
        help='seed for the synthetic AV2 forward/backward smoke')
    return parser.parse_args()


def resolve_repo_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_runtime_config(path):
    from pcdet.config import cfg_from_yaml_file

    config = EasyDict()
    with working_directory(REPO_ROOT):
        cfg_from_yaml_file(str(path), config)
    return config


def normalize_mirror_value(value):
    """Remove intentionally path-specific keys before mirror comparison."""
    if isinstance(value, dict):
        return {
            key: normalize_mirror_value(child)
            for key, child in value.items()
            if key not in {'_BASE_CONFIG_', 'DATA_PATH'}
        }
    if isinstance(value, list):
        return [normalize_mirror_value(child) for child in value]
    return value


def mirror_path(path):
    relative = path.relative_to(REPO_ROOT)
    parts = relative.parts
    if len(parts) >= 2 and parts[:2] == ('tools', 'cfgs'):
        return REPO_ROOT.joinpath('cfgs', *parts[2:])
    if parts and parts[0] == 'cfgs':
        return REPO_ROOT.joinpath('tools', 'cfgs', *parts[1:])
    raise ValueError('{} is not under tools/cfgs or cfgs'.format(path))


def find_key_paths(value, target, prefix=()):
    paths = []
    if isinstance(value, dict):
        for key, child in value.items():
            current = prefix + (str(key),)
            if key == target:
                paths.append('.'.join(current))
            paths.extend(find_key_paths(child, target, current))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(find_key_paths(child, target, prefix + (str(index),)))
    return paths


def find_key_values(value, target, prefix=()):
    matches = []
    if isinstance(value, dict):
        for key, child in value.items():
            current = prefix + (str(key),)
            if key == target:
                matches.append(('.'.join(current), child))
            matches.extend(find_key_values(child, target, current))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            matches.extend(
                find_key_values(child, target, prefix + (str(index),)))
    return matches


def iter_path_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_path_strings(child)


def is_absolute_machine_path(value):
    return (
        Path(value).is_absolute()
        or (len(value) >= 3 and value[1] == ':' and value[2] in {'/', '\\'})
    )


def find_absolute_config_paths(value, prefix=()):
    matches = []
    if isinstance(value, dict):
        for key, child in value.items():
            current = prefix + (str(key),)
            if 'PATH' in str(key).upper():
                for path_value in iter_path_strings(child):
                    if is_absolute_machine_path(path_value):
                        matches.append(('.'.join(current), path_value))
            matches.extend(find_absolute_config_paths(child, current))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            matches.extend(
                find_absolute_config_paths(
                    child, prefix + (str(index),)))
    return matches


def flatten_groups(groups):
    return [name for group in groups for name in group]


def check_exact_partition(results, label, class_names, groups):
    flattened = flatten_groups(groups)
    counts = collections.Counter(flattened)
    duplicates = sorted(name for name, count in counts.items() if count != 1)
    missing = sorted(set(class_names) - set(flattened))
    unknown = sorted(set(flattened) - set(class_names))
    results.check(
        not duplicates and not missing and not unknown
        and len(flattened) == len(class_names),
        '{} is an exact class partition'.format(label),
        '{} is not an exact class partition: duplicates={}, missing={}, '
        'unknown={}'.format(label, duplicates, missing, unknown),
    )


def check_mirror(results, name, path):
    try:
        counterpart = mirror_path(path)
    except Exception as error:
        results.fail('{} mirror path: {}'.format(name, error))
        return
    if not results.check(
            counterpart.is_file(), '{} config mirror exists'.format(name),
            '{} config mirror is missing: {}'.format(name, counterpart)):
        return
    try:
        # A mirror may be a thin repository-root wrapper whose base is the
        # canonical tools/cfgs file.  Compare effective configs, not source
        # text, so wrappers and full mirrors are treated identically.
        primary = normalize_mirror_value(load_runtime_config(path))
        mirror = normalize_mirror_value(load_runtime_config(counterpart))
    except Exception as error:
        results.fail('{} mirror YAML load failed: {}'.format(name, error))
        return
    if primary == mirror:
        results.pass_(
            '{} config mirrors match (ignoring _BASE_CONFIG_/DATA_PATH)'.format(
                name))
    else:
        primary_text = json.dumps(primary, sort_keys=True, ensure_ascii=False)
        mirror_text = json.dumps(mirror, sort_keys=True, ensure_ascii=False)
        results.fail(
            '{} config mirrors differ after normalization ({} vs {} bytes)'.format(
                name, len(primary_text), len(mirror_text)))


def check_component_config(results, name, config, spec):
    observed = {
        'dataset': config.DATA_CONFIG.DATASET,
        'detector': config.MODEL.NAME,
        'vfe': config.MODEL.VFE.NAME,
        'backbone': config.MODEL.BACKBONE_3D.NAME,
        'dense_head': config.MODEL.DENSE_HEAD.NAME,
        'eval_metric': config.MODEL.POST_PROCESSING.EVAL_METRIC,
        'rv_vox_chunk': int(config.MODEL.BACKBONE_3D.RV_VOX_CHUNK),
    }
    for field, expected in spec.items():
        if field == 'default':
            continue
        results.check(
            observed[field] == expected,
            '{} {} is {}'.format(name, field, expected),
            '{} {} must be {}, got {}'.format(
                name, field, expected, observed[field]),
        )

    candidate_paths = find_key_paths(config, 'CANDIDATE_NUM')
    results.check(
        not candidate_paths,
        '{} has no unused CANDIDATE_NUM'.format(name),
        '{} contains unused CANDIDATE_NUM at {}'.format(
            name, candidate_paths),
    )

    absolute_paths = find_absolute_config_paths(config)
    results.check(
        not absolute_paths,
        '{} contains only portable relative filesystem paths'.format(name),
        '{} contains machine-specific absolute paths: {}'.format(
            name, absolute_paths),
    )

    shared_memory_values = find_key_values(config, 'USE_SHARED_MEMORY')
    invalid_shared_memory = [
        (path, value) for path, value in shared_memory_values
        if value is not False
    ]
    results.check(
        bool(shared_memory_values) and not invalid_shared_memory,
        '{} portable defaults disable shared-memory databases'.format(name),
        '{} USE_SHARED_MEMORY defaults must exist and all be False: {}'.format(
            name, invalid_shared_memory),
    )

    canonical_pooling_paths = find_key_paths(
        config, 'GROUP_POOLING_KERNEL_SIZE')
    deprecated_pooling_paths = find_key_paths(
        config, 'GREOUP_POOLING_KERNEL_SIZE')
    results.check(
        bool(canonical_pooling_paths),
        '{} uses canonical GROUP_POOLING_KERNEL_SIZE'.format(name),
        '{} is missing GROUP_POOLING_KERNEL_SIZE'.format(name),
    )
    results.check(
        not deprecated_pooling_paths,
        '{} contains no deprecated GREOUP_POOLING_KERNEL_SIZE'.format(name),
        '{} contains deprecated GREOUP_POOLING_KERNEL_SIZE at {}'.format(
            name, deprecated_pooling_paths),
    )

    vfe_cfg = config.MODEL.VFE
    backbone_cfg = config.MODEL.BACKBONE_3D
    if vfe_cfg.NAME == 'RangeViewVFE' and backbone_cfg.get('RV_FUSION', 'none') != 'none':
        results.check(
            int(vfe_cfg.RV_FEATURE_DIM) == int(backbone_cfg.RV_FEATURE_DIM),
            '{} RV feature dimensions agree'.format(name),
            '{} VFE/BACKBONE RV feature dimensions differ: {} vs {}'.format(
                name, vfe_cfg.RV_FEATURE_DIM, backbone_cfg.RV_FEATURE_DIM),
        )
        results.check(
            int(backbone_cfg.RV_ATTN_DIM) % int(backbone_cfg.RV_HEADS) == 0,
            '{} GeoCVA attention dimension is head-divisible'.format(name),
        )


def check_waymo_variants(results, config):
    backbone = config.MODEL.BACKBONE_3D
    results.check(
        config.MODEL.VFE.RV_FEATURE_SHAPE == [64, 1024]
        and (backbone.RV_ATTN_DIM, backbone.RV_HEADS) == (80, 4)
        and (backbone.RV_DCA_K, backbone.RV_MAX_TOKENS) == (6, 12000)
        and backbone.RV_ALPHA_INIT == 0.3 and not backbone.RV_ALPHA_LEARNABLE,
        'waymo RV geometry, attention dimensions and fixed injection match the method')
    results.check(
        config.SEED == 666 and config.SYNC_BN and not config.OPTIMIZATION.USE_AMP
        and config.OPTIMIZATION.NUM_EPOCHS == 12,
        'waymo fixed-seed FP32/SyncBN 12-epoch defaults are explicit')
    overrides = {
        'voxel_only': {'RV_FUSION': 'none'}, 'add': {'RV_FUSION': 'add'},
        'concat': {'RV_FUSION': 'concat'}, 'geocva_range': {'RV_FUSION': 'geocva_range'},
        'wo_geocva': {'RV_FUSION': 'none'}, 'local_only': {'USE_ROUTER_STAGE2': False},
        'router_only': {'USE_LOCAL': False, 'USE_GATE': False},
        'wo_gate': {'USE_GATE': False}, 'z_maxpool': {'APPOOL_MODE': 'max'},
        'wo_afd': {'AFD': False},
    }
    for variant, expected in overrides.items():
        relative = 'waymo_models/ablations/rv_sdtm_{}.yaml'.format(variant)
        path = REPO_ROOT / 'tools/cfgs' / relative
        try:
            with working_directory(REPO_ROOT):
                root_cfg = load_runtime_config(REPO_ROOT / 'cfgs' / relative)
            from pcdet.config import cfg_from_yaml_file
            with working_directory(TOOLS_DIR):
                tools_cfg = cfg_from_yaml_file(path, EasyDict())
            resolved = root_cfg.MODEL.BACKBONE_3D
            unchanged = all(root_cfg[key] == config[key]
                            for key in ('OPTIMIZATION', 'SEED', 'SYNC_BN', 'CLASS_NAMES'))
            observed = dict(resolved)
            observed.update(resolved.SDTM)
            results.check(
                normalize_mirror_value(root_cfg) == normalize_mirror_value(tools_cfg)
                and unchanged and all(observed[key] == value for key, value in expected.items())
                and root_cfg.MODEL.VFE.RV_ENABLED == (variant != 'voxel_only'),
                'waymo {} override and both launch-directory configs agree'.format(variant))
        except Exception as error:
            results.fail('waymo {} config failed: {}'.format(variant, error))


def check_sparse_protocol(results, name, config, layout, order, channels):
    dense_cfg = config.MODEL.DENSE_HEAD
    explicit = 'REGRESSION_LAYOUT' in dense_cfg
    results.check(
        explicit, '{} regression layout is explicit'.format(name),
        '{} DENSE_HEAD.REGRESSION_LAYOUT must be explicit'.format(name))
    observed_layout = dense_cfg.get('REGRESSION_LAYOUT', None)
    results.check(
        observed_layout == layout,
        '{} uses {} regression'.format(name, layout),
        '{} regression layout must be {}, got {}'.format(
            name, layout, observed_layout),
    )
    observed_order = list(dense_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER)
    results.check(
        observed_order == order,
        '{} HEAD_ORDER matches {}'.format(name, layout),
        '{} HEAD_ORDER must be {}, got {}'.format(name, order, observed_order),
    )
    head_dict = dense_cfg.SEPARATE_HEAD_CFG.HEAD_DICT
    for field, expected_channels in channels.items():
        observed_channels = (
            int(head_dict[field]['out_channels']) if field in head_dict else None)
        results.check(
            observed_channels == expected_channels,
            '{} {} emits {} channels'.format(
                name, field, expected_channels),
            '{} {} must emit {} channels, got {}'.format(
                name, field, expected_channels, observed_channels),
        )
    code_size = sum(
        int(head_dict[field]['out_channels'])
        for field in observed_order if field in head_dict)
    results.check(
        code_size == 8, '{} regression code size is 8'.format(name),
        '{} regression code size must be 8, got {}'.format(name, code_size))
    if 'iou' in head_dict:
        results.check(
            'iou' not in observed_order
            and int(head_dict.iou.out_channels) == 1,
            '{} IoU is a one-channel quality branch'.format(name),
            '{} IoU must be one-channel and outside HEAD_ORDER'.format(name),
        )


def check_argo2_groups(results, config):
    data_cfg = config.DATA_CONFIG
    results.check(
        float(data_cfg.get('EVALUATE_RANGE', -1)) == 200.0,
        'argo2 explicitly uses the 200 m paper evaluation range',
        'argo2 EVALUATE_RANGE must be explicitly set to 200.0',
    )
    results.check(
        data_cfg.get('EVAL_ONLY_ROI_INSTANCES', None) is True,
        'argo2 explicitly enables ROI-only evaluation',
        'argo2 EVAL_ONLY_ROI_INSTANCES must be explicitly True',
    )

    class_names = list(config.CLASS_NAMES)
    results.check(
        class_names == AV2_CLASS_NAMES,
        'argo2 has the canonical ordered 26 classes',
        'argo2 class order differs from the canonical 26-class protocol')
    results.check(
        len(class_names) == len(set(class_names)) == 26,
        'argo2 class names are 26 unique values')

    dense_groups = [
        list(group) for group in config.MODEL.DENSE_HEAD.CLASS_NAMES_EACH_HEAD]
    results.check(
        len(dense_groups) == 6,
        'argo2 SparseDynamicHead has six class groups',
        'argo2 SparseDynamicHead must have six class groups, got {}'.format(
            len(dense_groups)))
    check_exact_partition(
        results, 'argo2 SparseDynamicHead groups', class_names, dense_groups)

    dynamic_pos_num = config.MODEL.DENSE_HEAD.DYNAMIC_POS_NUM
    if isinstance(dynamic_pos_num, (list, tuple)):
        valid_dynamic = (
            len(dynamic_pos_num) == len(dense_groups)
            and all(int(value) > 0 for value in dynamic_pos_num))
    else:
        valid_dynamic = int(dynamic_pos_num) > 0
    results.check(
        valid_dynamic,
        'argo2 DYNAMIC_POS_NUM is positive and group-compatible',
        'argo2 DYNAMIC_POS_NUM must be positive scalar or one value per head')

    backbone_cfg = config.MODEL.BACKBONE_3D
    afd_groups = [list(group) for group in backbone_cfg.GROUP_CLASS_NAMES]
    check_exact_partition(results, 'argo2 AFD groups', class_names, afd_groups)
    kernels = list(backbone_cfg.GROUP_POOLING_KERNEL_SIZE)
    results.check(
        len(kernels) == len(afd_groups) + 1,
        'argo2 AFD pooling kernels include each group plus background',
        'argo2 AFD pooling kernel count must be group count + 1')


def import_registries(results):
    try:
        import torch  # Load libtorch before the repository CUDA extensions.
        import pcdet
        from pcdet.datasets import __all__ as dataset_registry
        from pcdet.models.backbones_3d import __all__ as backbone_registry
        from pcdet.models.backbones_3d.vfe import __all__ as vfe_registry
        from pcdet.models.dense_heads import __all__ as dense_head_registry
        from pcdet.models.detectors import __all__ as detector_registry
    except Exception as error:
        results.fail('registry imports failed: {}: {}'.format(
            type(error).__name__, error))
        return None

    pcdet_path = Path(pcdet.__file__).resolve()
    results.check(
        REPO_ROOT in pcdet_path.parents,
        'pcdet imports from this release tree',
        'pcdet imported from {}, not {}'.format(pcdet_path, REPO_ROOT),
    )
    return {
        'dataset': dataset_registry,
        'detector': detector_registry,
        'vfe': vfe_registry,
        'backbone': backbone_registry,
        'dense_head': dense_head_registry,
    }


def read_registry_keys(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
                isinstance(target, ast.Name) and target.id == '__all__'
                for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            raise ValueError('__all__ is not a dictionary')
        keys = []
        for key_node in node.value.keys:
            if isinstance(key_node, ast.Constant) \
                    and isinstance(key_node.value, str):
                keys.append(key_node.value)
            else:
                raise ValueError('__all__ contains a non-string key')
        return set(keys)
    raise ValueError('dictionary assignment to __all__ was not found')


def check_source_registries(results, configs):
    registries = {}
    for registry_name, relative_path in REGISTRY_SOURCES.items():
        path = REPO_ROOT / relative_path
        if not results.check(
                path.is_file(), '{} registry source exists'.format(
                    registry_name),
                '{} registry source is missing: {}'.format(
                    registry_name, path)):
            continue
        try:
            registries[registry_name] = read_registry_keys(path)
        except Exception as error:
            results.fail('{} registry source parse failed: {}: {}'.format(
                registry_name, type(error).__name__, error))
    check_registries(results, registries, configs)


def check_registries(results, registries, configs):
    if registries is None:
        return
    for name, config in configs.items():
        observed = {
            'dataset': config.DATA_CONFIG.DATASET,
            'detector': config.MODEL.NAME,
            'vfe': config.MODEL.VFE.NAME,
            'backbone': config.MODEL.BACKBONE_3D.NAME,
            'dense_head': config.MODEL.DENSE_HEAD.NAME,
        }
        for registry_name, component_name in observed.items():
            if registry_name not in registries:
                continue
            results.check(
                component_name in registries[registry_name],
                '{} {} {} is registered'.format(
                    name, registry_name, component_name),
                '{} {} {} is not registered'.format(
                    name, registry_name, component_name),
            )


def get_voxel_size(data_config):
    for processor in reversed(data_config.DATA_PROCESSOR):
        if 'VOXEL_SIZE' in processor:
            return processor.VOXEL_SIZE
    raise ValueError('DATA_CONFIG.DATA_PROCESSOR has no VOXEL_SIZE')


def build_dataset_stub(config):
    import numpy as np

    point_cloud_range = np.asarray(
        config.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
    voxel_size = np.asarray(get_voxel_size(config.DATA_CONFIG), dtype=np.float32)
    grid_float = (
        point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size
    grid_size = np.round(grid_float).astype(np.int64)
    if not np.allclose(grid_float, grid_size, rtol=0, atol=1e-4):
        raise ValueError(
            'point-cloud range is not divisible by voxel size: {}'.format(
                grid_float.tolist()))
    num_point_features = len(
        config.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list)
    return SimpleNamespace(
        class_names=list(config.CLASS_NAMES),
        point_feature_encoder=SimpleNamespace(
            num_point_features=num_point_features),
        grid_size=grid_size,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        depth_downsample_factor=None,
    )


def build_smoke(results, configs, device):
    import torch
    from pcdet.models import build_network
    from train_utils.optimization import build_optimizer

    if not torch.cuda.is_available():
        results.fail('--build requested but CUDA is unavailable')
        return
    if device < 0 or device >= torch.cuda.device_count():
        results.fail(
            '--device {} is outside the visible CUDA range [0, {})'.format(
                device, torch.cuda.device_count()))
        return
    torch.cuda.set_device(device)
    print('[INFO] build smoke uses CUDA {} ({})'.format(
        device, torch.cuda.get_device_name(device)))

    for name, config in configs.items():
        model = None
        optimizer = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            dataset_stub = build_dataset_stub(config)
            model = build_network(
                model_cfg=config.MODEL,
                num_class=len(config.CLASS_NAMES),
                dataset=dataset_stub,
            )
            model.cuda(device)
            model.eval()
            optimizer = build_optimizer(model, config.OPTIMIZATION)

            trainable_by_id = {
                id(parameter): parameter_name
                for parameter_name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            optimizer_parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group['params']
            ]
            optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
            optimizer_id_set = set(optimizer_ids)
            trainable_id_set = set(trainable_by_id)
            duplicates = len(optimizer_ids) - len(optimizer_id_set)
            missing = sorted(
                trainable_by_id[parameter_id]
                for parameter_id in trainable_id_set - optimizer_id_set)
            extra = len(optimizer_id_set - trainable_id_set)

            results.check(
                not missing and not extra and duplicates == 0,
                '{} optimizer covers all {} trainable tensors exactly once'.format(
                    name, len(trainable_id_set)),
                '{} optimizer mismatch: missing={}, extra={}, duplicates={}'.format(
                    name, missing, extra, duplicates),
            )
            results.check(
                hasattr(model, 'vfe') and hasattr(model, 'backbone_3d')
                and hasattr(model, 'dense_head'),
                '{} detector constructed all three public modules'.format(name),
            )
            parameter_count = sum(
                parameter.numel() for parameter in model.parameters())
            peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print('[INFO] {} build: {:,} parameters, {:.1f} MiB peak'.format(
                name, parameter_count, peak_mib))
        except Exception as error:
            results.fail('{} build smoke failed: {}: {}'.format(
                name, type(error).__name__, error))
        finally:
            del optimizer
            del model
            torch.cuda.empty_cache()


def argo2_forward_smoke(results, config, device, seed):
    """Exercise the public AV2 route without reading or writing dataset files."""
    import numpy as np
    import torch
    from pcdet.models import build_network

    if not torch.cuda.is_available():
        results.fail('--argo2-forward requested but CUDA is unavailable')
        return
    if device < 0 or device >= torch.cuda.device_count():
        results.fail(
            '--device {} is outside the visible CUDA range [0, {})'.format(
                device, torch.cuda.device_count()))
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = None
    try:
        dataset_stub = build_dataset_stub(config)
        model = build_network(
            model_cfg=config.MODEL,
            num_class=len(config.CLASS_NAMES),
            dataset=dataset_stub,
        ).cuda(device)

        def make_points():
            torch_device = torch.device('cuda', device)
            num_ring = 4096
            angles = torch.linspace(
                -math.pi, math.pi, num_ring, device=torch_device)
            radii = 8.0 + 42.0 * torch.rand(
                num_ring, device=torch_device)
            ring = torch.stack([
                radii * torch.cos(angles),
                radii * torch.sin(angles),
                -1.0 + 2.0 * torch.rand(num_ring, device=torch_device),
                torch.rand(num_ring, device=torch_device),
            ], dim=1)

            num_box = 768
            box = torch.stack([
                8.0 + 4.0 * torch.rand(num_box, device=torch_device),
                -1.0 + 2.0 * torch.rand(num_box, device=torch_device),
                -0.8 + 1.6 * torch.rand(num_box, device=torch_device),
                torch.rand(num_box, device=torch_device),
            ], dim=1)
            raw_points = torch.cat([ring, box], dim=0)
            batch_column = torch.zeros(
                (raw_points.shape[0], 1), device=torch_device)
            return torch.cat(
                [batch_column, raw_points], dim=1).contiguous()

        model.eval()
        with torch.no_grad():
            pred_dicts, _ = model({
                'batch_size': 1,
                'points': make_points(),
            })
        prediction = pred_dicts[0]
        num_predictions = int(prediction['pred_boxes'].shape[0])
        max_predictions = (
            int(config.MODEL.DENSE_HEAD.POST_PROCESSING.MAX_OBJ_PER_SAMPLE)
            * len(config.MODEL.DENSE_HEAD.CLASS_NAMES_EACH_HEAD))
        output_valid = (
            len(pred_dicts) == 1
            and prediction['pred_boxes'].ndim == 2
            and prediction['pred_boxes'].shape[1] == 7
            and prediction['pred_scores'].ndim == 1
            and prediction['pred_labels'].ndim == 1
            and prediction['pred_boxes'].shape[0]
            == prediction['pred_scores'].shape[0]
            == prediction['pred_labels'].shape[0]
            and num_predictions <= max_predictions
            and bool(torch.isfinite(prediction['pred_boxes']).all())
            and bool(torch.isfinite(prediction['pred_scores']).all()))
        if num_predictions:
            output_valid = (
                output_valid
                and int(prediction['pred_labels'].min()) >= 1
                and int(prediction['pred_labels'].max())
                <= len(config.CLASS_NAMES))
        results.check(
            output_valid,
            'argo2 synthetic inference returns finite valid predictions')
        print('[INFO] argo2 synthetic inference: {} predictions'.format(
            num_predictions))

        model.train()
        model.zero_grad(set_to_none=True)
        gt_boxes = torch.zeros(
            (1, 4, 8), device=torch.device('cuda', device))
        gt_boxes[0, 0] = gt_boxes.new_tensor(
            [10.0, 0.0, 0.0, 4.0, 2.0, 1.6, 0.0, 1.0])
        return_dict, metric_dict, _ = model({
            'batch_size': 1,
            'points': make_points(),
            'gt_boxes': gt_boxes,
        })
        loss = return_dict['loss']
        results.check(
            loss.ndim == 0 and bool(torch.isfinite(loss)),
            'argo2 synthetic training loss is finite')
        loss.backward()

        gradient_groups = {
            'RV': ('vfe.rv_input_roj', 'vfe.rv_backbone'),
            'GeoCVA': ('backbone_3d.cross_view_dca',),
            'SDTM': ('backbone_3d.stage1', 'backbone_3d.stage2'),
            'APPool': ('backbone_3d.app',),
            'AFD': ('backbone_3d.cls_conv', 'backbone_3d.afd_layers'),
            'head': ('dense_head.heads_list',),
        }
        for label, prefixes in gradient_groups.items():
            selected = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if name.startswith(prefixes)
            ]
            with_gradient = [
                gradient for gradient in selected if gradient is not None]
            finite = all(
                bool(torch.isfinite(gradient).all())
                for gradient in with_gradient)
            nonzero = sum(
                bool((gradient != 0).any()) for gradient in with_gradient)
            results.check(
                bool(selected) and bool(with_gradient) and finite
                and nonzero > 0,
                'argo2 {} receives finite nonzero gradients'.format(label))
            print('[INFO] argo2 {} gradients: {}/{} present, {} nonzero'.format(
                label, len(with_gradient), len(selected), nonzero))

        scalar_metrics_finite = all(
            math.isfinite(float(value))
            for value in metric_dict.values()
            if isinstance(value, (int, float)))
        results.check(
            scalar_metrics_finite,
            'argo2 synthetic training metrics are finite')
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print('[INFO] argo2 synthetic training: loss={:.8f}, '
              'peak={:.1f} MiB'.format(float(loss.detach()), peak_mib))
    except Exception as error:
        results.fail('argo2 forward/backward smoke failed: {}: {}'.format(
            type(error).__name__, error))
    finally:
        del model
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    results = Results()
    config_paths = {
        'waymo': resolve_repo_path(args.waymo_config),
        'nuscenes': resolve_repo_path(args.nuscenes_config),
        'argo2': resolve_repo_path(args.argo2_config),
    }
    configs = {}

    for name, path in config_paths.items():
        if not results.check(
                path.is_file(), '{} config exists'.format(name),
                '{} config is missing: {}'.format(name, path)):
            continue
        try:
            config = load_runtime_config(path)
        except Exception as error:
            results.fail('{} config load failed: {}: {}'.format(
                name, type(error).__name__, error))
            continue
        configs[name] = config
        results.pass_('{} config and base config load'.format(name))
        check_mirror(results, name, path)
        check_component_config(results, name, config, CONFIG_SPECS[name])

    if 'waymo' in configs:
        check_waymo_variants(results, configs['waymo'])
        check_sparse_protocol(
            results, 'waymo', configs['waymo'], 'packed_bbox', ['bbox'],
            {'bbox': 8})
    if 'argo2' in configs:
        check_sparse_protocol(
            results, 'argo2', configs['argo2'], 'split_fields',
            ['center', 'center_z', 'dim', 'rot'],
            {'center': 2, 'center_z': 1, 'dim': 3, 'rot': 2})
        check_argo2_groups(results, configs['argo2'])

    check_source_registries(results, configs)

    if args.build or args.argo2_forward:
        registries = import_registries(results)
        check_registries(results, registries, configs)

    if args.build:
        missing_configs = sorted(set(CONFIG_SPECS) - set(configs))
        if missing_configs:
            results.fail(
                'build smoke cannot cover missing configs: {}'.format(
                    missing_configs))
        else:
            build_smoke(results, configs, args.device)

    if args.argo2_forward:
        if 'argo2' not in configs:
            results.fail('argo2 forward smoke cannot run without its config')
        else:
            argo2_forward_smoke(
                results, configs['argo2'], args.device, args.seed)

    print('[SUMMARY] {} passed checks, {} failures'.format(
        results.passes, len(results.failures)))
    if results.failures:
        for index, failure in enumerate(results.failures, start=1):
            print('[SUMMARY][{}] {}'.format(index, failure))
        return 1
    print('[SUMMARY] RV-SDTM release validation passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
