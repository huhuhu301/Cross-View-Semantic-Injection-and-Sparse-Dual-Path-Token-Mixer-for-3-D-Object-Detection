"""Check KITTI model selection and root/tools configuration parity without CUDA."""
from copy import deepcopy
from pathlib import Path

import _init_path
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file


ROOT = Path(__file__).resolve().parents[1]
# detector, epochs, frames per GPU, RoI GeoCVA, backbone SDTM
ENTRIES = {
    'kitti_pointpillars': ('PointPillar', 80, 4, False, False),
    'kitti_second': ('SECONDNet', 80, 4, False, False),
    'kitti_pointrcnn': ('PointRCNN', 80, 2, False, False),
    'kitti_parta2_anchor': ('PartA2Net', 80, 4, False, False),
    'kitti_ct3d': ('CT3D_3CAT', 200, 3, False, False),
    'kitti_rv_geocva': ('KittiRefinement', 100, 1, True, False),
    'kitti_sdtm': ('KittiRefinement', 100, 1, False, True),
    'kitti_rv_geocva_sdtm': ('KittiRefinement', 100, 1, True, True),
}


def main():
    common = []
    selections = {}
    for directory in (ROOT, ROOT / 'tools'):
        models = directory / 'cfgs' / 'kitti_models'
        assert {p.stem for p in models.glob('*.yaml')} == set(ENTRIES), models
        for name, (detector, epochs, batch, geo, mixer) in ENTRIES.items():
            cfg = cfg_from_yaml_file(models / (name + '.yaml'), EasyDict())
            assert cfg.CLASS_NAMES == ['Car', 'Pedestrian', 'Cyclist'], name
            data = cfg.DATA_CONFIG
            assert data.DATASET == 'KittiDataset' and data.DATA_SPLIT.test == 'val', name
            assert (directory / data.DATA_PATH).resolve() == (ROOT / 'data/kitti').resolve(), name
            data.DATA_PATH = 'data/kitti'
            model, opt = cfg.MODEL, cfg.OPTIMIZATION
            assert model.NAME == detector and model.POST_PROCESSING.EVAL_METRIC == 'kitti', name
            assert model.SPCONV_DIRECT_TABLE == (detector in ('CT3D_3CAT', 'KittiRefinement')), name
            assert opt.LEAF_MODULE_GROUPS, name
            assert (opt.NUM_EPOCHS, opt.BATCH_SIZE_PER_GPU) == (epochs, batch), name
            assert cfg == selections.setdefault(name, cfg), name
            head, backbone = model.get('ROI_HEAD', {}), model.get('BACKBONE_3D', {})
            assert ('RV_CASCADE_CLS' in head, 'RV_SPARSE_MIXER' in backbone) == (geo, mixer), name
            if detector == 'KittiRefinement':
                assert head.STAGES == 3 and head.NAME == 'CascadeRefinementHead', name
                assert backbone.NAME == 'KittiVoxelBackbone' and model.DENSE_HEAD.NAME == 'MaskedAnchorHead', name
                if directory == ROOT:
                    # Compare the full training setup after removing just the two module switches.
                    control = deepcopy(cfg)
                    control.MODEL.ROI_HEAD.pop('RV_CASCADE_CLS', None)
                    control.MODEL.BACKBONE_3D.pop('RV_SPARSE_MIXER', None)
                    common.append(control)
    assert len(common) == 3 and all(c == common[0] for c in common)
    print('KITTI: eight matched configurations; three ablations share one training setup.')


if __name__ == '__main__':
    main()
