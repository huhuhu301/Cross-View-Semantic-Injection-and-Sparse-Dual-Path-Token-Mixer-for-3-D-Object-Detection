# Modified by RV-SDTM contributors for this public release; see NOTICE.
from .detector3d_template import Detector3DTemplate
from .centerpoint import CenterPoint
from .transfusion import TransFusion

from .pointpillar import PointPillar
from .second_net import SECONDNet
from .point_rcnn import PointRCNN
from .ct3d import CT3D_3CAT
from .PartA2_net import PartA2Net
from .kitti_refinement import KittiRefinement

__all__ = {
    'PointPillar': PointPillar,
    'SECONDNet': SECONDNet,
    'PointRCNN': PointRCNN,
    'CT3D_3CAT': CT3D_3CAT,
    'PartA2Net': PartA2Net,
    'KittiRefinement': KittiRefinement,
    'Detector3DTemplate': Detector3DTemplate,
    'CenterPoint': CenterPoint,
    'TransFusion': TransFusion,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
