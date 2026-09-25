# Modified by RV-SDTM contributors for this public release; see NOTICE.
from .detector3d_template import Detector3DTemplate
from .centerpoint import CenterPoint
from .transfusion import TransFusion

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'CenterPoint': CenterPoint,
    'TransFusion': TransFusion,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
