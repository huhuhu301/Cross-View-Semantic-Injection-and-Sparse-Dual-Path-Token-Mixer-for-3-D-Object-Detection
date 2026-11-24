# Modified by RV-SDTM contributors; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Run the native KITTI metric in the dedicated Numba environment, without importing Torch."""
import json
import pickle
import sys
from pathlib import Path


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kitti_object_eval_python.eval import get_official_eval_result
    with Path(sys.argv[1]).open('rb') as stream:
        ground_truth, predicted, classes = pickle.load(stream)
    report, values = get_official_eval_result(ground_truth, predicted, classes)
    Path(sys.argv[2]).write_text(json.dumps([report, {k: float(v) for k, v in values.items()}], allow_nan=False))
