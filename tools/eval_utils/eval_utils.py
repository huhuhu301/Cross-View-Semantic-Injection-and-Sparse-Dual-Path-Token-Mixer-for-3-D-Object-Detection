# Modified by RV-SDTM contributors for this public release; see NOTICE.
import os
import time
import pickle
import subprocess
from pathlib import Path

import torch
from tqdm import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric[f"recall_roi_{cur_thresh}"] += ret_dict.get(f"roi_{cur_thresh}", 0)
        metric[f"recall_rcnn_{cur_thresh}"] += ret_dict.get(f"rcnn_{cur_thresh}", 0)
    metric["gt_num"] += ret_dict.get("gt", 0)

    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict[f"recall_{min_thresh}"] = (
        f"({metric[f'recall_roi_{min_thresh}']}, "
        f"{metric[f'recall_rcnn_{min_thresh}']}) / {metric['gt_num']}"
    )


def convert_pkl_file_to_bin(result_dir: Path, det_annos):
    """Serialize OpenPCDet Waymo predictions for the native metric binary."""
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2

    objects = metrics_pb2.Objects()

    with tqdm(total=len(det_annos)) as pbar:
        for det in det_annos:
            meta = det["metadata"]
            names = det["name"]
            boxes = det["boxes_lidar"]
            scores = det["score"]

            for i in range(len(names)):
                o = metrics_pb2.Object()
                o.context_name = meta["context_name"]
                o.frame_timestamp_micros = meta["timestamp_micros"]

                box = label_pb2.Label.Box()
                box.center_x = float(boxes[i][0])
                box.center_y = float(boxes[i][1])
                box.center_z = float(boxes[i][2])
                box.length = float(boxes[i][3])
                box.width = float(boxes[i][4])
                box.height = float(boxes[i][5])
                box.heading = float(boxes[i][6])
                o.object.box.CopyFrom(box)

                o.score = float(scores[i])

                cls_name = names[i]
                if cls_name == "Vehicle":
                    o.object.type = label_pb2.Label.TYPE_VEHICLE
                elif cls_name == "Pedestrian":
                    o.object.type = label_pb2.Label.TYPE_PEDESTRIAN
                elif cls_name == "Cyclist":
                    o.object.type = label_pb2.Label.TYPE_CYCLIST
                elif cls_name == "Sign":
                    o.object.type = label_pb2.Label.TYPE_SIGN
                else:
                    continue

                objects.objects.append(o)

            pbar.update(1)

    with open(result_dir / "result.bin", "wb") as f:
        f.write(objects.SerializeToString())


def _resolve_existing_path(path_value, cfg):
    if not path_value:
        return None

    path = Path(path_value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        root_dir = Path(getattr(cfg, 'ROOT_DIR', Path.cwd()))
        candidates.insert(0, root_dir / path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_waymo_fast_eval_paths(cfg, args):
    post_cfg = cfg.MODEL.POST_PROCESSING
    data_cfg = cfg.DATA_CONFIG

    metrics_binary = (
        getattr(args, 'waymo_metrics_binary', None)
        or getattr(args, 'waymo_metrics_bin', None)
        or post_cfg.get('WAYMO_METRICS_BINARY', None)
        or post_cfg.get('WAYMO_METRICS_BIN', None)
        or data_cfg.get('WAYMO_METRICS_BINARY', None)
        or data_cfg.get('WAYMO_METRICS_BIN', None)
    )
    gt_bin = (
        getattr(args, 'waymo_gt_bin', None)
        or post_cfg.get('WAYMO_GT_BIN', None)
        or data_cfg.get('WAYMO_GT_BIN', None)
    )

    data_path = Path(data_cfg.DATA_PATH).expanduser()
    metrics_binary = metrics_binary or data_path / 'compute_detection_metrics_main'
    gt_bin = gt_bin or data_path / 'gt.bin'

    metrics_binary = _resolve_existing_path(metrics_binary, cfg)
    gt_bin = _resolve_existing_path(gt_bin, cfg)
    if metrics_binary is not None and not os.access(metrics_binary, os.X_OK):
        metrics_binary = None
    return metrics_binary, gt_bin


def waymo_fast_eval(bin_file, metrics_binary, gt_bin):
    out = subprocess.check_output(
        [str(metrics_binary), str(bin_file), str(gt_bin)], text=True
    )

    # Preserve native output in the evaluation log for auditability.
    print(out)

    result = {}

    for line in out.splitlines():
        line = line.strip()
        if line.startswith("OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1:"):
            # OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1: [mAP 0.820739] [mAPH 0.816059]
            parts = line.split("[mAP")[1].split("]")
            mAP = float(parts[0])
            mAPH = float(line.split("[mAPH")[1].split("]")[0])
            result["Vehicle/L1 mAP"] = mAP
            result["Vehicle/L1 mAPH"] = mAPH

        elif line.startswith("OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2:"):
            mAP = float(line.split("[mAP")[1].split("]")[0])
            mAPH = float(line.split("[mAPH")[1].split("]")[0])
            result["Vehicle/L2 mAP"] = mAP
            result["Vehicle/L2 mAPH"] = mAPH

        elif line.startswith("OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1:"):
            mAP = float(line.split("[mAP")[1].split("]")[0])
            mAPH = float(line.split("[mAPH")[1].split("]")[0])
            result["Pedestrian/L1 mAP"] = mAP
            result["Pedestrian/L1 mAPH"] = mAPH

        elif line.startswith("OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2:"):
            mAP = float(line.split("[mAP")[1].split("]")[0])
            mAPH = float(line.split("[mAPH")[1].split("]")[0])
            result["Pedestrian/L2 mAP"] = mAP
            result["Pedestrian/L2 mAPH"] = mAPH

        # Sign is not included in the reported three-class aggregate.
        elif line.startswith("OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1:"):
            mAP = float(line.split("[mAP")[1].split("]")[0])
            mAPH = float(line.split("[mAPH")[1].split("]")[0])
            result["Cyclist/L1 mAP"] = mAP
            result["Cyclist/L1 mAPH"] = mAPH

        elif line.startswith("OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2:"):
            mAP = float(line.split("[mAP")[1].split("]")[0])
            mAPH = float(line.split("[mAPH")[1].split("]")[0])
            result["Cyclist/L2 mAP"] = mAP
            result["Cyclist/L2 mAPH"] = mAPH

    core_keys = [
        "Vehicle/L1 mAP", "Pedestrian/L1 mAP", "Cyclist/L1 mAP",
        "Vehicle/L1 mAPH", "Pedestrian/L1 mAPH", "Cyclist/L1 mAPH",
        "Vehicle/L2 mAP", "Pedestrian/L2 mAP", "Cyclist/L2 mAP",
        "Vehicle/L2 mAPH", "Pedestrian/L2 mAPH", "Cyclist/L2 mAPH",
    ]
    missing_keys = [key for key in core_keys if key not in result]
    if missing_keys:
        raise ValueError(
            'Waymo metrics output is missing expected values: %s'
            % missing_keys
        )

    # Compute the three-class aggregate used by the public RV-SDTM report.
    result["Overall/L1 mAP"] = sum(
        result[f"{name}/L1 mAP"]
        for name in ("Vehicle", "Pedestrian", "Cyclist")
    ) / 3.0
    result["Overall/L1 mAPH"] = sum(
        result[f"{name}/L1 mAPH"]
        for name in ("Vehicle", "Pedestrian", "Cyclist")
    ) / 3.0
    result["Overall/L2 mAP"] = sum(
        result[f"{name}/L2 mAP"]
        for name in ("Vehicle", "Pedestrian", "Cyclist")
    ) / 3.0
    result["Overall/L2 mAPH"] = sum(
        result[f"{name}/L2 mAPH"]
        for name in ("Vehicle", "Pedestrian", "Cyclist")
    ) / 3.0

    return result


def eval_one_epoch(
    cfg,
    args,
    model,
    dataloader,
    epoch_id,
    logger,
    dist_test=False,
    result_dir=None,
):
    """Evaluate one checkpoint using the dataset's official protocol."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir = result_dir / "final_result" / "data"
    if getattr(args, "save_to_file", False):
        final_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize recall counters.
    metric = {"gt_num": 0}
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric[f"recall_roi_{cur_thresh}"] = 0
        metric[f"recall_rcnn_{cur_thresh}"] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info(f"*************** EPOCH {epoch_id} EVALUATION *****************")

    model.eval()

    if cfg.LOCAL_RANK == 0:
        pbar = tqdm(total=len(dataloader), leave=True, desc="eval", dynamic_ncols=True)
    else:
        pbar = None

    start_time = time.time()

    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}
        statistics_info(cfg, ret_dict, metric, disp_dict)

        annos = dataset.generate_prediction_dicts(
            batch_dict,
            pred_dicts,
            class_names,
            output_path=final_output_dir if getattr(args, "save_to_file", False) else None,
        )
        det_annos += annos

        if pbar is not None:
            pbar.set_postfix(disp_dict)
            pbar.update()

    if pbar is not None:
        pbar.close()

    # Merge distributed predictions and recall counters.
    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(
            det_annos, len(dataset), tmpdir=result_dir / "tmpdir"
        )
        metric = common_utils.merge_results_dist(
            [metric], world_size, tmpdir=result_dir / "tmpdir"
        )

    logger.info(f"*************** Performance of EPOCH {epoch_id} *****************")
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info(f"Generate label finished(sec_per_example: {sec_per_example:.4f} second).")

    if cfg.LOCAL_RANK != 0:
        return {}

    if dist_test:
        world_size = len(metric)
        for key in metric[0].keys():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric["gt_num"]
    ret_dict = {}
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric[f"recall_roi_{cur_thresh}"] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric[f"recall_rcnn_{cur_thresh}"] / max(gt_num_cnt, 1)
        logger.info(f"recall_roi_{cur_thresh}: {cur_roi_recall:.6f}")
        logger.info(f"recall_rcnn_{cur_thresh}: {cur_rcnn_recall:.6f}")
        ret_dict[f"recall/roi_{cur_thresh}"] = cur_roi_recall
        ret_dict[f"recall/rcnn_{cur_thresh}"] = cur_rcnn_recall

    total_pred_objects = sum(len(a["name"]) for a in det_annos)
    logger.info(
        "Average predicted number of objects(%d samples): %.3f"
        % (len(det_annos), total_pred_objects / max(1, len(det_annos)))
    )

    with open(result_dir / "result.pkl", "wb") as f:
        pickle.dump(det_annos, f)

    eval_metric = cfg.MODEL.POST_PROCESSING.EVAL_METRIC
    if eval_metric == 'waymo':
        metrics_binary, gt_bin = resolve_waymo_fast_eval_paths(cfg, args)
        fast_eval_succeeded = False
        if metrics_binary is not None and gt_bin is not None:
            try:
                convert_pkl_file_to_bin(result_dir, det_annos)
                ap_dict = waymo_fast_eval(
                    result_dir / 'result.bin', metrics_binary=metrics_binary,
                    gt_bin=gt_bin,
                )
                ret_dict.update(ap_dict)
                fast_eval_succeeded = True
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                logger.warning(
                    'Waymo metrics binary failed (%s: %s); falling back to '
                    'the official Python Waymo evaluator.',
                    type(error).__name__, error,
                )
        if not fast_eval_succeeded:
            if metrics_binary is None or gt_bin is None:
                logger.warning(
                    'Waymo metrics binary or gt.bin was not found; falling '
                    'back to the official Python Waymo evaluator.'
                )
            result_str, result_dict = dataset.evaluation(
                det_annos,
                class_names,
                eval_metric=eval_metric,
                output_path=final_output_dir,
            )
            logger.info(result_str)
            ret_dict.update(result_dict)
    else:
        result_str, result_dict = dataset.evaluation(
            det_annos,
            class_names,
            eval_metric=eval_metric,
            output_path=final_output_dir,
        )
        logger.info(result_str)
        ret_dict.update(result_dict)

    logger.info(f"Result is saved to {result_dir}")
    logger.info("****************Evaluation done.*****************")
    return ret_dict
