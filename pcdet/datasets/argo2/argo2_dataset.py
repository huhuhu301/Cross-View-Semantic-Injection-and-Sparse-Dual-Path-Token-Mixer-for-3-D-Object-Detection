# Modified by RV-SDTM contributors for this public release; see NOTICE.
import argparse
import copy
import pickle
from os import path as osp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from av2.utils.io import read_feather
from easydict import EasyDict
from tqdm import tqdm

from ...config import cfg_from_yaml_file
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate
from .argo2_utils.constants import LABEL_ATTR
from .argo2_utils.so3 import quat_to_yaw, yaw_to_quat


def process_single_segment(segment_path, split, info_list, ts2idx, output_dir, save_bin):
    test_mode = 'test' in split
    if not test_mode:
        segment_anno = read_feather(Path(osp.join(segment_path, 'annotations.feather')))
    segname = Path(segment_path).name

    lidar_dir = Path(segment_path) / 'sensors' / 'lidar'
    frame_path_list = sorted(path.name for path in lidar_dir.glob('*.feather'))

    for frame_name in frame_path_list:
        ts = int(Path(frame_name).stem)

        if not test_mode:
            frame_anno = segment_anno[segment_anno['timestamp_ns'] == ts]
        else:
            frame_anno = None

        frame_path = osp.join(segment_path, 'sensors/lidar/', frame_name)
        frame_info = process_and_save_frame(frame_path, frame_anno, ts2idx, segname, output_dir, save_bin)
        info_list.append(frame_info)


def process_and_save_frame(frame_path, frame_anno, ts2idx, segname, output_dir, save_bin):
    frame_info = {}
    frame_info['uuid'] = segname + '/' + Path(frame_path).stem
    frame_info['sample_idx'] = ts2idx[frame_info['uuid']]
    frame_info['image'] = dict()
    frame_info['point_cloud'] = dict(
        num_features=4,
        velodyne_path=None,
    )
    frame_info['calib'] = dict()  # not need for lidar-only
    frame_info['pose'] = dict()  # not need for single frame
    frame_info['annos'] = dict(
        name=None,
        truncated=None,
        occluded=None,
        alpha=None,
        bbox=None,  # not need for lidar-only
        dimensions=None,
        location=None,
        rotation_y=None,
        index=None,
        group_ids=None,
        camera_id=None,
        difficulty=None,
        num_points_in_gt=None,
    )
    frame_info['sweeps'] = []  # not need for single frame
    if frame_anno is not None:
        frame_anno = frame_anno[frame_anno['num_interior_pts'] > 0]
        cuboid_params = frame_anno.loc[:, list(LABEL_ATTR)].to_numpy()
        cuboid_params = torch.from_numpy(cuboid_params)
        yaw = quat_to_yaw(cuboid_params[:, -4:])
        xyz = cuboid_params[:, :3]
        lwh = cuboid_params[:, [3, 4, 5]]

        cat = frame_anno['category'].to_numpy().tolist()
        cat = [c.lower().capitalize() for c in cat]
        cat = np.array(cat)

        num_obj = len(cat)

        annos = frame_info['annos']
        annos['name'] = cat
        annos['truncated'] = np.zeros(num_obj, dtype=np.float64)
        annos['occluded'] = np.zeros(num_obj, dtype=np.int64)
        annos['alpha'] = -10 * np.ones(num_obj, dtype=np.float64)
        annos['dimensions'] = lwh.numpy().astype(np.float64)
        annos['location'] = xyz.numpy().astype(np.float64)
        annos['rotation_y'] = yaw.numpy().astype(np.float64)
        annos['index'] = np.arange(num_obj, dtype=np.int32)
        annos['difficulty'] = np.zeros(num_obj, dtype=np.int32)
        annos['num_points_in_gt'] = frame_anno['num_interior_pts'].to_numpy().astype(np.int32)
    # frame_info['group_ids'] = np.arange(num_obj, dtype=np.int32)
    prefix2split = {'0': 'training', '1': 'training', '2': 'testing'}
    sample_idx = frame_info['sample_idx']
    split = prefix2split[sample_idx[0]]
    abs_save_path = osp.join(output_dir, split, 'velodyne', f'{sample_idx}.bin')
    rel_save_path = osp.join(split, 'velodyne', f'{sample_idx}.bin')
    frame_info['point_cloud']['velodyne_path'] = rel_save_path
    if save_bin:
        save_point_cloud(frame_path, abs_save_path)
    return frame_info


def save_point_cloud(frame_path, save_path):
    lidar = read_feather(Path(frame_path))
    lidar = lidar.loc[:, ['x', 'y', 'z', 'intensity']].to_numpy().astype(np.float32)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    lidar.tofile(save_path)


def prepare(root):
    root = Path(root)
    ts2idx = {}
    ts_list = []
    bin_idx_list = []
    seg_path_list = []
    seg_split_list = []
    if root.name != 'sensor':
        raise ValueError(f'Expected the raw AV2 root to end in "sensor", got: {root}')
    # include test if you need it
    splits = ['train', 'val']  # , 'test']
    num_train_samples = 0
    num_val_samples = 0
    num_test_samples = 0

    # 0 for training, 1 for validation and 2 for testing.
    prefixes = [0, 1, ]  # 2]

    for i in range(len(splits)):
        split = splits[i]
        prefix = prefixes[i]
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f'Missing AV2 split directory: {split_root}')
        seg_file_list = sorted(x.name for x in split_root.iterdir() if x.is_dir())
        print(f'num of {split} segments:', len(seg_file_list))
        for seg_idx, seg_name in enumerate(seg_file_list):
            seg_path = split_root / seg_name
            seg_path_list.append(str(seg_path))
            seg_split_list.append(split)
            assert seg_idx < 1000
            lidar_dir = seg_path / 'sensors' / 'lidar'
            frame_path_list = sorted(
                x.name for x in lidar_dir.iterdir() if x.is_file() and x.suffix == '.feather'
            )
            for frame_idx, frame_path in enumerate(frame_path_list):
                assert frame_idx < 1000
                bin_idx = str(prefix) + str(seg_idx).zfill(3) + str(frame_idx).zfill(3)
                ts = Path(frame_path).stem
                ts = seg_name + '/' + ts  # ts is not unique, so add seg_name
                ts2idx[ts] = bin_idx
                ts_list.append(ts)
                bin_idx_list.append(bin_idx)
        if split == 'train':
            num_train_samples = len(ts_list)
        elif split == 'val':
            num_val_samples = len(ts_list) - num_train_samples
        else:
            num_test_samples = len(ts_list) - num_train_samples - num_val_samples
    # print three num samples
    print('num of train samples:', num_train_samples)
    print('num of val samples:', num_val_samples)
    print('num of test samples:', num_test_samples)

    assert len(ts_list) == len(set(ts_list))
    assert len(bin_idx_list) == len(set(bin_idx_list))
    return ts2idx, seg_path_list, seg_split_list

def _process_argo2_segments(seg_path_list, seg_split_list, info_list, ts2idx, output_dir, save_bin,
                            token=0, num_process=1):
    for seg_i, seg_path in enumerate(seg_path_list):
        if seg_i % num_process != token:
            continue
        print(f'processing segment: {seg_i}/{len(seg_path_list)}')
        split = seg_split_list[seg_i]
        process_single_segment(seg_path, split, info_list, ts2idx, output_dir, save_bin)


class Argo2Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.argo2_infos = []
        self.include_argo2_data(self.mode)
        self.evaluate_range = float(dataset_cfg.get('EVALUATE_RANGE', 200.0))
        self.eval_only_roi_instances = bool(
            dataset_cfg.get('EVAL_ONLY_ROI_INSTANCES', True)
        )

    def include_argo2_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Argoverse2 dataset')
        argo2_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                argo2_infos.extend(infos)

        self.argo2_infos.extend(argo2_infos)

        if self.logger is not None:
            self.logger.info('Total samples for Argo2 dataset: %d' % (len(argo2_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_boxes_img = pred_boxes
            pred_boxes_camera = pred_boxes

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.argo2_infos) * self.total_epochs

        return len(self.argo2_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.argo2_infos)

        info = copy.deepcopy(self.argo2_infos[index])

        sample_idx = Path(info['point_cloud']['velodyne_path']).stem
        calib = None
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_bboxes_3d
            })

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            input_dict['points'] = points

        input_dict['calib'] = calib
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def format_results(self,
                       outputs,
                       class_names,
                       pklfile_prefix=None,
                       submission_prefix=None,
                       ):
        """Format the results to .feather file with argo2 format.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        import pandas as pd

        assert len(self.argo2_infos) == len(outputs)
        num_samples = len(outputs)
        print('\nGot {} samples'.format(num_samples))

        serialized_dts_list = []

        print('\nConvert predictions to Argoverse 2 format')
        for i in range(num_samples):
            out_i = outputs[i]
            log_id, ts = self.argo2_infos[i]['uuid'].split('/')
            track_uuid = None
            #cat_id = out_i['labels_3d'].numpy().tolist()
            #category = [class_names[i].upper() for i in cat_id]
            category = [class_name.upper() for class_name in out_i['name']]
            serialized_dts = pd.DataFrame(
                self.lidar_box_to_argo2(out_i['bbox']).numpy(), columns=list(LABEL_ATTR)
            )
            serialized_dts["score"] = out_i['score']
            serialized_dts["log_id"] = log_id
            serialized_dts["timestamp_ns"] = int(ts)
            serialized_dts["category"] = category
            serialized_dts_list.append(serialized_dts)

        dts = (
            pd.concat(serialized_dts_list)
            .set_index(["log_id", "timestamp_ns"])
            .sort_index()
        )

        dts = dts.sort_values("score", ascending=False).reset_index()

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.feather')):
                pklfile_prefix = f'{pklfile_prefix}.feather'
            dts.to_feather(pklfile_prefix)
            print(f'Result is saved to {pklfile_prefix}.')

        dts = dts.set_index(["log_id", "timestamp_ns"]).sort_index()

        return dts

    def lidar_box_to_argo2(self, boxes):
        boxes = torch.Tensor(boxes)
        cnt_xyz = boxes[:, :3]
        lwh = boxes[:, [3, 4, 5]]
        yaw = boxes[:, 6]

        quat = yaw_to_quat(yaw)
        argo_cuboid = torch.cat([cnt_xyz, lwh, quat], dim=1)
        return argo_cuboid

    def evaluation(self,
                 results,
                 class_names,
                 eval_metric='argo2',
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 output_path=None,
                 pipeline=None):
        """Evaluation in Argo2 protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            eval_metric (str): Metric protocol name. The canonical value is
                ``argo2``.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str: float]: results of each evaluation metric
        """
        if str(eval_metric).lower() not in {'argo2', 'av2'}:
            raise ValueError(
                'Argo2Dataset supports only the argo2 evaluation protocol, '
                f'got {eval_metric!r}'
            )

        from av2.evaluation.detection.constants import CompetitionCategories
        from av2.evaluation.detection.utils import DetectionCfg
        from av2.evaluation.detection.eval import evaluate
        from av2.utils.io import read_feather

        dts = self.format_results(results, class_names, pklfile_prefix, submission_prefix)
        argo2_root = self.root_path
        val_anno_path = osp.join(argo2_root, 'val_anno.feather')
        gts = read_feather(Path(val_anno_path))
        gts = gts.set_index(["log_id", "timestamp_ns"]).sort_values("category")

        # Restrict annotations to the configured validation infos, not to
        # frames that happened to produce a detection. Intersecting with the
        # detection index would silently discard false negatives on empty
        # prediction frames and inflate AP.
        evaluated_frames = {
            tuple(info['uuid'].split('/')) for info in self.argo2_infos
        }
        evaluated_frames = {
            (log_id, int(timestamp_ns))
            for log_id, timestamp_ns in evaluated_frames
        }
        gts = gts[gts.index.isin(evaluated_frames)].sort_index()

        categories = set(x.value for x in CompetitionCategories)
        categories &= set(gts["category"].unique().tolist())

        dataset_dir = Path(argo2_root) / 'sensor' / 'val'
        cfg = DetectionCfg(
            dataset_dir=dataset_dir,
            categories=tuple(sorted(categories)),
            max_range_m=self.evaluate_range,
            eval_only_roi_instances=self.eval_only_roi_instances,
        )

        # Evaluate using Argoverse detection API.
        eval_dts, eval_gts, metrics = evaluate(
            dts.reset_index(), gts.reset_index(), cfg
        )

        valid_categories = sorted(categories) + ["AVERAGE_METRICS"]
        ap_dict = {}
        for index, row in metrics.iterrows():
            for metric_name, metric_value in row.items():
                ap_dict[f'{index}/{metric_name}'] = float(metric_value)
        return metrics.loc[valid_categories], ap_dict

    def create_groundtruth_database(self, used_classes=None):
        """Create per-object files and the optional shared-memory backing array.

        The first pass extracts objects and records their final global offsets.  The
        second pass streams those files into an ``open_memmap`` array.  This avoids
        retaining every object's points in RAM or constructing one large
        ``np.concatenate`` input.
        """
        database_save_path = self.root_path / 'gt_database'
        db_info_save_path = self.root_path / 'argo2_dbinfos.pkl'
        global_data_save_path = self.root_path / 'argo2_dbinfos_global.npy'
        database_save_path.mkdir(parents=True, exist_ok=True)

        allowed_classes = set(used_classes) if used_classes is not None else None
        all_db_infos = {}
        ordered_db_infos = []
        global_offset = 0
        num_point_features = 4

        for info_idx in tqdm(range(len(self.argo2_infos)), desc='AV2 GT database pass 1/2'):
            info = copy.deepcopy(self.argo2_infos[info_idx])
            frame_id = Path(info['point_cloud']['velodyne_path']).stem
            points = self.get_lidar(frame_id)
            if points.ndim != 2 or points.shape[1] != num_point_features:
                raise ValueError(
                    f'Expected AV2 points with {num_point_features} features, got {points.shape} for {frame_id}'
                )

            annos = info.get('annos', {})
            gt_names = annos.get('name')
            if gt_names is None or len(gt_names) == 0:
                continue
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, :3], gt_boxes[:, :7])
            difficulties = annos.get('difficulty')

            for gt_idx, gt_name in enumerate(gt_names):
                gt_name = str(gt_name)
                if allowed_classes is not None and gt_name not in allowed_classes:
                    continue

                gt_points = points[point_masks[gt_idx] > 0].copy()
                gt_points[:, :3] -= gt_boxes[gt_idx, :3]
                safe_name = gt_name.replace('/', '_').replace(' ', '_')
                filename = f'{frame_id}_{safe_name}_{gt_idx}.bin'
                filepath = database_save_path / filename
                with open(filepath, 'wb') as f:
                    gt_points.astype(np.float32, copy=False).tofile(f)

                start_offset = global_offset
                global_offset += int(gt_points.shape[0])
                if difficulties is None:
                    difficulty = 0
                else:
                    difficulty = int(difficulties[gt_idx])
                db_info = {
                    'name': gt_name,
                    'path': filepath.relative_to(self.root_path).as_posix(),
                    'image_idx': frame_id,
                    'gt_idx': gt_idx,
                    'box3d_lidar': gt_boxes[gt_idx],
                    'num_points_in_gt': int(gt_points.shape[0]),
                    'difficulty': difficulty,
                    'global_data_offset': [start_offset, global_offset],
                }
                all_db_infos.setdefault(gt_name, []).append(db_info)
                ordered_db_infos.append(db_info)

        global_data = np.lib.format.open_memmap(
            global_data_save_path,
            mode='w+',
            dtype=np.float32,
            shape=(global_offset, num_point_features),
        )
        for db_info in tqdm(ordered_db_infos, desc='AV2 GT database pass 2/2'):
            object_path = self.root_path / db_info['path']
            object_points = np.fromfile(object_path, dtype=np.float32).reshape(-1, num_point_features)
            start_offset, end_offset = db_info['global_data_offset']
            expected_points = end_offset - start_offset
            if object_points.shape[0] != expected_points:
                raise RuntimeError(
                    f'GT database point-count mismatch for {object_path}: '
                    f'{object_points.shape[0]} != {expected_points}'
                )
            global_data[start_offset:end_offset] = object_points
        global_data.flush()
        del global_data

        all_db_infos = {name: all_db_infos[name] for name in sorted(all_db_infos)}
        for class_name, class_infos in all_db_infos.items():
            print(f'Database {class_name}: {len(class_infos)}')
        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

        return {
            'database_path': database_save_path,
            'db_info_path': db_info_save_path,
            'global_data_path': global_data_save_path,
            'num_objects': len(ordered_db_infos),
            'num_points': global_offset,
        }

def _write_image_sets(output_root, split_infos):
    image_sets_path = Path(output_root) / 'ImageSets'
    image_sets_path.mkdir(parents=True, exist_ok=True)
    for split_name in ('train', 'val'):
        sample_ids = sorted(str(info['sample_idx']) for info in split_infos[split_name])
        contents = ''.join(f'{sample_id}\n' for sample_id in sample_ids)
        (image_sets_path / f'{split_name}.txt').write_text(contents, encoding='utf-8')


def create_argo2_infos(output_root, save_bin=True):
    """Convert raw ``<output_root>/sensor`` data into OpenPCDet AV2 files."""
    output_root = Path(output_root).expanduser().resolve()
    sensor_root = output_root / 'sensor'
    ts2idx, seg_path_list, seg_split_list = prepare(sensor_root)

    (output_root / 'training' / 'velodyne').mkdir(parents=True, exist_ok=True)
    info_list = []
    _process_argo2_segments(
        seg_path_list, seg_split_list, info_list, ts2idx, output_root, save_bin,
        token=0, num_process=1
    )
    if not info_list:
        raise RuntimeError(f'No AV2 frames were discovered below {sensor_root}')
    info_list.sort(key=lambda info: str(info['sample_idx']))

    split_infos = {
        'train': [info for info in info_list if str(info['sample_idx']).startswith('0')],
        'val': [info for info in info_list if str(info['sample_idx']).startswith('1')],
    }
    known_count = len(split_infos['train']) + len(split_infos['val'])
    if known_count != len(info_list):
        raise RuntimeError(f'Unexpected AV2 sample prefix for {len(info_list) - known_count} frames')

    for split_name, infos in split_infos.items():
        info_path = output_root / f'argo2_infos_{split_name}.pkl'
        with open(info_path, 'wb') as f:
            pickle.dump(infos, f)
    _write_image_sets(output_root, split_infos)

    val_seg_paths = [
        Path(seg_path) for seg_path, split in zip(seg_path_list, seg_split_list) if split == 'val'
    ]
    if not val_seg_paths:
        raise RuntimeError(f'No AV2 validation segments were discovered below {sensor_root / "val"}')
    val_annotations = []
    for segment_path in val_seg_paths:
        segment_annos = read_feather(segment_path / 'annotations.feather')
        segment_annos['log_id'] = segment_path.name
        val_annotations.append(segment_annos)
    pd.concat(val_annotations, ignore_index=True).to_feather(output_root / 'val_anno.feather')

    return {
        'train_info_path': output_root / 'argo2_infos_train.pkl',
        'val_info_path': output_root / 'argo2_infos_val.pkl',
        'val_annotation_path': output_root / 'val_anno.feather',
        'num_train_samples': len(split_infos['train']),
        'num_val_samples': len(split_infos['val']),
    }


def create_gt_database(output_root, dataset_cfg, used_classes=None):
    output_root = Path(output_root).expanduser().resolve()
    dataset = Argo2Dataset(
        dataset_cfg=dataset_cfg,
        class_names=None,
        root_path=output_root,
        logger=common_utils.create_logger(),
        training=True,
    )
    if not dataset.argo2_infos:
        raise RuntimeError(
            f'No training infos were loaded from {output_root}; run --func create_infos first'
        )
    return dataset.create_groundtruth_database(used_classes=used_classes)


def _load_dataset_cfg(cfg_file):
    loaded_cfg = cfg_from_yaml_file(str(cfg_file), EasyDict())
    return loaded_cfg.DATA_CONFIG if 'DATA_CONFIG' in loaded_cfg else loaded_cfg


def parse_config():
    parser = argparse.ArgumentParser(description='Prepare Argoverse 2 data for RV-SDTM')
    parser.add_argument(
        '--root_path', type=Path, default=Path('data/argo2'),
        help='Processed dataset root; raw AV2 sensor data must be at <root_path>/sensor',
    )
    parser.add_argument(
        '--cfg_file', type=Path, default=None,
        help='Dataset or model YAML; required for create_gt_database and create_all',
    )
    parser.add_argument(
        '--func', choices=('create_infos', 'create_gt_database', 'create_all'), default='create_all',
    )
    parser.add_argument('--used_classes', nargs='*', default=None)
    parser.add_argument('--no_save_bin', action='store_true', help='Only write metadata, not point .bin files')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    if args.func in ('create_gt_database', 'create_all') and args.cfg_file is None:
        raise ValueError('--cfg_file is required for create_gt_database and create_all')
    if args.func == 'create_all' and args.no_save_bin:
        raise ValueError(
            '--no_save_bin cannot be combined with create_all because the GT '
            'database consumes the generated point files'
        )

    if args.func in ('create_infos', 'create_all'):
        create_argo2_infos(args.root_path, save_bin=not args.no_save_bin)
    if args.func in ('create_gt_database', 'create_all'):
        create_gt_database(
            args.root_path,
            dataset_cfg=_load_dataset_cfg(args.cfg_file),
            used_classes=args.used_classes,
        )
