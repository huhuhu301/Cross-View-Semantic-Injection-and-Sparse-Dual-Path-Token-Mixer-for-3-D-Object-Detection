#!/usr/bin/env python3
# Modified by RV-SDTM contributors for this public release; see NOTICE.
"""Evaluate independently filtered Waymo 3-D box-center distance intervals."""

import argparse
import json
import math
import re
import subprocess
from pathlib import Path


DISTANCE_INTERVALS = ((0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (50, 75))
CLASSES = ('VEHICLE', 'PEDESTRIAN', 'CYCLIST')
METRIC_LINE = re.compile(
    r'^OBJECT_TYPE_TYPE_(VEHICLE|PEDESTRIAN|CYCLIST)_LEVEL_([12]):\s*'
    r'\[mAP\s+([0-9eE+.-]+)\]\s*\[mAPH\s+([0-9eE+.-]+)\]\s*$')


def center_distance(box):
    return math.sqrt(box.center_x ** 2 + box.center_y ** 2 + box.center_z ** 2)


def filter_objects(objects, lower, upper):
    filtered = type(objects)()
    filtered.no_label_zone_objects.extend(objects.no_label_zone_objects)
    for item in objects.objects:
        if lower <= center_distance(item.object.box) < upper:
            filtered.objects.add().CopyFrom(item)
    return filtered


def parse_metrics(output):
    metrics = {}
    for line in output.splitlines():
        match = METRIC_LINE.match(line.strip())
        if match:
            category, level, ap, aph = match.groups()
            metrics['{}/L{}'.format(category, level)] = {'mAP': float(ap), 'mAPH': float(aph)}
    expected = {'{}/L{}'.format(category, level) for category in CLASSES for level in (1, 2)}
    if set(metrics) != expected or any(not math.isfinite(v) for item in metrics.values() for v in item.values()):
        raise ValueError('Official evaluator did not return six finite class/level metric pairs')
    for level in (1, 2):
        metrics['AVERAGE/L{}'.format(level)] = {
            name: sum(metrics['{}/L{}'.format(category, level)][name] for category in CLASSES) / 3
            for name in ('mAP', 'mAPH')}
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True, help='Waymo prediction Objects .bin')
    parser.add_argument('--ground-truth', type=Path, required=True, help='matching validation GT Objects .bin')
    parser.add_argument('--metrics-binary', type=Path, required=True, help='official compute_detection_metrics_main')
    parser.add_argument('--output-dir', type=Path, required=True, help='new output directory')
    args = parser.parse_args()
    for path in (args.predictions, args.ground_truth, args.metrics_binary):
        if not path.is_file():
            parser.error('Missing input: {}'.format(path))
    if args.output_dir.exists():
        parser.error('Output directory already exists: {}'.format(args.output_dir))

    from waymo_open_dataset.protos import metrics_pb2
    predictions, ground_truth = metrics_pb2.Objects(), metrics_pb2.Objects()
    predictions.ParseFromString(args.predictions.read_bytes())
    ground_truth.ParseFromString(args.ground_truth.read_bytes())
    args.output_dir.mkdir(parents=True)
    summary = {'distance': '3-D box-center norm', 'interval': '[lower, upper)', 'metric_scale': 1.0,
               'predictions': str(args.predictions), 'ground_truth': str(args.ground_truth), 'results': {}}
    for lower, upper in DISTANCE_INTERVALS:
        name = '{}_{}m'.format(lower, upper)
        target = args.output_dir / name
        target.mkdir()
        pred_filtered = filter_objects(predictions, lower, upper)
        gt_filtered = filter_objects(ground_truth, lower, upper)
        pred_file, gt_file = target / 'predictions.bin', target / 'ground_truth.bin'
        pred_file.write_bytes(pred_filtered.SerializeToString())
        gt_file.write_bytes(gt_filtered.SerializeToString())
        result = subprocess.run([str(args.metrics_binary.resolve()), str(pred_file), str(gt_file)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (target / 'official_metrics.txt').write_text(result.stdout)
        result.check_returncode()
        summary['results'][name] = {
            'lower_m': lower, 'upper_m': upper,
            'prediction_count': len(pred_filtered.objects), 'gt_count': len(gt_filtered.objects),
            'metrics': parse_metrics(result.stdout)}
        print(name, summary['results'][name]['metrics']['AVERAGE/L2'])
    (args.output_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
