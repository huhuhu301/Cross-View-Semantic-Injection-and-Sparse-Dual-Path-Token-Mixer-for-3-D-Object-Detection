#!/usr/bin/env python3
# Modified by RV-SDTM contributors for this public release; see NOTICE.
"""Complete-detector FP32 latency and peak allocated memory, excluding data loading."""

import argparse
import json
import statistics
import time
from pathlib import Path

import _init_path
import torch
from easydict import EasyDict

from pcdet.config import cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--frames', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--output', type=Path, help='optional new JSON file')
    parser.add_argument('--set', dest='set_cfgs', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.warmup < 0 or min(args.frames, args.runs) <= 0 or args.workers < 0:
        parser.error('warmup/workers must be nonnegative; frames/runs must be positive')
    if not args.ckpt.is_file():
        parser.error('Checkpoint does not exist: {}'.format(args.ckpt))
    if args.output and args.output.exists():
        parser.error('Output already exists: {}'.format(args.output))
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    config = cfg_from_yaml_file(args.cfg_file, EasyDict())
    if args.set_cfgs:
        cfg_from_list(args.set_cfgs, config)
    for processor in config.DATA_CONFIG.DATA_PROCESSOR:
        if processor.NAME == 'transform_points_to_voxels':
            raise ValueError('Use dynamic model-side voxelization so voxelization is inside the timed boundary')
    common_utils.set_random_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logger = common_utils.create_logger()
    dataset, loader, _ = build_dataloader(
        dataset_cfg=config.DATA_CONFIG, class_names=config.CLASS_NAMES,
        batch_size=1, dist=False, workers=args.workers, logger=logger,
        training=False, seed=args.seed)
    if len(loader) < args.warmup + args.frames:
        raise ValueError('Dataset must contain at least warmup + frames distinct samples')
    model = build_network(config.MODEL, len(config.CLASS_NAMES), dataset).cuda().float().eval()
    model.load_params_from_file(str(args.ckpt), logger, to_cpu=True, strict=True)
    results = []
    for run in range(args.runs):
        common_utils.set_random_seed(args.seed)
        iterator = iter(loader)
        latencies = []
        for index in range(args.warmup + args.frames):
            batch = next(iterator)
            batch.pop('gt_boxes', None)
            load_data_to_gpu(batch)
            torch.cuda.synchronize()
            if index == args.warmup:
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                prediction = model(batch)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            if index >= args.warmup:
                latencies.append(elapsed_ms)
            del prediction, batch
        results.append({'run': run + 1, 'mean_latency_ms': statistics.mean(latencies),
                        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3})
        del iterator
    report = {
        'config': args.cfg_file, 'checkpoint': str(args.ckpt),
        'device': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda, 'batch_size': 1, 'precision': 'FP32', 'tf32': False,
        'warmup_frames_per_run': args.warmup, 'measured_frames_per_run': args.frames,
        'seed': args.seed, 'parameters': sum(p.numel() for p in model.parameters()),
        'boundary': 'model-side voxelization, RV encoder, backbone, head, decoding and NMS; excludes loading and H2D',
        'runs': results,
        'mean_latency_ms': statistics.mean(r['mean_latency_ms'] for r in results),
        'peak_allocated_gib': max(r['peak_allocated_gib'] for r in results),
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as output:
            output.write(serialized)


if __name__ == '__main__':
    main()
