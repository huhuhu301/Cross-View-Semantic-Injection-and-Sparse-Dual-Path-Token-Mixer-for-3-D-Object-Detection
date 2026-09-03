# Modified by RV-SDTM contributors for this public release; see NOTICE.
import importlib
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate


__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'NuScenesDataset': None,
    'WaymoDataset': None,
    'Argo2Dataset': None,
}

_DATASET_IMPORTS = {
    'NuScenesDataset': ('.nuscenes.nuscenes_dataset', 'NuScenesDataset'),
    'WaymoDataset': ('.waymo.waymo_dataset', 'WaymoDataset'),
    'Argo2Dataset': ('.argo2.argo2_dataset', 'Argo2Dataset'),
}


def _get_dataset_class(dataset_name):
    if dataset_name not in _DATASET_IMPORTS:
        raise KeyError(
            'Unsupported dataset {!r}; available datasets are {}'.format(
                dataset_name, sorted(_DATASET_IMPORTS)
            )
        )
    dataset_class = __all__[dataset_name]
    if dataset_class is None:
        module_name, class_name = _DATASET_IMPORTS[dataset_name]
        try:
            module = importlib.import_module(module_name, package=__name__)
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                '{} dependencies are unavailable; install requirements.txt '
                'or the matching dataset requirement file'.format(
                    dataset_name
                )
            ) from error
        dataset_class = getattr(module, class_name)
        __all__[dataset_name] = dataset_class
    return dataset_class


def __getattr__(name):
    if name in _DATASET_IMPORTS:
        return _get_dataset_class(name)
    raise AttributeError("module {!r} has no attribute {!r}".format(
        __name__, name
    ))


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = _get_dataset_class(dataset_cfg.DATASET)(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )

    return dataset, dataloader, sampler
