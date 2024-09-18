# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from typing import Iterator, List, Optional, Sized, Union

import numpy as np
import torch 
from mmdet.core.utils import sync_random_seed
from mmcv.runner import get_dist_info
from torch.utils.data.sampler import Sampler

class InfiniteGroupBatchSamplercp(Sampler):
    """Similar to `BatchSampler` warping a `GroupSampler. It is designed for
    iteration-based runners like `IterBasedRunner` and yields a mini-batch
    indices each time, all indices in a batch should be in the same group.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (object): The dataset.
        batch_size (int): When model is :obj:`DistributedDataParallel`,
            it is the number of training samples on each GPU.
            When model is :obj:`DataParallel`, it is
            `num_gpus * samples_per_gpu`.
            Default : 1.
        world_size (int, optional): Number of processes participating in
            distributed training. Default: None.
        rank (int, optional): Rank of current process. Default: None.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the indices of a dummy `epoch`, it
            should be noted that `shuffle` can not guarantee that you can
            generate sequential indices because it need to ensure
            that all indices in a batch is in a group. Default: True.
    """  # noqa: W605

    def __init__(self,
                 dataset,
                 batch_size=1,
                 world_size=None,
                 rank=None,
                 seed=0,
                 shuffle=True):
        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = sync_random_seed(seed)
        self.shuffle = shuffle

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        # buffer used to save indices of each group
        self.buffer_per_group = {k: [] for k in range(len(self.group_sizes))}

        self.size = len(dataset)
        self.indices = self._indices_of_rank()

    def _infinite_indices(self):
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()

            else:
                yield from torch.arange(self.size).tolist()

    def _indices_of_rank(self):
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), self.rank, None,
                                    self.world_size)

    def __iter__(self):
        # once batch size is reached, yield the indices
        group_buffer = [5,30]
        for idx in self.indices:
            flag = self.flag[idx]
            group_buffer = self.buffer_per_group[flag]
            group_buffer.append(idx)
            if len(group_buffer) == self.batch_size:
                yield group_buffer[:]
                del group_buffer[:]
            yield group_buffer

    def __len__(self):
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch):
        """Not supported in `IterationBased` runner."""
        raise NotImplementedError


class InfiniteBatchSamplercp(Sampler):
    """Similar to `BatchSampler` warping a `DistributedSampler. It is designed
    iteration-based runners like `IterBasedRunner` and yields a mini-batch
    indices each time.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (object): The dataset.
        batch_size (int): When model is :obj:`DistributedDataParallel`,
            it is the number of training samples on each GPU,
            When model is :obj:`DataParallel`, it is
            `num_gpus * samples_per_gpu`.
            Default : 1.
        world_size (int, optional): Number of processes participating in
            distributed training. Default: None.
        rank (int, optional): Rank of current process. Default: None.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the dataset or not. Default: True.
    """  # noqa: W605

    def __init__(self,
                 dataset,
                 batch_size=1,
                 world_size=None,
                 rank=None,
                 seed=0,
                 shuffle=True):
        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.batch_size = batch_size
        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)
        self.shuffle = shuffle
        self.size = len(dataset)
        self.indices = self._indices_of_rank()

    def _infinite_indices(self):
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()

            else:
                yield from torch.arange(self.size).tolist()

    def _indices_of_rank(self):
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), self.rank, None,
                                    self.world_size)

    def __iter__(self):
        # once batch size is reached, yield the indices
        batch_buffer = []
        for idx in self.indices:
            batch_buffer.append(idx)
            if len(batch_buffer) == self.batch_size: 
                assert False, batch_buffer
                yield batch_buffer
                batch_buffer = []

    def __len__(self):
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch):
        """Not supported in `IterationBased` runner."""
        raise NotImplementedError

 
class MultiSourceSampler(Sampler):
    r"""Multi-Source Infinite Sampler.

    According to the sampling ratio, sample data from different
    datasets to form batches.

    Args:
        dataset (Sized): The dataset.
        batch_size (int): Size of mini-batch.
        source_ratio (list[int | float]): The sampling ratio of different
            source datasets in a mini-batch.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.

    Examples:
        >>> dataset_type = 'ConcatDataset'
        >>> sub_dataset_type = 'CocoDataset'
        >>> data_root = 'data/coco/'
        >>> sup_ann = '../coco_semi_annos/instances_train2017.1@10.json'
        >>> unsup_ann = '../coco_semi_annos/' \
        >>>             'instances_train2017.1@10-unlabeled.json'
        >>> dataset = dict(type=dataset_type,
        >>>     datasets=[
        >>>         dict(
        >>>             type=sub_dataset_type,
        >>>             data_root=data_root,
        >>>             ann_file=sup_ann,
        >>>             data_prefix=dict(img='train2017/'),
        >>>             filter_cfg=dict(filter_empty_gt=True, min_size=32),
        >>>             pipeline=sup_pipeline),
        >>>         dict(
        >>>             type=sub_dataset_type,
        >>>             data_root=data_root,
        >>>             ann_file=unsup_ann,
        >>>             data_prefix=dict(img='train2017/'),
        >>>             filter_cfg=dict(filter_empty_gt=True, min_size=32),
        >>>             pipeline=unsup_pipeline),
        >>>         ])
        >>>     train_dataloader = dict(
        >>>         batch_size=5,
        >>>         num_workers=5,
        >>>         persistent_workers=True,
        >>>         sampler=dict(type='MultiSourceSampler',
        >>>             batch_size=5, source_ratio=[1, 4]),
        >>>         batch_sampler=None,
        >>>         dataset=dataset)
    """

    def __init__(self,
                 dataset: Sized,
                 batch_size: int,
                 source_ratio: List[Union[int, float]],
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:

        assert hasattr(dataset, 'cumulative_sizes'),\
            f'The dataset must be ConcatDataset, but get {dataset}'
        assert isinstance(batch_size, int) and batch_size > 0, \
            'batch_size must be a positive integer value, ' \
            f'but got batch_size={batch_size}'
        assert isinstance(source_ratio, list), \
            f'source_ratio must be a list, but got source_ratio={source_ratio}'
        assert len(source_ratio) == len(dataset.cumulative_sizes), \
            'The length of source_ratio must be equal to ' \
            f'the number of datasets, but got source_ratio={source_ratio}'

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.cumulative_sizes = [0] + dataset.cumulative_sizes
        self.batch_size = batch_size
        self.source_ratio = source_ratio

        self.num_per_source = [
            int(batch_size * sr / sum(source_ratio)) for sr in source_ratio
        ]
        self.num_per_source[0] = batch_size - sum(self.num_per_source[1:])

        assert sum(self.num_per_source) == batch_size, \
            'The sum of num_per_source must be equal to ' \
            f'batch_size, but get {self.num_per_source}'

        self.seed = sync_random_seed() if seed is None else seed
        self.shuffle = shuffle
        self.source2inds = {
            source: self._indices_of_rank(len(ds))
            for source, ds in enumerate(dataset.datasets)
        }

    def _infinite_indices(self, sample_size: int) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(sample_size, generator=g).tolist()
            else:
                yield from torch.arange(sample_size).tolist()

    def _indices_of_rank(self, sample_size: int) -> Iterator[int]:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(
            self._infinite_indices(sample_size), self.rank, None,
            self.world_size)


    def __iter__(self) -> Iterator[int]:
        batch_buffer = []
        while True:
            for source, num in enumerate(self.num_per_source):
                batch_buffer_per_source = []
                for idx in self.source2inds[source]:
                    idx += self.cumulative_sizes[source]
                    batch_buffer_per_source.append(idx)
                    if len(batch_buffer_per_source) == num:
                        batch_buffer += batch_buffer_per_source
                        break
            # assert False, batch_buffer
            yield from [batch_buffer]
            batch_buffer = []

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        """Not supported in `epoch-based runner."""
        pass

 
class GroupMultiSourceSampler(MultiSourceSampler):
    r"""Group Multi-Source Infinite Sampler.

    According to the sampling ratio, sample data from different
    datasets but the same group to form batches.

    Args:
        dataset (Sized): The dataset.
        batch_size (int): Size of mini-batch.
        source_ratio (list[int | float]): The sampling ratio of different
            source datasets in a mini-batch.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.
    """

    def __init__(self,
                 dataset,
                 batch_size: int,
                 source_ratio: List[Union[int, float]],
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            source_ratio=source_ratio,
            shuffle=shuffle,
            seed=seed)

        self._get_source_group_info()
        self.group_source2inds = [{
            source:
            self._indices_of_rank(self.group2size_per_source[source][group])
            for source in range(len(dataset.datasets))
        } for group in range(len(self.group_ratio))]

    def _get_source_group_info(self) -> None:
        self.group2size_per_source = [{0: 0, 1: 0}, {0: 0, 1: 0}]
        self.group2inds_per_source = [{0: [], 1: []}, {0: [], 1: []}]
        for source, dataset in enumerate(self.dataset.datasets):
            for idx in range(len(dataset)):
                data_info = dataset.get_data_info(idx)
                width, height = data_info['width'], data_info['height']
                group = 0 if width < height else 1
                self.group2size_per_source[source][group] += 1
                self.group2inds_per_source[source][group].append(idx)

        self.group_sizes = np.zeros(2, dtype=np.int64)
        for group2size in self.group2size_per_source:
            for group, size in group2size.items():
                self.group_sizes[group] += size
        self.group_ratio = self.group_sizes / sum(self.group_sizes)

    def __iter__(self) -> Iterator[int]:
        batch_buffer = []
        while True:
            group = np.random.choice(
                list(range(len(self.group_ratio))), p=self.group_ratio)
            for source, num in enumerate(self.num_per_source):
                batch_buffer_per_source = []
                for idx in self.group_source2inds[group][source]:
                    idx = self.group2inds_per_source[source][group][
                        idx] + self.cumulative_sizes[source]
                    batch_buffer_per_source.append(idx)
                    if len(batch_buffer_per_source) == num:
                        batch_buffer += batch_buffer_per_source
                        break
            yield from batch_buffer
            batch_buffer = []
