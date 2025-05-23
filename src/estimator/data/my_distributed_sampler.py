# Copyright (c) Facebook, Inc. and its affiliates.
# modified from detectron2
import itertools
import math
from collections import defaultdict
from typing import Optional, Sized
import torch
from torch.utils.data.sampler import Sampler
from ..utils import common as comm
import random
import numpy as np


class TrainingSampler(Sampler):
    """In training, we only care about the "infinite stream" of training data.

    So this sampler produces an infinite stream of indices and all
    workers cooperate to correctly shuffle the indices and sample
    different indices. The samplers in each worker effectively produces
    `indices[worker_id::num_workers]` where `indices` is an infinite
    stream of indices consisting of `shuffle(range(size)) +
    shuffle(range(size)) + ...` (if shuffle is True) or `range(size) +
    range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                # print(torch.randperm(self._size, generator=g))
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)
                

class TrainingSampler_Pair(Sampler):
    """In training, we only care about the "infinite stream" of training data.

    So this sampler produces an infinite stream of indices and all
    workers cooperate to correctly shuffle the indices and sample
    different indices. The samplers in each worker effectively produces
    `indices[worker_id::num_workers]` where `indices` is an infinite
    stream of indices consisting of `shuffle(range(size)) +
    shuffle(range(size)) + ...` (if shuffle is True) or `range(size) +
    range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size: int, list_scene_num, interval = 5, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        self.list_scene_num = list_scene_num
        self.interval = interval
        assert size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        for num in self.list_scene_num:
            if num%2 != 0:
                self._size += 1
        

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
            # print(1)
            random.seed(self._seed)
            while True:
                indices = []
                scene_id = 0
                scene_last_num = 0
                scene_num = 0
                for num in self.list_scene_num:
                    scene_num += num
                    scene_sampler = [j for j in range(scene_last_num, scene_num)]
                    random.shuffle(scene_sampler)
                    if len(scene_sampler)%2 != 0:
                        scene_sampler.append(random.randrange(scene_last_num, scene_num))
                    scene_last_num = scene_num
                    for i in range(0, len(scene_sampler), 2):
                        pair = [scene_sampler[i], scene_sampler[i+1]]
                        indices.append(pair)
                    
                # for i in range(0, self._size, self.interval):
                #     if i >=  scene_num:
                #         scene_id +=1
                #         scene_last_num = scene_num
                #         scene_num += self.list_scene_num[scene_id]
                #     scene_sampler = [j for j in range(scene_last_num, scene_num)]
                #     print(scene_sampler)
                    
                #     pair = []
                #     pair.append(i)
                #     if i < scene_num:
                #         id_insert = random.randrange(scene_last_num, scene_num)
                #     pair.append(id_insert)
                #     indices.append(pair)
                # exit()
                if self._shuffle:
                    random.shuffle(indices)
                    indices_shuffle = []
                    for indice in indices:
                        indices_shuffle.append(indice[0])
                        indices_shuffle.append(indice[1])
                    # print(indices)
                    # print(len(indices_shuffle))
                    # exit()
                    yield from indices_shuffle
                else:
                    indices_woshuffle = []
                    for indice in indices:
                        indices_woshuffle.append(indice[0])
                        indices_woshuffle.append(indice[1])
                    yield from indices_woshuffle   
  
        
# class TrainingSampler_Pair(Sampler):
#     """In training, we only care about the "infinite stream" of training data.

#     So this sampler produces an infinite stream of indices and all
#     workers cooperate to correctly shuffle the indices and sample
#     different indices. The samplers in each worker effectively produces
#     `indices[worker_id::num_workers]` where `indices` is an infinite
#     stream of indices consisting of `shuffle(range(size)) +
#     shuffle(range(size)) + ...` (if shuffle is True) or `range(size) +
#     range(size) + ...` (if shuffle is False)
#     """

#     def __init__(self, size: int, list_scene_num, interval = 5, shuffle: bool = True, seed: Optional[int] = None):
#         """
#         Args:
#             size (int): the total number of data of the underlying dataset to sample from
#             shuffle (bool): whether to shuffle the indices or not
#             seed (int): the initial seed of the shuffle. Must be the same
#                 across all workers. If None, will use a random seed shared
#                 among workers (require synchronization among all workers).
#         """
#         self._size = size
#         self.list_scene_num = list_scene_num
#         self.interval = interval
#         assert size > 0
#         self._shuffle = shuffle
#         if seed is None:
#             seed = comm.shared_random_seed()
#         self._seed = int(seed)

#         self._rank = comm.get_rank()
#         self._world_size = comm.get_world_size()
        

#     def __iter__(self):
#         start = self._rank
#         yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

#     def _infinite_indices(self):
#             # print(1)
#             random.seed(self._seed)
#             while True:
#                 indices = []
#                 scene_id = 0
#                 scene_last_num = 0
#                 scene_num = self.list_scene_num[scene_id]
#                 for i in range(0, self._size, self.interval):
#                     if i >=  scene_num:
#                         scene_id +=1
#                         scene_last_num = scene_num
#                         scene_num += self.list_scene_num[scene_id]
#                     pair = []
#                     pair.append(i)
#                     if i < scene_num:
#                         id_insert = random.randrange(scene_last_num, scene_num)
#                     pair.append(id_insert)
#                     indices.append(pair)
#                 if self._shuffle:
#                     random.shuffle(indices)
#                     indices_shuffle = []
#                     for indice in indices:
#                         indices_shuffle.append(indice[0])
#                         indices_shuffle.append(indice[1])
#                     # print(indices)
#                     print(indices_shuffle)
#                     exit()
#                     yield from indices_shuffle
#                 else:
#                     indices_woshuffle = []
#                     for indice in indices:
#                         indices_woshuffle.append(indice[0])
#                         indices_woshuffle.append(indice[1])
#                     yield from indices_woshuffle


class TrainingSampler_Pair0(Sampler):
    '''During the traing stage, we use a pair of images. Meanwhile, each pair must within the same
     scene.
    '''
    def __init__(self, size: int, list_scene_num, shuffle: bool = True, seed: Optional[int] = None) :
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self.list_scene_num = list_scene_num
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()


    def __iter__(self):
            start = self._rank
            yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices1(self):
            g = torch.Generator()
            g.manual_seed(self._seed)
            while True:
                if self._shuffle:
                    # print(torch.randperm(self._size, generator=g))
                    yield from torch.randperm(self._size, generator=g)
                else:
                    yield from torch.arange(self._size)
    def _infinite_indices(self):
            print(1)
            random.seed(self._seed)
            while True:
                indices = []
                scene_id = 0
                scene_last_num = 0
                scene_num = self.list_scene_num[scene_id]
                for i in range(size):
                    if i >=  scene_num:
                        scene_id +=1
                        scene_last_num = scene_num
                        scene_num += self.list_scene_num[scene_id]
                    pair = []
                    pair.append(i)
                    if i < scene_num:
                        id_insert = random.randrange(scene_last_num, scene_num)
                    pair.append(id_insert)
                    indices.append(pair)
                if self._shuffle:
                    indices = random.shuffle(indices)
                    indices_shuffle = []
                    for indice in indices:
                        indices_shuffle.append(indice[0])
                        indices_shuffle.append(indice[1])
                    print(indices)
                    print(indices_shuffle)
                    yield from indices_shuffle
                else:
                    indices_woshuffle = []
                    for indice in indices:
                        indices_woshuffle.append(indice[0])
                        indices_woshuffle.append(indice[1])
                    yield from indices_woshuffle

    

class RepeatFactorTrainingSampler(Sampler):
    """Similar to TrainingSampler, but a sample may appear more times than
    others based on its "repeat factor".

    This is suitable for training on class imbalanced datasets like
    LVIS.
    """

    def __init__(self, repeat_factors, *, shuffle=True, seed=None):
        """
        Args:
            repeat_factors (Tensor): a float vector, the repeat factor for each indice. When it's
                full of ones, it is equivalent to ``TrainingSampler(len(repeat_factors), ...)``.
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        # Split into whole number (_int_part) and fractional (_frac_part) parts.
        self._int_part = torch.trunc(repeat_factors)
        self._frac_part = repeat_factors - self._int_part

    @staticmethod
    def repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh):
        """Compute (fractional) per-image repeat factors based on category
        frequency.

        The repeat factor for an image is a function of the frequency of the rarest
        category labeled in that image. The "frequency of category c" in [0, 1] is defined
        as the fraction of images in the training set (without repeats) in which category c
        appears.
        See :paper:`lvis` (>= v2) Appendix B.2.
        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
                If the frequency is half of `repeat_thresh`, the image will be
                repeated twice.
        Returns:
            torch.Tensor: the i-th element is the repeat factor for the dataset image
                at index i.
        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        num_images = len(dataset_dicts)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq)) for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids})
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)

    def _get_epoch_indices(self, generator):
        """Create a list of dataset indices (with repeats) to use for one
        epoch.

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.
        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            indices = self._get_epoch_indices(g)
            if self._shuffle:
                randperm = torch.randperm(len(indices), generator=g)
                yield from indices[randperm]
            else:
                yield from indices


class InferenceSampler(Sampler):
    """Produce indices for inference.

    Inference needs to run on the __exact__ set of samples, therefore
    when the total number of samples is not divisible by the number of
    workers, this sampler produces different number of samples on
    different workers.
    """

    def __init__(self, size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        shard_size = (self._size - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._size)
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
