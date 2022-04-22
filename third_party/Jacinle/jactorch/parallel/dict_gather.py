#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dict_gather.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import six
import functools
import collections

from torch.autograd import Variable
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel._functions import Gather


__all__ = [
    'data_parallel_dict_gather',
    'DictGatherDataParallel',
    'patch_dict_gathering'
]


def data_parallel_dict_gather(data_parallel, outputs, output_device):
    return dict_gather(outputs, output_device, dim=data_parallel.dim)


class DictGatherDataParallel(DataParallel):
    """Add support for modules that return dicts."""
    def gather(self, outputs, output_device):
        return data_parallel_dict_gather(self, outputs, output_device)


def patch_dict_gathering(data_parallel):
    assert isinstance(data_parallel, DataParallel)
    data_parallel.gather = functools.partial(data_parallel_dict_gather, data_parallel=data_parallel)


def dict_gather(outputs, target_device, dim=0):
    """
    Gathers variables from different GPUs on a specified device
      (-1 means the CPU), with dictionary support.
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, Variable):
            if out.dim() == 0:
                outputs = [o.unsqueeze(0) for o in outputs]
            return Gather.apply(target_device, dim, *outputs)
        elif out is None:
            return None
        elif isinstance(out, collections.Mapping):
            return {k: gather_map([o[k] for o in outputs]) for k in out}
        elif isinstance(out, six.string_types):
            return outputs
        elif isinstance(out, collections.Sequence):
            return type(out)(map(gather_map, zip(*outputs)))
        return outputs
    return gather_map(outputs)
