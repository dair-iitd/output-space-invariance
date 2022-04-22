#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : printing.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import io
import sys
import numpy as np
import collections

import threading
from .registry import LockRegistry

__all__ = ['stprint', 'stformat', 'kvprint', 'kvformat', 'PrintToStringContext', 'print_to_string', 'print2format']


def _indent_print(msg, indent, prefix=None, end='\n', file=None):
    print('  ' * indent, end='', file=file)
    if prefix is not None:
        print(prefix, end='', file=file)
    print(msg, end=end, file=file)


def stprint(data, key=None, indent=0, file=None, need_lock=True, max_depth=100):
    """
    Structure print.

    Example:

        >>> data = dict(a=np.zeros(shape=(10, 10)), b=3)
        >>> stprint(data)
        dict{
            a: ndarray(10, 10), dtype=float64
            b: 3
        }

    Args:
        data: data to be print. Currently support Sequnce, Mappings and primitive types.
        key: for recursion calls. Do not use it if you don't know how it works.
        indent: indent level.
    """
    t = type(data)
    if file is None:
        file = sys.stdout

    with stprint.locks.synchronized(file, need_lock):
        if t is tuple:
            if max_depth == 0:
                _indent_print('(tuple of length {}) ...'.format(len(data)), indent, prefix=key, file=file)
                return
            _indent_print('tuple[', indent, prefix=key, file=file)
            for v in data:
                stprint(v, indent=indent + 1, file=file, need_lock=False, max_depth=max_depth - 1)
            _indent_print(']', indent, file=file)
        elif t is list:
            if max_depth == 0:
                _indent_print('(list of length {}) ...'.format(len(data)), indent, prefix=key, file=file)
                return
            _indent_print('list[', indent, prefix=key, file=file)
            for v in data:
                stprint(v, indent=indent + 1, file=file, need_lock=False, max_depth=max_depth - 1)
            _indent_print(']', indent, file=file)
        elif t in (dict, collections.OrderedDict):
            if max_depth == 0:
                _indent_print('(dict of length {}) ...'.format(len(data)), indent, prefix=key, file=file)
                return
            typename = 'dict' if t is dict else 'ordered_dict'
            keys = sorted(data.keys()) if t is dict else data.keys()
            _indent_print(typename + '{', indent, prefix=key, file=file)
            for k in keys:
                v = data[k]
                stprint(v, indent=indent + 1, key='{}: '.format(k), file=file, need_lock=False, max_depth=max_depth - 1)
            _indent_print('}', indent, file=file)
        elif t is np.ndarray:
            _indent_print('ndarray{}, dtype={}'.format(data.shape, data.dtype), indent, prefix=key, file=file)
        else:
            _indent_print(data, indent, prefix=key, file=file)


stprint.locks = LockRegistry()


def stformat(data, key=None, indent=0, max_depth=100):
    return print2format(stprint)(data, key=key, indent=indent, need_lock=False, max_depth=max_depth)


def kvprint(data, indent=0, sep=' : ', end='\n', max_key_len=None, file=None, need_lock=True):
    if len(data) == 0:
        return
    with kvprint.locks.synchronized(file, need_lock):
        keys = sorted(data.keys())
        lens = list(map(len, keys))
        if max_key_len is not None:
            max_len = max_key_len
        else:
            max_len = max(lens)
        for k in keys:
            print('  ' * indent, end='')
            print(k + ' ' * (max_len - len(k)), data[k], sep=sep, end=end, file=file, flush=True)


kvprint.locks = LockRegistry()


def kvformat(data, indent=0, sep=' : ', end='\n', max_key_len=None):
    return print2format(kvprint)(data, indent=indent, sep=sep, end=end, max_key_len=max_key_len, need_lock=False)


class PrintToStringContext(object):
    __global_locks = LockRegistry()

    def __init__(self, target='STDOUT', stream=None, need_lock=True):
        assert target in ('STDOUT', 'STDERR')
        self._target = target
        self._need_lock = need_lock
        if stream is None:
            self._stream = io.StringIO()
        else:
            self._stream = stream
        self._stream_lock = threading.Lock()
        self._backup = None
        self._value = None

    def _swap(self, rhs):
        if self._target == 'STDOUT':
            sys.stdout, rhs = rhs, sys.stdout
        else:
            sys.stderr, rhs = rhs, sys.stderr

        return rhs

    def __enter__(self):
        if self._need_lock:
            self.__global_locks[self._target].acquire()
        self._backup = self._swap(self._stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stream = self._swap(self._backup)
        if self._need_lock:
            self.__global_locks[self._target].release()

    def _ensure_value(self):
        if self._value is None:
            self._value = self._stream.getvalue()
            self._stream.close()

    def get(self):
        self._ensure_value()
        return self._value


def print_to_string(target='STDOUT'):
    return PrintToStringContext(target, need_lock=True)


def print2format(print_func):
    def format_func(*args, **kwargs):
        f = io.StringIO()
        print_func(*args, file=f, **kwargs)
        value = f.getvalue()
        f.close()
        return value
    return format_func
