#! /usr/bin/env python3
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for PyTorch."""

import torch
import torch.nn.functional as F
import numpy as np 

from jactorch.utils.meta import as_tensor, as_float, as_cpu
from IPython.core.debugger import Pdb
__all__ = [
    'binary_accuracy', 'rms', 'monitor_saturation', 'monitor_paramrms',
    'monitor_gradrms'
]


def is_safe_nqueens(grid):
    size = int(len(grid)**0.5)

    grid = grid.reshape(size, size)
    indices = torch.nonzero(grid)
    if len(indices) != size:
        return False
    for x in range(size):
        r1, c1 = indices[x]
        for y in range(x+1, size):
            r2, c2 = indices[y]
            if (r1 == r2) or (c1 == c2) or (torch.abs(r1-r2) == torch.abs(c1-c2)):
                return False
    return True

def check_validity(grid, constraints=None):
    grid = grid.cpu().numpy()
    constraints = constraints.cpu().numpy()
    grid = grid.argmax(axis=2)
    for x in range(len(grid)):
        row = set(grid[x])
        if len(row)!=len(grid):
            return False
        col = set(grid[:,x])
        if len(col)!=len(grid):
            return False
    if constraints is None:
        return True
    gt = zip(*np.nonzero(constraints[0]))
    for ind in gt:
        next_ind = (ind[0],ind[1]+1)
        if grid[next_ind]>grid[ind]:
            return False
    lt = zip(*np.nonzero(constraints[1]))
    for ind in lt:
        next_ind = (ind[0],ind[1]+1)
        if grid[next_ind]<grid[ind]:
            return False
    return True 

def is_safe_futoshiki(grid,constraints):
    size = int(len(grid)**0.3334)
    grid = grid.reshape(size,size,size).float()
    gold = torch.ones(size,size).cuda()
    if torch.sum(torch.abs(grid.sum(dim=0)-gold))>0:
        return False
    if torch.sum(torch.abs(grid.sum(dim=1)-gold))>0:
        return False
    if torch.sum(torch.abs(grid.sum(dim=2)-gold))>0:
        return False
     
    constraints = constraints.transpose(0,1)
    constraints = constraints.reshape(2,size,size,size)
    constraints = constraints[:,:,:,0]
    return check_validity(grid,constraints)

def instance_accuracy(label, raw_pred, return_float=True, correct_accuracy=True, feed_dict=None, pred_aux=None, check_futo=False):
    if check_futo==False:
        return instance_accuracy_nqueens(label,raw_pred, return_float, feed_dict, pred_aux)
    else:
        return instance_accuracy_futoshiki(label,raw_pred, return_float, feed_dict, pred_aux)

def instance_accuracy_futoshiki(label, raw_pred, return_float=True, feed_dict=None, pred_aux=None):
    """get instance-wise accuracy for structured prediction task instead of pointwise task"""
    pred = as_tensor(raw_pred)
    pred = (pred > 0.5).float()

    label = as_tensor(label).float()
    diff = torch.abs(label-pred)
    point_acc = 1 - torch.sum(diff)/label.numel()
    incorrect_count = torch.sum(diff, dim=1)
    incorrect = len(torch.nonzero(incorrect_count))

    in_acc = 1-incorrect/len(label)

    errors = []
    corrected_acc = 0
    for i, x in enumerate(pred):
        constraints = feed_dict["query"][i][:,1:]
        if is_safe_futoshiki(x,constraints):
            corrected_acc += 1
        else:
            errors.append(feed_dict["count"][i].item())
    corrected_acc /= len(pred)

    if pred_aux is not None:
        pred_aux = (pred_aux > 0.5).float()
        classification_acc = 1 - \
            torch.sum(
                torch.abs(pred_aux-feed_dict["is_ambiguous"].float()))/len(pred_aux)
    else:
        classification_acc = torch.zeros(1)

    if return_float:
        return {"accuracy": in_acc,
                "corrected accuracy": corrected_acc,
                "pointwise accuracy": point_acc.item(),
                "classification accuracy": classification_acc.item()}, errors 
    return {"accuracy": torch.tensor(in_acc),
            "corrected accuracy": torch.tensor(corrected_acc),
            "pointwise accuracy": point_acc,
            "classification accuracy": classification_acc}, errors

def instance_accuracy_nqueens(label, raw_pred, return_float=True, feed_dict=None, pred_aux=None):
    """get instance-wise accuracy for structured prediction task instead of pointwise task"""
    pred = as_tensor(raw_pred)
    pred = (pred > 0.5).float()

    label = as_tensor(label).float()
    diff = torch.abs(label-pred)
    point_acc = 1 - torch.sum(diff)/label.numel()
    incorrect_count = torch.sum(diff, dim=1)
    incorrect = len(torch.nonzero(incorrect_count))

    in_acc = 1-incorrect/len(label)

    errors = []
    corrected_acc = 0
    for i, x in enumerate(pred):
        if is_safe_nqueens(x):
            corrected_acc += 1
        else:
            errors.append(feed_dict["count"][i].item())
    corrected_acc /= len(pred)

    if pred_aux is not None:
        pred_aux = (pred_aux > 0.5).float()
        classification_acc = 1 - \
            torch.sum(
                torch.abs(pred_aux-feed_dict["is_ambiguous"].float()))/len(pred_aux)
    else:
        classification_acc = torch.zeros(1)

    if return_float:
        return {"accuracy": in_acc,
                "corrected accuracy": corrected_acc,
                "pointwise accuracy": point_acc.item(),
                "classification accuracy": classification_acc.item()}, errors 
    return {"accuracy": torch.tensor(in_acc),
            "corrected accuracy": torch.tensor(corrected_acc),
            "pointwise accuracy": point_acc,
            "classification accuracy": classification_acc}, errors


def binary_accuracy(label, raw_pred, eps=1e-20, return_float=True):
    """get accuracy for binary classification problem."""
    pred = as_tensor(raw_pred).squeeze(-1)
    pred = (pred > 0.5).float()
    label = as_tensor(label).float()
    # The $acc is micro accuracy = the correct ones / total
    acc = label.eq(pred).float()

    # The $balanced_accuracy is macro accuracy, with class-wide balance.
    nr_total = torch.ones(
        label.size(), dtype=label.dtype, device=label.device).sum(dim=-1)
    nr_pos = label.sum(dim=-1)
    nr_neg = nr_total - nr_pos
    pos_cnt = (acc * label).sum(dim=-1)
    neg_cnt = acc.sum(dim=-1) - pos_cnt
    balanced_acc = ((pos_cnt + eps) / (nr_pos + eps) + (neg_cnt + eps) /
                    (nr_neg + eps)) / 2.0

    # $sat means the saturation rate of the predication,
    # measure how close the predections are to 0 or 1.
    sat = 1 - (raw_pred - pred).abs()
    if return_float:
        acc = as_float(acc.mean())
        balanced_acc = as_float(balanced_acc.mean())
        sat_mean = as_float(sat.mean())
        sat_min = as_float(sat.min())
    else:
        sat_mean = sat.mean(dim=-1)
        sat_min = sat.min(dim=-1)[0]

    return {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'satuation/mean': sat_mean,
        'satuation/min': sat_min,
    }


def rms(p):
    """Root mean square function."""
    return as_float((as_tensor(p)**2).mean()**0.5)


def monitor_saturation(model):
    """Monitor the saturation rate."""
    monitors = {}
    for name, p in model.named_parameters():
        p = F.sigmoid(p)
        sat = 1 - (p - (p > 0.5).float()).abs()
        monitors['sat/' + name] = sat
    return monitors


def monitor_paramrms(model):
    """Monitor the rms of the parameters."""
    monitors = {}
    for name, p in model.named_parameters():
        monitors['paramrms/' + name] = rms(p)
    return monitors


def monitor_gradrms(model):
    """Monitor the rms of the gradients of the parameters."""
    monitors = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            monitors['gradrms/' + name] = rms(p.grad) / max(rms(p), 1e-8)
    return monitors
