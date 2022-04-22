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
import math
from jactorch.utils.meta import as_tensor, as_float, as_cpu
from IPython.core.debugger import Pdb
__all__ = [
    'binary_accuracy', 'rms', 'monitor_saturation', 'monitor_paramrms',
    'monitor_gradrms'
]

def match_query_batch(query, pred):
    mask = (query  == 0)
    dif = (query - pred).abs()
    dif[mask] = 0
    return (dif.sum(dim=-1) == 0) 

def match_gcp_query_batch(query, pred, mask_nodes):
    
    mask = (query  == 0)
    dif = query - pred
    dif[mask] = 0
    dif[~mask_nodes] = 0
    return (dif.sum(dim=-1) == 0) 

def match_query(query, pred):
    mask = (query>0)
    return torch.equal(query[mask], pred[mask])

 

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

def is_safe_sudoku_tensor_batch(x,n):
    grid = x.detach().long()
    batch_size, n2 = grid.size(0), grid.size(-1)
    n = int(math.sqrt(n2))
    grid = grid.reshape(batch_size,n,n)
    ind_row_diff = torch.zeros(batch_size, n,n+1, device=x.device)
    
    def check_all_diff_each_row(a):
        ind_row_diff.fill_(0)
        c = torch.arange(a.size(0)).unsqueeze(-1).unsqueeze(-1).expand_as(a).long()
        b = torch.arange(a.size(1)).unsqueeze(0).unsqueeze(-1).expand_as(a).long()
        ind_row_diff[c,b,a.long()] = 1
        all_diff = (ind_row_diff.sum(dim=-1) == a.size(2)).all(dim=1)
        no_zero = (ind_row_diff[:,:,0].sum(dim=-1) == 0)
        return (all_diff & no_zero)

    block_shape_dict = {6: (2, 3),
                     8: (2, 4),
                     9: (3, 3),
                     10: (2, 5),
                     12: (2, 6),
                     14: (2, 7),
                     15: (3, 5),
                     16: (4, 4)}

    
    
    b_x, b_y = block_shape_dict[n]

    check_row = check_all_diff_each_row(grid)

    check_column = check_all_diff_each_row(grid.transpose(-1,-2))

    all_blocks = []
    for i in range(n):
        b_row = i//b_x 
        b_col = i%b_x
        all_blocks.append(grid[:,b_x*b_row:b_x*(b_row+1),b_y*b_col:b_y*(b_col+1)].reshape(batch_size,-1))
    #
    check_blocks = check_all_diff_each_row(torch.stack(all_blocks,dim=-1).transpose(-1,-2))
    return (check_row & check_column & check_blocks)



def is_safe_sudoku_tensor(x,n):
    grid = x.detach().int()
    n = int(math.sqrt(grid.size(0)))
    grid = grid.reshape(n,n)
    ind_row_diff = torch.zeros(n,n+1, device=x.device)
    
    def check_all_diff_each_row(a):
        ind_row_diff.fill_(0)
        b = torch.arange(a.size(0)).unsqueeze(-1).expand_as(a).long()
        ind_row_diff[b,a.long()] = 1
        all_diff = (ind_row_diff.sum(dim=1) == a.size(1)).all()
        no_zero = (ind_row_diff[:,0].sum() == 0)
        return all_diff and no_zero

    block_shape_dict = {6: (2, 3),
                     8: (2, 4),
                     9: (3, 3),
                     10: (2, 5),
                     12: (2, 6),
                     14: (2, 7),
                     15: (3, 5),
                     16: (4, 4)}

    b_x, b_y = block_shape_dict[n]

    if not check_all_diff_each_row(grid):
        return False

    if not check_all_diff_each_row(grid.transpose(-1,-2)):
        return False

    all_blocks = []
    for i in range(n):
        b_row = i//b_x 
        b_col = i%b_x
        all_blocks.append(grid[b_x*b_row:b_x*(b_row+1),b_y*b_col:b_y*(b_col+1)].flatten())
    #
    if not check_all_diff_each_row(torch.stack(all_blocks)):
        return False
    #
    return True




def is_safe_sudoku(x,n):
    block_shape_dict = {6: (2, 3),
                     8: (2, 4),
                     9: (3, 3),
                     10: (2, 5),
                     12: (2, 6),
                     14: (2, 7),
                     15: (3, 5),
                     16: (4, 4)}

    
    grid = x.detach().cpu().numpy().astype(int)
    n = int(math.sqrt(grid.size))
    grid = grid.reshape(n,n)
    
    b_x, b_y = block_shape_dict[n]

    for i in range(n):
        if len(set(grid[i]))<n:
            return False 
        if len(set(grid[:,i]))<n:
            return False 

        b_row = i//b_x 
        b_col = i%b_x
#         if n!=9:
#             Pdb().set_trace()
        if len(set(grid[b_x*b_row:b_x*(b_row+1),b_y*b_col:b_y*(b_col+1)].flatten()))<n:
            return False 
    return True

def is_colored_gcp_batch(x,ed):
   
#     Pdb().set_trace()
    grid = x.detach().long()
    edges = ed.detach().long()
    
    batch_size, num_edges = edges.shape[0], edges.shape[1]
    cycle_edges = (edges[:,:,0]==edges[:,:,1])
    
    grid_idx1 = torch.arange(batch_size).repeat_interleave(num_edges*2)
    grid_idx2 = edges.view(grid_idx1.shape[0])
    
    color_edges = grid[grid_idx1, grid_idx2].view(edges.shape)
    valid_edges = (color_edges[:,:,0]!=color_edges[:,:,1])
    
    graph_check = (valid_edges | cycle_edges).all(dim=1)
    return graph_check

def is_safe_futo_batch(x, edges):
    
    x_copy = x.detach().long()
    grid = x.detach().long()
    batch_size, n2 = grid.size(0), grid.size(-1)
    n = int(math.sqrt(n2))
    grid = grid.reshape(batch_size,n,n)
    ind_row_diff = torch.zeros(batch_size, n,n+1, device=x.device)
    
    def check_all_diff_each_row(a):
        #Pdb().set_trace()
        ind_row_diff.fill_(0)
        c = torch.arange(a.size(0)).unsqueeze(-1).unsqueeze(-1).expand_as(a).long()
        b = torch.arange(a.size(1)).unsqueeze(0).unsqueeze(-1).expand_as(a).long()
        ind_row_diff[c,b,a.long()] = 1
        all_diff = (ind_row_diff.sum(dim=-1) == a.size(2)).all(dim=1)
        no_zero = (ind_row_diff[:,:,0].sum(dim=-1) == 0)
        return (all_diff & no_zero)
    
    # Check all rows and columns are fine
    check_row = check_all_diff_each_row(grid)
    check_column = check_all_diff_each_row(grid.transpose(-1,-2))
    
    # Honour the less-than constraints
    edges = edges.detach().long()
    num_edges = edges.shape[1]
    cycle_edges = (edges[:,:,0]==edges[:,:,1])
    
    grid_idx1 = torch.arange(batch_size).repeat_interleave(num_edges*2)
    grid_idx2 = edges.view(grid_idx1.shape[0])
        
    cons_edges = x_copy[grid_idx1, grid_idx2].view(edges.shape)
    valid_edges = (cons_edges[:,:,0]<cons_edges[:,:,1])
    
    lt_check = (valid_edges | cycle_edges).all(dim=1)
    return lt_check & check_row & check_column
 
    

def instance_accuracy(label, raw_pred, return_float=True, feed_dict=None, task='gcp',args=None):
    with torch.no_grad():
        # query doesn't have to be matched for towers
        
        compare_func_gcp = lambda x,query,edges,mask_nodes: match_gcp_query_batch(query,x,mask_nodes) & is_colored_gcp_batch(x, edges)
        compare_func_futo = lambda x,query,lt_edges: match_query_batch(query,x) & is_safe_futo_batch(x, lt_edges)
        
#         compare_func_gcp = lambda x,query,edges: is_colored_gcp_batch(x, edges)

        compare_func = None 
        if task=='futoshiki':
            compare_func = compare_func_futo
        elif task in ['gcp', 'sudoku']:
            compare_func = compare_func_gcp
        # 
        return _instance_accuracy(label,raw_pred, compare_func, return_float, feed_dict,args)


def _instance_accuracy(label, raw_pred, compare_func, return_float=True, feed_dict=None, args=None):
    """get instance-wise accuracy for structured prediction task instead of pointwise task"""
    # Accouting for difference in number of nodes for gcp
    batch_num_nodes = raw_pred.shape[2]
    label = label[:,:batch_num_nodes]
    feed_dict['query'] = feed_dict['query'][:,:batch_num_nodes]
    feed_dict['target'] = feed_dict['target'][:,:batch_num_nodes]
    feed_dict['target_set'] = feed_dict['target_set'][:,:,:batch_num_nodes]
    
    return_monitors = {}
    step_wise_accuracy = None 
    # disctretize output predictions
    query_info = None 
        
    if args.task_is_gcp or args.task_is_sudoku:
        
        max_num_nodes, num_nodes_list = feed_dict['num_nodes'].max(), feed_dict['num_nodes']
        mask_of_num_nodes = torch.arange(num_nodes_list.max(), device = num_nodes_list.device)[None,:] < num_nodes_list[:,None] 
        
        step_pred = as_tensor(raw_pred.argmax(dim=1)).float()
        
        pred = step_pred[:,:,-1]
        label = as_tensor(label).type(pred.dtype)
        
        step_pred = step_pred.transpose(1,2)
        if 'get_stepwise_accuracy' in args and args.get_stepwise_accuracy:
#             Pdb().set_trace()
            num_missing = ((feed_dict['query'] == 0)*mask_of_num_nodes).sum(dim=-1)
            #num_colors = feed_dict['target'].max(dim=-1)[0]
            query_info = torch.stack([feed_dict['num_nodes'], num_missing,feed_dict['chromatic_num']]).transpose(0,1).cpu().numpy()

            step_wise_accuracy = torch.zeros(step_pred.size()[:2])
            mask_nodes = mask_of_num_nodes.unsqueeze(1).expand_as(step_pred)
            mask_diff = ((label.unsqueeze(1).expand_as(step_pred) == step_pred)*mask_nodes).float()
            point_acc_step_wise_qw = (mask_diff.sum(dim=-1)/mask_nodes.sum(dim=-1).float())
            point_acc_step_wise= point_acc_step_wise_qw.mean(dim=0).cpu().numpy()
            point_acc_step_wise = dict([('s.pw.{}'.format(i),x) for i,x in enumerate(point_acc_step_wise)])
            return_monitors.update(point_acc_step_wise)
        
    elif args.task_is_futoshiki:
        #Pdb().set_trace()
        step_pred = as_tensor(raw_pred.argmax(dim=1)).float()
        pred = step_pred[:,:,-1]
        label = as_tensor(label).type(pred.dtype)
        # step pred is batch_size x 81 x num_steps
        # transpose for more efficient reward calculation
        # new shape is batch_size x num_Steps x 81
        step_pred = step_pred.transpose(1,2)
        
        if 'get_stepwise_accuracy' in args and args.get_stepwise_accuracy:
            num_missing = (feed_dict['query'] == 0).sum(dim=-1)
            num_colors = feed_dict['target'].max(dim=-1)[0]
            #Pdb().set_trace()
            query_info = torch.stack([feed_dict['qid'].squeeze(-1), feed_dict['n'], num_missing,num_colors.long()]).cpu().numpy()

            step_wise_accuracy = torch.zeros(step_pred.size()[:2])
            point_acc_step_wise_qw = (label.unsqueeze(1).expand_as(step_pred) == step_pred).float().mean(dim=-1)
            point_acc_step_wise= point_acc_step_wise_qw.mean(dim=0).cpu().numpy()
            point_acc_step_wise = dict([('s.pw.{}'.format(i),x) for i,x in enumerate(point_acc_step_wise)])
            return_monitors.update(point_acc_step_wise)
            
    else:
        pred = as_tensor(raw_pred)
        pred = (pred > 0.5).float()
        label = as_tensor(label).type(pred.dtype)
        
    #
    if args.task_is_gcp or args.task_is_sudoku:
#         Pdb().set_trace()
        max_num_nodes, num_nodes_list = feed_dict['num_nodes'].max(), feed_dict['num_nodes']
        mask_nodes = torch.arange(num_nodes_list.max(), device = num_nodes_list.device)[None,:] < num_nodes_list[:,None] 
        mask_diff = ((label == pred)*mask_nodes).float()
        point_acc = mask_diff.sum()/mask_nodes.sum().float()
        
        mask_diff = mask_diff.masked_fill(~mask_nodes,1.0)
        
        incorrect = torch.min(mask_diff,dim=1)[0]
        in_acc = torch.sum(incorrect).float()/len(label)
#         Pdb().set_trace()
    else:
        diff = (label==pred)
        point_acc = torch.sum(diff).float()/label.numel()
        incorrect = torch.min(diff,dim=1)[0]
        in_acc = torch.sum(incorrect).float()/len(label)

    corrected_acc = 0
    reward = []
    new_targets = []
    
    if args.task_is_gcp or args.task_is_sudoku:
        fd_query = feed_dict['query'].type(pred.dtype)[:,:batch_num_nodes]
        debug_acc = compare_func(pred, fd_query, feed_dict['edges'], mask_nodes)
    elif args.task_is_futoshiki:
        fd_query = feed_dict['query'].type(pred.dtype)
        debug_acc = compare_func(pred, fd_query, feed_dict['lt_edges'])
    else:
        debug_acc = compare_func(pred, feed_dict['query'].type(pred.dtype))
        
    for i, x in enumerate(pred):
        #if compare_func(x,feed_dict['query'][i].type(x.dtype)) != debug_acc[i]: 
        #        Pdb().set_trace()
                #debug_compare_func(x,feed_dict['query'][i].type(x.dtype))
        
        if debug_acc[i]:
        #if compare_func(x,feed_dict['query'][i].type(x.dtype)):
            corrected_acc += 1
            # check if pred matches any target
            fd_target_set = feed_dict['target_set'][:,:,:batch_num_nodes]
            if ((fd_target_set[i].type(x.dtype)==x).sum(dim=1)==x.shape[0]).sum()>0:
                new_targets.append((None,None))
            else:
                new_targets.append((x, 0))
        else:
            new_targets.append((None,None))
        #Pdb().set_trace() 
        if args.task_is_futoshiki or args.task_is_sudoku or args.task_is_gcp:
            #if args.use_gpu:
            #    diff = torch.zeros(len(feed_dict['target_set'][i]),step_pred.shape[1], device=torch.device("cuda"))
            #else:
            #    diff = torch.zeros(len(feed_dict['target_set'][i]),step_pred.shape[1]).cuda()
            #for target_idx,target in enumerate(feed_dict['target_set'][i,:feed_dict['count'][i]].float()):
            #    diff[target_idx] = torch.sum(~(step_pred[i]==target), dim=1).float()
            #for target_idx in range(feed_dict['count'][i],diff.shape[0]):
            #    diff[target_idx] = diff[target_idx-1]
            #
            #alternative tensor way
            fd_target_set = feed_dict['target_set'][:,:,:batch_num_nodes]
            NS,NN,TS = step_pred.size(1),step_pred.size(2), feed_dict['target_set'].size(1) 
            diff = (step_pred[i].unsqueeze(-1).expand(NS,NN,TS).transpose(0,2).float() != fd_target_set[i].unsqueeze(-1).expand(TS,NN,NS).float()).sum(dim=1).float()

            reward.append(diff.mean(dim=1))
            #
            #get accuracy at each step
            if 'get_stepwise_accuracy' in args and args.get_stepwise_accuracy:
                
#                 Pdb().set_trace()
                fd_query = feed_dict['query'][i,:batch_num_nodes].unsqueeze(0).expand_as(step_pred[i]).type(step_pred.dtype)
                
                if args.task_is_gcp or args.task_is_sudoku:
                    fd_edges = feed_dict['edges'][i].unsqueeze(0).expand(
                        step_pred[i].shape[0],*feed_dict['edges'][i].size()).type(step_pred.dtype)
                    
                    fd_mask = mask_nodes[i].unsqueeze(0).expand_as(step_pred[i])

                    debug_step_acc = compare_func(step_pred[i], fd_query, fd_edges, fd_mask)
                elif args.task_is_futoshiki:
                    step_lt_edges = feed_dict['lt_edges'][i].unsqueeze(0).expand(step_pred[i].size(0),-1,-1)
                    debug_step_acc = compare_func(step_pred[i], fd_query, step_lt_edges.contiguous())
                    #fd_edges = feed_dict['edges'][i].unsqueeze(0).expand(
                    #    step_pred[i].shape[0],*feed_dict['edges'][i].size()).type(step_pred.dtype)

                    #debug_step_acc = compare_func(step_pred[i], fd_query, fd_edges)
                    
                step_wise_accuracy[i] = debug_step_acc
                #for j,this_step_pred in enumerate(step_pred[i]):
                    #if compare_func(this_step_pred,feed_dict['query'][i].type(
                    #    this_step_pred.dtype)) != debug_step_acc[j]:
                    #        Pdb().set_trace()
                            #debug_compare_func(this_step_pred,feed_dict['query'][i].type(this_step_pred.dtype))

                #    if compare_func(this_step_pred,feed_dict['query'][i].type(this_step_pred.dtype)):
                #        step_wise_accuracy[i,j]  = 1
        else:
            diff = torch.sum(~(feed_dict["target_set"][i].type(x.dtype)==x),dim=1).float()
            reward.append(diff)
    #
    corrected_acc /= len(pred)
    querywise_accuracy = None
    if step_wise_accuracy is not None:
        #stepwise pointwise 
        querywise_accuracy = torch.stack([point_acc_step_wise_qw.cpu(), step_wise_accuracy.cpu()],dim=-1).cpu().numpy() 
        
        step_wise_accuracy = step_wise_accuracy.float().mean(dim=0).cpu().numpy()
        step_wise_accuracy = dict([('s.ca.{}'.format(i),x) for i,x in enumerate(step_wise_accuracy)])
        return_monitors.update(step_wise_accuracy) 
    # 
    reward = -torch.stack(reward)
    
    # Assert corrected accuracy is greater than normal accuracy
    if (incorrect.float()>debug_acc.float()).any().item():
        raise
        #Pdb().set_trace()
    if return_float:
        return_monitors.update({"accuracy": in_acc.item(),
                "corrected accuracy": corrected_acc,
                "pointwise accuracy": point_acc.item()
                })
    else: 
        return_monitors.update({"accuracy": torch.tensor(in_acc),
            "corrected accuracy": torch.tensor(corrected_acc),
            "pointwise accuracy": point_acc})
    # 
    
    return return_monitors, {'reward': reward, 'new_targets': new_targets, 
                    'querywise_accuracy': querywise_accuracy, 'query_info': query_info} 
         

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
            monitors['gradrms/' + name] = (rms(p.grad) / max(rms(p), 1e-8))
    return monitors
