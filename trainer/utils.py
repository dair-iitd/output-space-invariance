import math
import time
import os, sys
import numpy as np
import torch
import torch.nn as nn
from IPython.core.debugger import Pdb
from torch.utils.data._utils.collate import default_collate
import copy
import dgl
import dataset.futoshiki_data as fd
import torch
import random


def setup_seed(manualSeed,use_gpu):
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    if use_gpu: 
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        # 
        torch.backends.cudnn.enabled = False 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_edge_ids(args):
    if args.task_is_futoshiki:
        return fd.get_edge_ids(args)


def collate_graph(batch, args):
    graph_list = []
    class_graph_list = []
    num_edges = []
    for i,this_element in enumerate(batch):
#         num_edges.append(this_element['lt_edges'].shape[0])
        if 'graph' in this_element:
            graph = this_element.pop('graph')
            graph.nodes['cell'].data['sno'] = torch.zeros(graph.number_of_nodes('cell')).long() + i
            if 'cluster' in graph.ntypes:
                graph.nodes['cluster'].data['sno'] = torch.zeros(graph.number_of_nodes('cluster')).long() + i
            #
            if 'pos' in graph.etypes:
                for etype in ['pos','neg','unk']:
                    graph[etype].edata['sno']  = torch.zeros(graph[etype].number_of_edges()).long() + i

            graph_list.append(graph)
        if 'class_graph' in this_element:
            class_graph = this_element.pop('class_graph')
            class_graph.nodes['cluster'].data['sno'] = torch.zeros(graph.number_of_nodes('cluster')).long() + i
            class_graph_list.append(class_graph)

#     if len(set(num_edges)) > 1:
#         Pdb().set_trace()

    feed_dict = default_collate(batch)
    if len(graph_list) == len(batch):
        g = dgl.batch_hetero(graph_list)
        feed_dict['bg'] = g
        
    elif len(graph_list) > 0:
        raise
    if len(class_graph_list) == len(batch):
        feed_dict['bcg'] = dgl.batch_hetero(class_graph_list)
    elif len(class_graph_list) > 0:
        raise

    return feed_dict

def add_missing_target(feed_dict,new_targets,reward):
    flag = False
    for i,(this_target,this_reward) in enumerate(new_targets):
        if this_target is not None:
            #Pdb().set_trace()
            flag = True 
            this_count =feed_dict['count'][i]  
            feed_dict['target_set'][i,this_count] = this_target
            feed_dict['mask'][i,this_count] = 1
            reward[i][this_count] = this_reward
            feed_dict['is_ambiguous'][i] = 1
            #
            #Pdb().set_trace()
            if 'nlm_target_set' in feed_dict:
                nlm_type_target = convert_to_unary_predicates(this_target,feed_dict['num_nodes'][i], feed_dict['chromatic_num'][i])
                feed_dict['nlm_target_set'][i,this_count] = nlm_type_target

            feed_dict['count'][i] += 1

def convert_to_unary_predicates(query,n,k):
    #n = number of nodes
    #k = number of classes
    predicate = nn.functional.one_hot(query.long()-1).flatten()
    return predicate
    
    #predicate = torch.zeros(n*k).to(query.device)
    #Pdb().set_trace()
    #nz_indices = query.nonzero().squeeze()
    #for posn_ind in nz_indices:
    #    value = query[posn_ind] - 1 # -1 because values in the query start from 1 and not 0.
    #    ind = posn_ind*k + value
    #    start_ind = posn_ind*k
    #    predicate[ind] = 1
    #Pdb().set_trace()
    #return predicate
 
def compute_param_norm(parameters, norm_type= 2): 
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0 
    for p in parameters:
        if p.is_sparse:
            # need to coalesce the repeated indices before finding norm
            grad = p.data.coalesce()
            param_norm = grad._values().norm(norm_type)
        else:
            param_norm = p.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def compute_grad_norm(parameters, norm_type= 2): 
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0 
    for p in parameters:
        if p.grad.is_sparse:
            # need to coalesce the repeated indices before finding norm
            grad = p.grad.data.coalesce()
            param_norm = grad._values().norm(norm_type)
        else:
            param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm 

#copied from allennlp.trainer.util
def sparse_clip_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Supports sparse gradients.

    Parameters
    ----------
    parameters : ``(Iterable[torch.Tensor])``
        An iterable of Tensors that will have gradients normalized.
    max_norm : ``float``
        The max norm of the gradients.
    norm_type : ``float``
        The type of the used p-norm. Can be ``'inf'`` for infinity norm.

    Returns
    -------
    Total norm of the parameters (viewed as a single vector).
    """
    # pylint: disable=invalid-name,protected-access
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            if p.grad.is_sparse:
                # need to coalesce the repeated indices before finding norm
                grad = p.grad.data.coalesce()
                param_norm = grad._values().norm(norm_type)
            else:
                param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad.is_sparse:
                p.grad.data._values().mul_(clip_coef)
            else:
                p.grad.data.mul_(clip_coef)
    return total_norm


# === performing gradient descent
#copied from allennlp.trainer.util
def rescale_gradients(model, grad_norm = None):
    """
    Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
    """
    if grad_norm:
        parameters_to_clip = [p for p in model.parameters()
                              if p.grad is not None]
        return sparse_clip_norm(parameters_to_clip, grad_norm)
    return None

#called after loss.backward and optimizer.zero_grad. Before optimizer.step()
def gradient_normalization(model, grad_norm = None):
    # clip gradients
    #grad norm before clipping
    parameters = [p for p in model.parameters() if p.grad is not None]
    grad_norm_before_clip = compute_grad_norm(parameters)
    grad_norm_after_clip = grad_norm_before_clip
    param_norm_before_clip = compute_param_norm(parameters)
    grad_before_rescale = rescale_gradients(model, grad_norm)
    
    #torch.nn.utils.clip_grad_norm(model.parameters(), clip_val)
    grad_norm_after_clip = compute_grad_norm(parameters)
    #
    return grad_norm_before_clip.item(), grad_norm_after_clip.item(), param_norm_before_clip.item() 
