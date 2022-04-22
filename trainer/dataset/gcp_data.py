import csv
import os
import urllib.request
import zipfile
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataset import Dataset
import torch
import dgl
from copy import deepcopy
from copy import copy
import math
from IPython.core.debugger import Pdb
from itertools import product

from jacinle.logging import get_logger, set_output_file
logger = get_logger(__file__)
#Three relations: 
#Cluster, Contains, Cell
#Cluster, MayContain, Cell
#Cell, Diff, Cell

def _get_cluster_graph(puzzle, chromatic_num, have_may_edges):
    #Pdb().set_trace()
    query, edges = puzzle
    graph_size = query.numel()

    filled_cells = query.nonzero().view(-1)

    contains_graph = dgl.bipartite((query[filled_cells].tolist(),filled_cells.tolist()),'cluster','contains','cell',num_nodes = (chromatic_num+1,graph_size))

    if not have_may_edges:
        return  contains_graph,  None
    

    blank_cells = (query==0).nonzero().view(-1)
    num_blank_cells = blank_cells.numel()
    rpt_blank_cells = blank_cells.repeat(chromatic_num)
    rpt_cluster_cells = torch.arange(1,chromatic_num+1).repeat_interleave(num_blank_cells)


    may_contain_graph = dgl.bipartite((rpt_cluster_cells,rpt_blank_cells),'cluster','may_contain','cell',num_nodes = (chromatic_num+1,graph_size))

    return  contains_graph,  may_contain_graph 


def _get_constraint_graph(puzzle):
    
    query, edges_fwd = puzzle
#     Pdb().set_trace()
    edges_bck = torch.stack([edges_fwd[:,1], edges_fwd[:,0]]).transpose(0,1)
    edges = torch.cat((edges_fwd,edges_bck), dim=0)
        
    
    g = [dgl.graph(list(edges),'cell', 'diff', query.numel())]
    return g 


def get_hetero_domain(puzzle, chromatic_num, have_may_edges = True):
    
    query, edges = puzzle
    graph_size = query.numel()

    constraint_graph_list = _get_constraint_graph(puzzle)
    contains_graph, may_contain_graph = _get_cluster_graph(puzzle, chromatic_num, have_may_edges)
   
    which_edges = [contains_graph]
    
    if have_may_edges:
        which_edges.append(may_contain_graph)
        
    g = dgl.hetero_from_relations(constraint_graph_list + which_edges)
    return g

def get_hetero_binary(puzzle, chromatic_num, args):

    query, edges_undir = puzzle
    graph_size = query.numel()
    
    if args.logk:
        num_levels = math.floor(math.log2(chromatic_num)) + 1
    else:
        num_levels = chromatic_num
    
    #Pdb().set_trace() 
    level_offset = torch.arange(num_levels).repeat_interleave(edges_undir.shape[0])*graph_size
    edges_fwd = edges_undir.repeat(num_levels,1) + level_offset.unsqueeze(-1)
    edges_bck = torch.stack([edges_fwd[:,1], edges_fwd[:,0]]).transpose(0,1)
    intra_level_edges = torch.cat((edges_fwd,edges_bck), dim=0)
    
    inter_level_edges_base = [(i,j) for i in range(num_levels) for j in range(num_levels) if i!=j]
    inter_level_edges_base = torch.Tensor(inter_level_edges_base)*graph_size
    
    position_offset = torch.arange(graph_size).repeat_interleave(inter_level_edges_base.shape[0])
    inter_level_edges = inter_level_edges_base.repeat(graph_size,1).long() + position_offset.unsqueeze(-1)
        
    
    g_list = []
    
    intra_graph = dgl.graph(intra_level_edges.tolist(), 'cell', 'intra_diff', graph_size*num_levels)
    inter_graph = dgl.graph(inter_level_edges.tolist(), 'cell', 'inter_diff', graph_size*num_levels)
    g_list += [intra_graph, inter_graph]
    
    g = dgl.hetero_from_relations(g_list)
    
    return g


def get_cell_data(query, chromatic_num, args):
    
    if args.logk:
        num_levels = math.floor(math.log2(chromatic_num)) + 1
        mask = 2**torch.arange(num_levels).to(query.device, query.dtype)
        cell_data = ((query.unsqueeze(-1) & mask) > 0).int()
    else:
        one_hot = torch.eye(chromatic_num+1)[query.long()] 
        # initialise masked cells with -1 across all levels
        cell_data = one_hot[:,1:] - one_hot[:,:1] 
    return cell_data

def get_hetero_graph(puzzle, chromatic_num, args):
    
    have_may_edges = True
    query, _ = puzzle
    if args.binary_model:
        graph = get_hetero_binary(puzzle, chromatic_num, args)
        cell_data = get_cell_data(query, chromatic_num, args)
        graph.nodes['cell'].data['q'] = cell_data.transpose(0,1).flatten()
        graph.nodes['cell'].data['t'] = torch.zeros(graph.number_of_nodes('cell')).long()
        return graph
    else:    
        graph = get_hetero_domain(puzzle, chromatic_num, have_may_edges)
        graph.nodes['cell'].data['q'] = torch.tensor(query)
        graph.nodes['cell'].data['t'] = torch.zeros(graph.number_of_nodes('cell')).long()
        graph.nodes['cluster'].data['t'] = torch.ones(graph.number_of_nodes('cluster')).long()
        graph.nodes['cluster'].data['id'] = torch.arange(graph.number_of_nodes('cluster')).long()
        return graph
   

def get_class_graph(chromatic_num, args):
    #add lt and gt edges
    num_classes = chromatic_num
    graph_list = []
    g = None
    if args.diff_edges_in_class_graph:
        diff_edges = list(product(range(1,num_classes+1), range(1,num_classes+1)))
        diff_edges  = [x for x in diff_edges if x[0] != x[1]]
        diff_graph = dgl.graph(diff_edges, 'cluster','diff',num_classes+1)
        graph_list.append(diff_graph)
        g = dgl.hetero_from_relations(graph_list)
    # 
    return g


