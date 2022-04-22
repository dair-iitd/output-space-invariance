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
from collections import defaultdict

logger = get_logger(__file__)
#Three relations: 
#Cluster, Contains, Cell
#Cluster, MayContain, Cell
#Cell, Diff, Cell

def _get_cluster_graph(query, have_may_edges):
    #Pdb().set_trace()
    all_graphs = defaultdict(lambda: None)
    puzzle = query[0]
    board_size = int(math.sqrt(puzzle.numel()))

    filled_cells = puzzle.nonzero().view(-1)

    contains_graph = dgl.bipartite((puzzle[filled_cells].tolist(),filled_cells.tolist()),'cluster','contains','cell',num_nodes = (board_size+1,puzzle.numel()))
    all_graphs['contains_graph'] = contains_graph
    
    if not have_may_edges:
        return all_graphs

    blank_cells = (puzzle==0).nonzero().view(-1)
    num_blank_cells = blank_cells.numel()
    rpt_blank_cells = blank_cells.repeat(board_size)
    rpt_cluster_cells = torch.arange(1,board_size+1).repeat_interleave(num_blank_cells)


    may_contain_graph = dgl.bipartite((rpt_cluster_cells,rpt_blank_cells),'cluster','may_contain','cell',num_nodes = (board_size+1,puzzle.numel()))
    all_graphs['may_contain_graph'] = may_contain_graph
    return all_graphs

def get_futo_edges(board_size):
    edges, num_cells = set(), board_size*board_size
    for i in range(num_cells):
        row, col = i // board_size, i % board_size 
        # same row and col
        row_src, col_src = row * board_size, col
        for _ in range(board_size):
            if row_src != i:
                edges.add((row_src, i))
            if col_src != i:
                edges.add((col_src, i))
            row_src += 1
            col_src += board_size 
    return edges

def _get_constraint_graph(puzzle):
    
#     Pdb().set_trace()
    
    query, lt_edges = puzzle
    lt_edges = lt_edges.int()
    gt_edges = torch.stack((lt_edges[:,1],lt_edges[:,0])).transpose(0,1)
    num_cells = query.shape[0]
    
    lt_graph = dgl.graph(lt_edges.tolist(),'cell','lt',num_cells)
    gt_graph = dgl.graph(gt_edges.tolist(),'cell','gt',num_cells)
    edges = get_futo_edges(int(math.sqrt(num_cells)))
      
#     Pdb().set_trace()
        
    g = [dgl.graph(list(edges),'cell', 'diff', query.numel())]
    return g + [lt_graph, gt_graph]


def get_class_graph(puzzle_size, diff_edges_in_class_graph = 1,edge_name_prefix = '', get_list_of_graphs = False):
    #add lt and gt edges
    num_classes = int(math.sqrt(puzzle_size))
    graph_list = []
    lt_edges = []
    gt_edges = []
    for start_node in range(1,num_classes):
        for end_node in range(start_node+1,num_classes+1):
            lt_edges.append((start_node, end_node))
            gt_edges.append((end_node, start_node))


    lt_graph = dgl.graph(lt_edges,'cluster',edge_name_prefix+'lt',num_classes+1)
    gt_graph = dgl.graph(gt_edges,'cluster',edge_name_prefix+'gt',num_classes+1)
    graph_list.append(lt_graph)
    graph_list.append(gt_graph)
    if diff_edges_in_class_graph:
        diff_edges = list(product(range(1,num_classes+1), range(1,num_classes+1)))
        diff_edges  = [x for x in diff_edges if x[0] != x[1]]
        diff_graph = dgl.graph(diff_edges, 'cluster',edge_name_prefix+'diff',num_classes+1)
        graph_list.append(diff_graph)
    #
    if get_list_of_graphs:
        return graph_list
    #
    g = dgl.hetero_from_relations(graph_list)
    return g

def get_hetero_domain(puzzle, have_may_edges = True):
    #Pdb().set_trace() 
    query, edges = puzzle
    graph_size = query.numel()

    constraint_graph_list = _get_constraint_graph(puzzle)

    all_graphs = _get_cluster_graph(puzzle, have_may_edges)
    which_edges = [all_graphs['contains_graph']]
    if have_may_edges:
        which_edges.append(all_graphs['may_contain_graph'])
    g = dgl.hetero_from_relations(constraint_graph_list + which_edges)
    return g



def get_hetero_binary(puzzle, args):
    
    query, lt_edges = puzzle
    gt_edges = torch.stack((lt_edges[:,1],lt_edges[:,0])).transpose(0,1)
    
    board_size, num_cells = int(math.sqrt(query.shape[0])), query.shape[0]
    diff_edges = torch.Tensor(sorted(get_futo_edges(board_size))).long()
    
    ## INTRA LEVEL 
    # Given edges, returns intra level edges for that type
    def make_intra_level_edges(edges):
    
        level_offset = torch.arange(board_size).repeat_interleave(edges.shape[0])*num_cells
        intra_level_edges = edges.repeat(board_size,1) + level_offset.unsqueeze(-1)
        return intra_level_edges
    
    intra_diff, intra_lt, intra_gt = [make_intra_level_edges(x) for x  in [diff_edges, lt_edges, gt_edges]]
    
    
    ## INTER LEVEL
    # Give basic pattern for edges in [1:board_size]
    def make_inter_level_edges(edges):
        inter_level_edges_base = torch.Tensor(edges)*num_cells
        position_offset = torch.arange(num_cells).repeat_interleave(inter_level_edges_base.shape[0])
        inter_level_edges = inter_level_edges_base.repeat(num_cells,1).long() + position_offset.unsqueeze(-1)
        return inter_level_edges
        
    inter_diff = make_inter_level_edges([(j,i) for i in range(board_size) for j in range(board_size) if i!=j])
    inter_lt = make_inter_level_edges([(j,i) for i in range(board_size) for j in range(i)])
    inter_gt = make_inter_level_edges([(i,j) for i in range(board_size) for j in range(i)])
      
    intra_diff_graph = dgl.graph(intra_diff.tolist(), 'cell', 'intra_diff', board_size**3)
    inter_diff_graph = dgl.graph(inter_diff.tolist(), 'cell', 'inter_diff', board_size**3)
    
    
    if args.share_lt_edges:
        intra_lt_graph = dgl.graph(intra_lt.tolist()+inter_lt.tolist(), 'cell', 'intra_lt', board_size**3)
        intra_gt_graph = dgl.graph(intra_gt.tolist()+inter_gt.tolist(), 'cell', 'intra_gt', board_size**3)
        g_list = [intra_diff_graph, inter_diff_graph, intra_lt_graph, intra_gt_graph]
        
    else:
    
        intra_lt_graph = dgl.graph(intra_lt.tolist(), 'cell', 'intra_lt', board_size**3)
        inter_lt_graph = dgl.graph(inter_lt.tolist(), 'cell', 'inter_lt', board_size**3)
        intra_gt_graph = dgl.graph(intra_gt.tolist(), 'cell', 'intra_gt', board_size**3)
        inter_gt_graph = dgl.graph(inter_gt.tolist(), 'cell', 'inter_gt', board_size**3)
        g_list = [intra_diff_graph, inter_diff_graph, intra_lt_graph, inter_lt_graph, intra_gt_graph, inter_gt_graph]
    
    
    g = dgl.hetero_from_relations(g_list)
    
    return g


def get_hetero_graph(puzzle, args, kwargs):
    have_may_edges = True
    query, _ = puzzle
    num_classes = int(math.sqrt(query.numel())) 
    
    if args.binary_model:
        board_size = int(math.sqrt(query.shape[0]))
        graph = get_hetero_binary(puzzle, args)
        one_hot = torch.eye(board_size+1)[query.long()] 
        # initialise masked cells with -1 across all levels
        cell_data = one_hot[:,1:] - one_hot[:,:1]      
        graph.nodes['cell'].data['q'] = cell_data.transpose(0,1).flatten()
        graph.nodes['cell'].data['t'] = torch.zeros(graph.number_of_nodes('cell')).long()
        return graph
        
    else:    
        graph = get_hetero_domain(puzzle, have_may_edges)
        graph.nodes['cell'].data['q'] = torch.tensor(query)
        graph.nodes['cell'].data['t'] = torch.zeros(graph.number_of_nodes('cell')).long()
        graph.nodes['cluster'].data['t'] = torch.ones(graph.number_of_nodes('cluster')).long()
        graph.nodes['cluster'].data['id'] = torch.arange(graph.number_of_nodes('cluster')).long()
        return graph
    
    
