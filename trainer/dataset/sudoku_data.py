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

from jacinle.logging import get_logger, set_output_file
logger = get_logger(__file__)
#Three relations: 
#Cell , BelongsTo, Cluster
#Cluster, Contains, Cell
#Cell, Diff, Cell

def _get_cluster_graph_fast(puzzle, have_may_edges):
    #Pdb().set_trace()
    board_size = int(math.sqrt(puzzle.numel()))

    filled_cells = puzzle.nonzero().view(-1)

    belongs_graph = dgl.bipartite((filled_cells.tolist(), puzzle[filled_cells].tolist()),'cell','belongs','cluster',num_nodes = (puzzle.numel(),board_size+1))
    contains_graph = dgl.bipartite((puzzle[filled_cells].tolist(),filled_cells.tolist()),'cluster','contains','cell',num_nodes = (board_size+1,puzzle.numel()))

    if not have_may_edges:
        return belongs_graph, contains_graph, None, None
    

    blank_cells = (puzzle==0).nonzero().view(-1)
    num_blank_cells = blank_cells.numel()
    rpt_blank_cells = blank_cells.repeat(board_size)
    rpt_cluster_cells = torch.arange(1,board_size+1).repeat_interleave(num_blank_cells)


    may_belong_graph = dgl.bipartite((rpt_blank_cells,rpt_cluster_cells),'cell','may_belong','cluster',num_nodes = (puzzle.numel(),board_size+1))
    may_contain_graph = dgl.bipartite((rpt_cluster_cells,rpt_blank_cells),'cluster','may_contain','cell',num_nodes = (board_size+1,puzzle.numel()))

    return belongs_graph, contains_graph, may_belong_graph, may_contain_graph 

def _get_cluster_graph(puzzle, have_may_edges):
    board_size = int(math.sqrt(puzzle.numel()))
    belongs_edges = set() #from  cell to  cluster
    contains_edges = set() # from cluster to cell
    may_belong_edges = set() #from  cell to  cluster
    may_contain_edges = set() # from cluster to cell

    #Pdb().set_trace()
    for i in range(board_size+1):
        cells = (puzzle == i).nonzero().view(-1)
        if i == 0 and have_may_edges:
            #add to may belong and may contain graph
            for j in cells:
                for k in range(1,board_size+1):
                    #belongs_edges.add((j.item(),k))
                    #contains_edges.add((k,j.item()))
                    may_belong_edges.add((j.item(),k))
                    may_contain_edges.add((k,j.item()))
        else:
            for j in cells:
                belongs_edges.add((j.item(),i))
                contains_edges.add((i,j.item()))
    #
    belongs_graph = dgl.bipartite(list(belongs_edges),'cell','belongs','cluster',num_nodes = (puzzle.numel(),board_size+1))
    contains_graph = dgl.bipartite(list(contains_edges),'cluster','contains','cell',num_nodes = (board_size+1,puzzle.numel()))
    if have_may_edges:
        may_belong_graph = dgl.bipartite(list(may_belong_edges),'cell','may_belong','cluster',num_nodes = (puzzle.numel(),board_size+1))
        may_contain_graph = dgl.bipartite(list(may_contain_edges),'cluster','may_contain','cell',num_nodes = (board_size+1,puzzle.numel()))
    else:
        may_belong_graph, may_contain_graph = None, None
    
    return belongs_graph, contains_graph, may_belong_graph, may_contain_graph 


def refine_diff(puzzle, edges):

    filled_cells = set(puzzle.nonzero().view(-1).tolist())
    
    edges1, edges2, edges3, edges4 = set(), set(), set(), set()

    for x in edges:
        src, dst = x
        if (src in filled_cells) and (dst in filled_cells):
            edges1.add(x)
        if (src not in filled_cells) and (dst in filled_cells):
            edges2.add(x)
        if (src in filled_cells) and (dst not in filled_cells):
            edges3.add(x)
        if (src not in filled_cells) and (dst not in filled_cells):
            edges4.add(x)

    
    g = [dgl.graph(list(edges1),'cell', 'diff1', puzzle.numel()), dgl.graph(list(edges2),'cell', 'diff2', puzzle.numel()),dgl.graph(list(edges3),'cell', 'diff3', puzzle.numel()), dgl.graph(list(edges4),'cell', 'diff4', puzzle.numel())]
    return g 


def _get_constraint_graph(board_size=9, puzzle=None, refine_diff_edges = False):
    
    num_cells = board_size*board_size

    global block_shape_dict, grid_cache    
    
    block_x, block_y = block_shape_dict[board_size]
    num_cells = board_size*board_size
    
    grid = grid_cache[board_size]
    edges = set()
    for i in range(num_cells):
        row, col = i // board_size, i % board_size 
        # same row and col
        row_src = row * board_size 
        col_src = col
        for _ in range(board_size):
            if row_src != i:
                edges.add((row_src, i))
            if col_src != i:
                edges.add((col_src, i))
            row_src += 1
            col_src += board_size 
        
        # same grid
        
        b_row = (i//board_size)//block_x 
        b_col = (i%board_size)//block_y

        block_values = grid[block_x*b_row:block_x*(b_row+1),block_y*b_col:block_y*(b_col+1)].flatten()       
        for n in block_values: 
            if n != i: 
                edges.add((n, i))
    if refine_diff_edges:
        g = refine_diff(puzzle, edges)
    else:
#         Pdb().set_trace()
        g = [dgl.graph(list(edges),'cell', 'diff')]
    return g 


def get_hetero_graph(puzzle,have_may_edges = True, refine_diff_edges = False, cell2cluster_edges=True):
    board_size = int(math.sqrt(puzzle.numel()))
    global constraints_graph_cache
    #
    #constraint_graph = deepcopy(constraints_graph_cache[board_size])
    constraint_graph_list = _get_constraint_graph(board_size, puzzle, refine_diff_edges)
    

    belongs_graph,contains_graph,may_belong_graph, may_contain_graph = _get_cluster_graph_fast(puzzle, have_may_edges)
    which_edges = [contains_graph]
    if cell2cluster_edges:
        which_edges.append(belongs_graph)
        if have_may_edges:
            which_edges.append(may_belong_graph)
            which_edges.append(may_contain_graph)
    elif have_may_edges:
        which_edges.append(may_contain_graph)
        
        #g = dgl.hetero_from_relations(constraint_graph_list + [belongs_graph, contains_graph,may_belong_graph, may_contain_graph])
        #g = dgl.hetero_from_relations(constraint_graph_list + [belongs_graph, contains_graph])

    g = dgl.hetero_from_relations(constraint_graph_list + which_edges)
    return g


def _basic_sudoku_graph(board_size = 9):
    g = dgl.DGLGraph()
    block_size = int(math.sqrt(board_size))
    num_cells = board_size*board_size
    #grid = np.array(range(num_cells)).reshape(board_size,board_size) 
    global grid_cache
    grid = grid_cache[board_size]
    g.add_nodes(num_cells)
    for i in range(num_cells):
        row, col = i // board_size, i % board_size 
        # same row and col
        row_src = row * board_size 
        col_src = col
        for _ in range(board_size):
            if row_src != i:
                g.add_edges(row_src, i)
            if col_src != i:
                g.add_edges(col_src, i)
            row_src += 1
            col_src += board_size 
        
        # same grid
        block_r = row-row%block_size
        block_c = col-col%block_size
        block_values = grid[block_r:block_r+block_size,block_c:block_c+block_size].flatten()
        for n in block_values: 
            if n != i and (not g.has_edge_between(n,i)):
                g.add_edges(n, i)
    return g



def __basic_sudoku_graph():
    grids = [[0, 1, 2, 9, 10, 11, 18, 19, 20],
             [3, 4, 5, 12, 13, 14, 21, 22, 23],
             [6, 7, 8, 15, 16, 17, 24, 25, 26],
             [27, 28, 29, 36, 37, 38, 45, 46, 47],
             [30, 31, 32, 39, 40, 41, 48, 49, 50],
             [33, 34, 35, 42, 43, 44, 51, 52, 53],
             [54, 55, 56, 63, 64, 65, 72, 73, 74],
             [57, 58, 59, 66, 67, 68, 75, 76, 77],
             [60, 61, 62, 69, 70, 71, 78, 79, 80]]
    g = dgl.DGLGraph()
    g.add_nodes(81)
    for i in range(81):
        row, col = i // 9, i % 9
        # same row and col
        row_src = row * 9
        col_src = col
        for _ in range(9):
            if row_src != i:
                g.add_edges(row_src, i)
            if col_src != i:
                g.add_edges(col_src, i)
            row_src += 1
            col_src += 9
        # same grid
        grid_row, grid_col = row // 3, col // 3
        for n in grids[grid_row*3 + grid_col]:
            if n != i:
                g.add_edges(n, i)
    return g

grid_cache = {}
block_shape_dict = {6: (2, 3),
                 8: (2, 4),
                 9: (3, 3),
                 10: (2, 5),
                 12: (2, 6),
                 14: (2, 7),
                 15: (3, 5),
                 16: (4, 4),
                 25: (5, 5)}

for x in block_shape_dict.keys():
    grid_cache[x] = np.array(range(x*x)).reshape(x,x) 
    
# grid_cache[9] = np.array(range(81)).reshape(9,9) 
# grid_cache[16] = np.array(range(256)).reshape(16,16) 
# grid_cache[25] = np.array(range(625)).reshape(25,25) 

constraints_graph_cache = {}
for x in block_shape_dict.keys():
    constraints_graph_cache[x] = _get_constraint_graph(x)

# constraints_graph_cache[9] = _get_constraint_graph(9)
# constraints_graph_cache[16] = _get_constraint_graph(16)
# constraints_graph_cache[25] = _get_constraint_graph(25)


class ListDataset(Dataset):
    def __init__(self, *lists_of_data):
        assert all(len(lists_of_data[0]) == len(d) for d in lists_of_data)
        self.lists_of_data = lists_of_data

    def __getitem__(self, index):
        return tuple(d[index] for d in self.lists_of_data)

    def __len__(self):
        return len(self.lists_of_data[0])


def _get_sudoku_dataset(segment='train'):
    assert segment in ['train', 'valid', 'test']
    url = "https://data.dgl.ai/dataset/sudoku-hard.zip"
    zip_fname = "/tmp/sudoku-hard.zip"
    dest_dir = '/tmp/sudoku-hard/'

    if not os.path.exists(dest_dir):
        print("Downloading data...")

        urllib.request.urlretrieve(url, zip_fname)
        with zipfile.ZipFile(zip_fname) as f:
            f.extractall('/tmp/')

    def read_csv(fname):
        print("Reading %s..." % fname)
        with open(dest_dir + fname) as f:
            reader = csv.reader(f, delimiter=',')
            return [(q, a) for q, a in reader]

    data = read_csv(segment + '.csv')

    def encode(samples):
        def parse(x):
            return list(map(int, list(x)))

        encoded = [(parse(q), parse(a)) for q, a in samples]
        return encoded

    data = encode(data)

    return data


def sudoku_dataloader(batch_size, segment='train'):
    """
    Get a DataLoader instance for dataset of sudoku. Every iteration of the dataloader returns
    a DGLGraph instance, the ndata of the graph contains:
    'q': question, e.g. the sudoku puzzle to be solved, the position is to be filled with number from 1-9
         if the value in the position is 0
    'a': answer, the ground truth of the sudoku puzzle
    'row': row index for each position in the grid
    'col': column index for each position in the grid
    :param batch_size: Batch size for the dataloader
    :param segment: The segment of the datasets, must in ['train', 'valid', 'test']
    :return: A pytorch DataLoader instance
    """
    data = _get_sudoku_dataset(segment)
    q, a = zip(*data)

    dataset = ListDataset(q, a)
    if segment == 'train':
        data_sampler = RandomSampler(dataset)
    else:
        data_sampler = SequentialSampler(dataset)

    basic_graph = _basic_sudoku_graph()
    sudoku_indices = np.arange(0, 81)
    rows = sudoku_indices // 9
    cols = sudoku_indices % 9

    def collate_fn(batch):
        graph_list = []
        for q, a in batch:
            q = torch.tensor(q, dtype=torch.long)
            a = torch.tensor(a, dtype=torch.long)
            graph = copy(basic_graph)
            graph.ndata['q'] = q  # q means question
            graph.ndata['a'] = a  # a means answer
            graph.ndata['row'] = torch.tensor(rows, dtype=torch.long)
            graph.ndata['col'] = torch.tensor(cols, dtype=torch.long)
            graph_list.append(graph)
        batch_graph = dgl.batch(graph_list)
        return batch_graph

    dataloader = DataLoader(dataset, batch_size, sampler=data_sampler, collate_fn=collate_fn)
    return dataloader
