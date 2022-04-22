#! /usr/bin/env python3
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for datasets."""

import collections
import copy
import jacinle.random as random
import numpy as np
import math
from . import sudoku_data as sd
from IPython.core.debugger import Pdb
grid_cache = {}
block_shape_dict = {6: (2, 3),
                 8: (2, 4),
                 9: (3, 3),
                 10: (2, 5),
                 12: (2, 6),
                 14: (2, 7),
                 15: (3, 5),
                 16: (4, 4),
                 24: (4, 6),
                 25: (5, 5)}

for x in block_shape_dict.keys():
    grid_cache[x] = np.array(range(x*x)).reshape(x,x) 


def convert_sudoku_to_gcp(sudoku_data):
    #gcp_keys =  ['edges', 'target', 'query', 'chromatic_num', 'num_nodes', 'num_edges', 'target_set']
    #sudoku_keys = ['query', 'target_set', 'count', 'givens']
    gcp_data  = []
    edges_cache = {}

    for this_data in sudoku_data:
        this_data['chromatic_num'] = int(math.sqrt(this_data['query'].shape[0]))
        this_data['target'] = this_data['target_set'][0]
        this_data['num_nodes'] = this_data['query'].shape[0]
        if this_data['chromatic_num'] not in edges_cache:
            edges_cache[this_data['chromatic_num']] = get_sudoku_edges(this_data['chromatic_num'])

        this_data['edges'] = copy.deepcopy(edges_cache[this_data['chromatic_num']])
        this_data['num_edges'] = this_data['edges'].shape[0]
        #Pdb().set_trace()
        this_data['target_set']= np.stack(this_data['target_set'],  axis=0)
        gcp_data.append(this_data)
    #
    return gcp_data


def get_sudoku_edges(board_size):
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
            if row_src < i:
                edges.add((row_src, i))
            if col_src < i:
                edges.add((col_src, i))
            row_src += 1
            col_src += board_size 
        
        # same grid
        
        b_row = (i//board_size)//block_x 
        b_col = (i%board_size)//block_y

        block_values = grid[block_x*b_row:block_x*(b_row+1),block_y*b_col:block_y*(b_col+1)].flatten()       
        for n in block_values: 
            if n < i: 
                edges.add((n, i))
    #
    edges = np.array(list(edges))
    return edges
