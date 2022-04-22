from ortools.sat.python import cp_model
import networkx as nx
import time, math, os
import numpy as np
import random, copy
import pdb, pickle, itertools, multiprocessing, argparse
import pandas as pd
from collections import Counter
import pandas as pd
import pickle as pkl


# Sample usage: python pw_local_search.py --task gcp --test-file ../../domain-size-inv/data/gcp_test_k-4_n-40to150_mask-30to70.pkl --pred-file ../../domain-size-inv/trained_models/gcp/bin-to-avgcpt-split/ae-inter.diff.intra.diff_avgcpt-checkpoint.avg41-50.pth_bs-16_cpt-checkpoint.30.pth_ef-0_es-800_e-50_hd-96_ict-1_ln-1_lr-0.0001_ml-1_mdl-gcp_rm-lstm_s-2011_shall-0_shlt-0_task-gcp_tbs-4_saa-0/k4_test_querywise.pkl --num-cores 10 --timeout 100

parser = argparse.ArgumentParser()

parser.add_argument('--task',type=str, choices = ['sudoku','gcp'], help='Task Choices: gcp or sudoku')
parser.add_argument('--test-file',type=str, help='Path to test file')
parser.add_argument('--pred-file',type=str, help='Path to pickle file with predictions')

parser.add_argument('--num-cores',type=int, help='cores to be used', default=1)
parser.add_argument('--timeout',type=float, help='timeout for soln finding in sec', default=100)


args = parser.parse_args()


block_shapes = {
    8: (2,4), 9: (3,3), 10: (2,5), 12: (2,6), 15: (3,5), 16: (4,4), 24: (4,6), 25: (5,5)
}

def close_valid_soln(edges, query, pred, k, timeout=-1):
   
    model, solver = cp_model.CpModel(), cp_model.CpSolver()
    if timeout!=-1:
        solver.parameters.max_time_in_seconds = timeout
                
    num_nodes = query.shape[0]
    
    # Declare node variables
    node_vars = [model.NewIntVar(1, k,'node'+str(i)) for i in range(num_nodes)]
    
    # Constraint: Neighbours will be different
    _ = [model.Add(node_vars[i]!=node_vars[j]) for i,j in edges]
    
    # Constraint: Query colors are fixed
    _ = [model.Add(node_vars[i]==query[i]) for i in range(num_nodes) if query[i]]

    # Goal: Get as close as possible to pred
    bool_vars = [model.NewBoolVar(str(i)) for i in range(num_nodes)]

    for node_var, bool_var, pred_clr in zip(node_vars,bool_vars,pred):
        model.Add(node_var==pred_clr).OnlyEnforceIf(bool_var)
        model.Add(node_var!=pred_clr).OnlyEnforceIf(bool_var.Not())

    c_count = sum(bool_vars)
    model.Maximize(c_count)
    
    
    status = solver.Solve(model)
    if status == cp_model.FEASIBLE:
        return {
            'state': 'feasible',
            'soln': [solver.Value(x) for x in node_vars],
            'overlap': solver.Value(c_count)
        }
    elif status == cp_model.OPTIMAL:
        return {
            'state': 'optimal',
            'soln': [solver.Value(x) for x in node_vars],
            'overlap': solver.Value(c_count)
        }
    else:
        return -1
    
    



# Create sudoku_graph
def sudoku_graph(n, block_shape):
    edges = set()

    num_cells, board_size = n*n, n
    block_x,block_y = block_shape[0], block_shape[1]

    grid = np.array(range(board_size*board_size)).reshape(board_size,board_size) 

    for i in range(num_cells):
        row, col = i // board_size, i % board_size 
        # same row and col
        row_src = row * board_size 
        col_src = col
        for _ in range(board_size):
            if row_src > i:
                edges.add((row_src, i))
            if col_src > i:
                edges.add((col_src, i))
            row_src += 1
            col_src += board_size 

        # same grid

        b_row = (i//board_size)//block_x 
        b_col = (i%board_size)//block_y

        block_values = grid[block_x*b_row:block_x*(b_row+1),block_y*b_col:block_y*(b_col+1)].flatten()       
        for n in block_values: 
            if n>i:
                edges.add((n,i))
            elif n<i:
                edges.add((i,n))


    g = nx.Graph(sorted(edges))
    return g
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def worker(worker_ind, worker_shares, is_sudoku, return_dict):

    worker_dicts = []
    for x,y in worker_shares:
        
        query = x['query'].astype('int')
        target = x['target'].astype('int')
        pred = y.numpy().astype('int')
        
        if not is_sudoku:
            query = query[:x['num_nodes']]
            target = target[:x['num_nodes']]
            pred = pred[:x['num_nodes']]
            
        k = int(math.sqrt(x['query'].shape[0])) if is_sudoku else x['chromatic_num']
        edges = sudoku_graphs[k].edges() if is_sudoku else [tuple(edge) for edge in x['edges'] if edge[0]!=edge[1]]

        
        res = close_valid_soln(edges, query, pred, k, timeout=args.timeout)
        
        # No soln found: use the soln given in query file
        if res == -1:
            query_dict = {'Soln state':'Pre-computed', 'Overlap':(pred==target).sum(), 
                        'Given': (query>0).sum(), 'Masked': (query==0).sum()}
        else:
            query_dict = {'Soln state':res['state'], 'Overlap':res['overlap'], 
                        'Given': (query>0).sum(), 'Masked': (query==0).sum()}
            assert(sum([res['soln'][i]==res['soln'][j] for (i,j) in edges]) == 0)
            
        worker_dicts.append(query_dict)
    return_dict[worker_ind] = worker_dicts
    
    
# Divide (query, pred) pairs amongst all workers
def par_search(is_sudoku, eval_list):
    
    bsz = math.ceil(len(eval_list)/args.num_cores)
    worker_shares = [(bsz*i, min(bsz*(i+1), len(eval_list))) for i in range(args.num_cores)]
    worker_share = [eval_list[i:j] for i,j in worker_shares]
    
    with multiprocessing.Manager() as manager: 

        return_dict = manager.dict() 
        proc_list = [multiprocessing.Process(target=worker, args=(i, worker_share[i], 
                                                                  is_sudoku, return_dict)) 
                                                             for i in range(args.num_cores)]

        _ = [p.start() for p in proc_list]
        _ = [p.join() for p in proc_list]
        final_dict = return_dict.copy()
    return final_dict
    

    
# LOAD QUERIES AND PREDICTIONS
qry_data = pkl.load(open(args.test_file,'rb'))
print('Query data stats:',len(qry_data), qry_data[0].keys())

pred_data = pkl.load(open(args.pred_file,'rb'))
print('Pred data batches',len(pred_data))


# Create the list of (query, prediction) pairs

sudoku_graphs = {}
eval_list = []
num_batches = len(pred_data)    # Pred data is stored in batches
for bs in range(num_batches):
    num_queries = pred_data[bs][0].shape[0]
    for i in range(num_queries):
        q_id = pred_data[bs][2]['qid'][i,0]
        qry, pred = qry_data[q_id], pred_data[bs][2]['pred'][i,:,-1]
        
        if args.task == 'sudoku':
            board_size = int(math.sqrt(qry['query'].shape[0]))
            if board_size not in sudoku_graphs:
                sudoku_graphs[board_size] = sudoku_graph(board_size, block_shapes[board_size])
        
        eval_list.append((qry, pred))

        
# Get approx. pointwise accuracy with multiprocessing
tic = time.time()
eval_dict = par_search(args.task=='sudoku', eval_list)
print('Time Taken', time.time()-tic)


def get_stats(eval_dict):
    eval_l = []
    for v in eval_dict.values():
        eval_l += v

    pw = np.array([x['Overlap']/(x['Given'] + x['Masked']) for x in eval_l]).mean()
    ca = np.array([x['Overlap']==(x['Given'] + x['Masked']) for x in eval_l]).mean()
    states = [x['Soln state'] for x in eval_l]
    stats = {'pw': round(pw,5), 'ca': round(ca,5), 'state_counter': Counter(states)}
    return stats  
    
stats = get_stats(eval_dict)
print(stats)
    
    