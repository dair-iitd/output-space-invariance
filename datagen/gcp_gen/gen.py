from ortools.sat.python import cp_model
import networkx as nx
import time, argparse, os
import numpy as np
import random, copy
import pdb, pickle, itertools, multiprocessing
import pandas as pd
import glob
from collections import Counter

## Sample usage: python gen.py --chromatic-num 5 --num-puzzles 300 --mask-low 60 --mask-high 70 --nodes-low 90 --nodes-high 120  --num-cores 5 --save-path temp-puzzles.pkl

parser = argparse.ArgumentParser()

parser.add_argument('--chromatic-num',type=int, help='Chromatic number of graphs to be generated')
parser.add_argument('--num-puzzles',type=int, help='Numer of puzzles')
parser.add_argument('--mask-low',type=int, help='Min percentage of masked node-colors')
parser.add_argument('--mask-high',type=int, help='Max percentage of masked node-colors')
parser.add_argument('--nodes-low',type=int, help='Min nodes')
parser.add_argument('--nodes-high',type=int, help='Max nodes')

parser.add_argument('--num-cores',type=int, help='cores to be used', default=1)
parser.add_argument('--timeout',type=float, help='timeout to find a graph', default=2)

parser.add_argument('--data-dir',type=str, help='Data directory to dump worker logs in', default='worker-temp')
parser.add_argument('--save-path',type=str, help='Pickle file in which puzzles are saved', default='')
                    
# timeout_start = 2
# timeout_mult = 1
# timeout_max = 60


args = parser.parse_args()
os.makedirs(args.data_dir,exist_ok=True)


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        for v in self.__variables:
            print('%s=%i' % (v, self.Value(v)), end=' ')
        print()

    def solution_count(self):
        return self.__solution_count
    

def gcp(g,k,assign,timeout=-1,make_equitable=False):
    """Minimal CP-SAT example to showcase calling the solver."""
   
    model, solver = cp_model.CpModel(), cp_model.CpSolver()
    if timeout!=-1:
        solver.parameters.max_time_in_seconds = timeout
                
    node_vars = [model.NewIntVar(1, k,'node'+str(i)) for i in range(g.number_of_nodes())]
    _ = [model.Add(node_vars[i]!=node_vars[j]) for i,j in g.edges()]
   
    if make_equitable:
        alpha = g.number_of_nodes()//k
        for c in range(1,k+1):
            bool_vars = [model.NewBoolVar('{}_{}'.format(c,i)) for i in range(len(node_vars))]

            for node_var, bool_var in zip(node_vars,bool_vars):
                model.Add(node_var==c).OnlyEnforceIf(bool_var)
                model.Add(node_var!=c).OnlyEnforceIf(bool_var.Not())

            c_count = sum(bool_vars)
            model.Add(alpha<=c_count)
            model.Add(c_count<=alpha+1)
        
    
    status = solver.Solve(model)
    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL :
        soln1 = [solver.Value(x) for x in node_vars]
        return 1, soln1
    elif status == cp_model.INFEASIBLE:
        return 0, None
    else:
        return -1, None
    

    
seed = 10
random.seed(seed)




all_puzzles = {}


def get_prob_constraints(num_nodes):
    
    prob_constraints = {
            55 : { 4:(0.1,0.2), 5:(0.2,0.25), 6:(0.2,0.25), 7:(0.325,0.375)}, 
            70 : { 4:(0.05,0.1), 5:(0.1,0.2), 6:(0.15,0.25), 7:(0.275,0.325)}, 
            80 : { 4:(0.05,0.1), 5:(0.1,0.2), 6:(0.17,0.2), 7:(0.22,0.30)}, 
            100 : { 4:(0.05,0.1), 5:(0.075,0.12), 6:(0.15,0.18), 7:(0.23,0.275)}, 
            130 : { 4:(0.02,0.05), 5:(0.075,0.1), 6:(0.12,0.16), 7:(0.2,0.275)}, 
            150 : { 4:(0.02,0.05), 5:(0.05,0.075), 6:(0.12,0.15), 7:(0.3,0.4)}, 
        }
    
    for max_nodes in prob_constraints:
        if num_nodes <= max_nodes:
            return prob_constraints[max_nodes]
    

# mask_range = (30,70)
# nodes_range = (40,100)
# k_list = [4]
                   
# samples_per_bucket = 10
# num_workers = 10
                    
mask_range = (args.mask_low,args.mask_high)
nodes_range = (args.nodes_low,args.nodes_high)

mask_list = [i for i in range(mask_range[0],mask_range[1]+1)]
nodes_list = [i for i in range(nodes_range[0],nodes_range[1]+1)]
                    
num_buckets = (args.mask_high - args.mask_low + 1)*(args.nodes_high - args.nodes_low + 1)
samples_per_bucket = args.num_puzzles//num_buckets + 1
num_workers = args.num_cores
print(f'Num buckets: {num_buckets}, Samples per bucket: {samples_per_bucket}')


def worker(worker_ind, return_dict):
    
    worker_log = f'{args.data_dir}/log-{worker_ind}'
    open(worker_log, "w").close()
    
    log_write = open(worker_log,'w')
    worker_puzzles = []
    
    worker_mask_list = [mask_list[i] for i in range(worker_ind,len(mask_list),num_workers)]
    expected_count = samples_per_bucket*len(worker_mask_list)*len(nodes_list)
    last_count = 0
    for mask_percentage, num_nodes in itertools.product(worker_mask_list, nodes_list):
                    
        k = args.chromatic_num
       
        
        tic = time.time()
        bucket_puzzles = []
        need_loosen, need_tighten = 0, 0
        while len(bucket_puzzles)!=samples_per_bucket:
            prob_range = get_prob_constraints(num_nodes)
            p = np.random.uniform(prob_range[k][0],prob_range[k][1])
            g = nx.fast_gnp_random_graph(num_nodes,p)

            # Check if graph is (k-1) colorable
            stat,soln = gcp(g,k-1,[],args.timeout)
            if stat!=0: # Graph is either (k-1)-colorable or we don't know, indicate that more edges might be needed
                need_tighten+=1
                continue

            # Check if graph is k colorable
            stat,soln = gcp(g,k,[],args.timeout)
            if stat!=1: # Graph is either not k-colorable or we don't know, indicate that fewer edges might be needed
                need_loosen+=1
                continue

            mask_nodes = int(num_nodes*mask_percentage/100)
            query = soln.copy()
            for i in np.random.permutation(num_nodes)[:mask_nodes]:
                query[i] = 0 
            puzzle_dict = {'edges': list(g.edges()), 'target': soln, 'query': query, 'chromatic_num': k,
                          'num_nodes': len(soln), 'num_edges': g.number_of_edges()}
            bucket_puzzles.append(puzzle_dict)

        # Pad graph to make numpy arrays
        max_nodes = nodes_range[1] + 1
        max_edges = (max_nodes*max_nodes)//4

        for data in bucket_puzzles:
            pad_nodes = max_nodes - len(data['target'])
            data['target'] = np.array(data['target']+[1]*pad_nodes)
            data['query'] = np.array(data['query']+[1]*pad_nodes)
            data['target_set'] = np.copy(data['target']).reshape(1,data['target'].shape[0])

            pad_edges = max_edges - len(data['edges'])
            data['edges'] = np.array(data['edges']+[(0,0)]*pad_edges)

        toc = time.time()

        log_msg = 'K: {} | Mask: {} |  Nodes: {} | Samples: {} in Time: {}s\n'.format(k,mask_percentage,num_nodes,samples_per_bucket,toc-tic)
        log_msg += 'Need Tighten: {} , Need Loosen: {}, Timeout: {}\n\n'.format(need_tighten, need_loosen, args.timeout)
        log_write.write(log_msg)
        print("So far: {}. Expected: {}".format(len(worker_puzzles), expected_count), file = log_write)
        log_write.flush()
    
        worker_puzzles += bucket_puzzles
        if len(worker_puzzles) - last_count > expected_count//samples_per_bucket:
            last_count = len(worker_puzzles)
            print("Dump {} sample. Worker id: {} ".format(len(worker_puzzles), worker_ind),file = log_write)
            file_name = f'{args.data_dir}/worker_{worker_ind}_{len(worker_puzzles)}.pkl'
            with open(file_name, 'wb') as fh:
                pickle.dump(worker_puzzles,fh)

    
    file_name = f'{args.data_dir}/worker_{worker_ind}_{len(worker_puzzles)}.pkl'
    with open(file_name, 'wb') as fh:
        pickle.dump(worker_puzzles,fh)
        
    return_dict[worker_ind] = worker_puzzles


tic = time.time()
with multiprocessing.Manager() as manager: 
    
    return_dict = manager.dict() 
    proc_list = [multiprocessing.Process(target=worker, args=(i, return_dict)) for i in range(num_workers)]
    
    _ = [p.start() for p in proc_list]
    _ = [p.join() for p in proc_list]
    final_dict = return_dict.copy()
         
print(time.time() - tic)
print(type(final_dict))



# Write puzzles to save path sorted by mask-cells
if args.save_path!='':
    k_puzzles = []
    for name in glob.glob(args.data_dir+'/*.pkl'):
        with open(name,'rb') as fh:
            dat = pickle.load(fh)
            k_puzzles+=dat
    sorted_puzzles = sorted(k_puzzles, key = lambda x: (x['num_nodes'], np.sum(x['query']==0)))
       
    print('Node buckets:', Counter([x['num_nodes'] for x in sorted_puzzles]))
    print('Mask buckets:', Counter([round((x['query'][:x['num_nodes']]==0).mean()*100) for x in sorted_puzzles]))
    print(f'Writing to {args.save_path}')
    with open(args.save_path,'wb') as f:
        pickle.dump(sorted_puzzles, f)




