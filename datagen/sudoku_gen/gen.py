from copy import deepcopy
import numpy as np
import pandas as pd
import os, argparse
import pickle
from collections import Counter
from collections import defaultdict

from tqdm.auto import tqdm
import pandas as pd
import math, time, random
from datetime import datetime as dt
import multiprocessing
from itertools import product
from IPython.core.debugger import Pdb
import sys
import multiprocessing
import glob


## Sample usage: python gen.py --board-size 8 --num-puzzles 50 --mask-low 19 --mask-high 43 --unq-flag 1 --num-cores 5 --save-path temp-puzzles.pkl

parser = argparse.ArgumentParser()

parser.add_argument('--board-size',type=int, help='Board Size of sudokus to be generated')
parser.add_argument('--num-puzzles',type=int, help='Numer of puzzles')
parser.add_argument('--mask-low',type=int, help='Min mask boxes')
parser.add_argument('--mask-high',type=int, help='Max mask boxes')

parser.add_argument('--num-cores',type=int, help='cores to be used', default=1)
parser.add_argument('--max-trials',type=int, help='Tries to find a sudoku', default=1000000)
parser.add_argument('--puzzle-dump',type=int, help='Each core updates after generating puzzle_dump puzzles', default=1)
parser.add_argument('--unq-flag',type=int, help='Ensure sudoku has unique soln', default=0)

parser.add_argument('--data-dir',type=str, help='Data directory to dump worker logs in', default='worker-temp')
parser.add_argument('--save-path',type=str, help='Pickle file in which puzzles are saved', default='')


args = parser.parse_args()
os.makedirs(args.data_dir,exist_ok=True)

# Constants
template_dict = {(2, 4): './gss/templates/template8.gss',
                 (3, 3): './gss/templates/template9.gss',
                 (2, 5): './gss/templates/template10.gss',
                 (2, 6): './gss/templates/template12_1.gss',
                 (2, 7): './gss/templates/template14.gss',
                 (3, 5): './gss/templates/template15.gss',
                 (4, 4): './gss/templates/template16.gss',
                }

block_shapes = {
    8: (2,4), 9: (3,3), 10: (2,5), 12: (2,6), 15: (3,5), 16: (4,4), 24: (4,6), 25: (5,5)
}


# puzzle-spec-list contains individual puzzle specifications
puzzle_spec_list = []

# jobs_puzzle_spec_list contains lists of puzzle specifications for each core they need to generate
jobs_puzzle_spec_list = [{'core_ind': i, 'puzzle_spec_list':[]} for i in range(args.num_cores)]


# See how many puzzles already exist in the dump
def get_puzzle_counts(start_x=0, end_x=2000):
    all_puzzles = {}
    for bs in [args.board_size]:
        all_puzzles[bs] = {}
        for i in range(start_x,end_x):
            all_puzzles[bs][i] = 0
        all_puzzles[bs]['all'] = 0

    for name in glob.glob(args.data_dir+'/dump_*'):
        with open(name,'rb') as fh:
            dat = pickle.load(fh)
            for x in dat:
                bs, mask_cells = x['spec']['board_size'], x['spec']['mask_cells']
                if bs in all_puzzles:
                    all_puzzles[bs][mask_cells] += 1
                    all_puzzles[bs]['all'] += 1   
    return all_puzzles



# Generate list having specifications of puzzles to be generated
all_puzzles = get_puzzle_counts()
num_cells = args.board_size*args.board_size
mask_cell_range = (args.mask_low, args.mask_high + 1)
puzzles_per_mask_cell = args.num_puzzles//(args.mask_high - args.mask_low + 1) + 1
if args.board_size == 8:
    print(puzzles_per_mask_cell)
for mask_cells in range(*mask_cell_range):
    puzzles_left = max(0,puzzles_per_mask_cell - all_puzzles[args.board_size][mask_cells])    # Check how many puzzles left to do
    puzzle_spec_list += [{'board_size': args.board_size, 'mask_cells': mask_cells, 'unq': args.unq_flag} for _ in range(puzzles_left)]


for i, puzzle_spec in enumerate(puzzle_spec_list):
    jobs_puzzle_spec_list[i%args.num_cores]['puzzle_spec_list'].append(puzzle_spec)
    
    
class Sudoku_Solver:
    
    def __init__(self,block_size, num_required_solutions = 1):
        self.block_m, self.block_n = block_size
        self.board_size = self.block_m * self.block_n
        self.solutions = []
        self.num_required_solutions = min(num_required_solutions,100)
        
        
    def find_empty_locations(self,arr): 
        ind = list(zip(*np.nonzero(arr==0)))
        return ind

    def find_suitable_values(self,arr,row,col):
        unallowed_values = set(arr[row])
        unallowed_values = unallowed_values.union(set(arr[:,col]))
        block_r = row-row%self.block_m
        block_c = col-col%self.block_n
        unallowed_values = unallowed_values.union(set(arr[block_r:block_r+self.block_m,block_c:block_c+self.block_n].flatten()))
        return_list = list(set(range(1,self.board_size+1)).difference(unallowed_values))

        np.random.seed(dt.now().microsecond)
        
        np.random.shuffle(return_list)
        return return_list
    
    # Takes a partially filled-in grid and attempts to assign values to 
    # all unassigned locations in such a way to meet the requirements 
    # for Sudoku solution (non-duplication across rows, columns, and boxes) 
    # return codes: 1 for successful finding of required solutions, 0 for not enough solutions, -1 for timeout
    def solve_sudoku(self,arr,empty_locations, start_ind , timeout, start_time): 
        
        if (start_ind == len(empty_locations)):        
            self.solutions.append(deepcopy(arr))
            if len(self.solutions)>= self.num_required_solutions:
                return 1
            return 0

        row,col = empty_locations[start_ind]

        # consider all digits 
        suitable_values = self.find_suitable_values(arr,row,col)
        for num in suitable_values: 
            
            if timeout >=0 and (time.time() - start_time > timeout):
                return -1
            
            # make tentative assignment 
            arr[row][col]=num 
            
            # return, if success, ya! 
            self.solve_sudoku(arr,empty_locations, start_ind + 1, timeout, start_time)
            
            # failure, unmake & try again 
            arr[row][col] = 0
            
            
            if len(self.solutions)>= self.num_required_solutions:
                return 1
        
        return 0
        
            
        
    # return codes: requried number of solutions if they are found, 0 for not enough solutions, -1 for timeout
    def solve(self,arr, num_required_solutions=1, timeout = -1):
        back_up_num_required_solutions = self.num_required_solutions
        if num_required_solutions > 0:
            self.num_required_solutions = min(num_required_solutions,100)
        #
        self.solutions = []
        query = deepcopy(arr.reshape(self.board_size,self.board_size))
        empty_locations = self.find_empty_locations(query)
        status = self.solve_sudoku(query,empty_locations,0,timeout, time.time())
        self.num_required_solutions = back_up_num_required_solutions
        if(status==1):
            assert(len(self.solutions) == num_required_solutions)
            return [x.flatten() for x in self.solutions]
        
        return status

    
def generate_query_from_solution(solution,blank_cells = 0):
    
    query = deepcopy(solution)
    l   = len(query)
    board_size = int(math.sqrt(l))
    mask_essential = np.random.choice(l,blank_cells, replace=False)
    query[mask_essential]=0
    while len(set(range(board_size+1)).difference(set(np.unique(query)))) >= 2:
        query = deepcopy(solution)
        np.random.seed(dt.now().microsecond)
        mask_essential = np.random.choice(l,blank_cells, replace=False)
        query[mask_essential]=0
        
    target = solution
    target_set = np.expand_dims(deepcopy(solution), 0)
    return query


def solve_puzzle_gss(block_shape, query, job_id):         
    m, n = block_shape
    board_size = m*n
    if block_shape not in template_dict:
        print('GSS cannot be used with the given block shape since no template exists. No template might exist because the board size is too big and finding unique solutions for the same requires enormous compute')
        exit(-1)
    template_file = template_dict[block_shape]
    
    scratch_temp, scratch = 'scratch_template_{}.txt'.format(job_id), 'scratch_{}.txt'.format(job_id)
    query = query.reshape(board_size,board_size)
    query_sudoku = [' '.join(y)+'\n' for y in query.astype(int).astype(str).reshape(board_size,board_size)]
    
    with open(template_file, 'r') as f:
        empty_template = f.readlines()
        ind = next(i for i, x in enumerate(empty_template) if x.strip() == '<sudoku>')
        filled_template = empty_template[:ind+1] + query_sudoku
    
    with open(scratch_temp,'w') as f:
        f.writelines(filled_template)
        
    os.system('./gss/gss {} -q > {}'.format(scratch_temp, scratch))
    
    with open(scratch, 'r') as f:
        
        solution_ore = f.readlines()
        if(not(len(solution_ore) == board_size or len(solution_ore)%(board_size + 1) == 1)):
            print(solution_ore)
        assert(len(solution_ore) == board_size or len(solution_ore)%(board_size + 1) == 1)
        
        # Case of single solution
        if len(solution_ore) == board_size:
            target_set = np.zeros((1, board_size*board_size))
            target_set[0] = np.array([list((map(int, x.split()))) for x in solution_ore]).flatten()
        # Case of multiple solution
        else:
            solution_ore = solution_ore[1:]
            n_sols = len(solution_ore)//(board_size+1)
            target_set = np.zeros((n_sols, board_size*board_size))

            for i in range(n_sols):
                sudoku_i = solution_ore[i*(board_size+1)+1:(i+1)*(board_size+1)]
                target_set[i] = np.array([list((map(int, x.split()))) for x in sudoku_i]).flatten()
        
    os.system('rm {}'.format(scratch_temp))
    os.system('rm {}'.format(scratch))
    
    return target_set


def generate_puzzle(puzzle_specs, job_id):
    board_size = puzzle_specs['board_size']
    block_shape = block_shapes[board_size]
    
    # Generate a target
    test_solver = Sudoku_Solver(block_size = block_shape)
    blank_board = np.zeros((board_size, board_size))

    for trial_count in range(args.max_trials):
        target = -1
        while(target == 0 or target == -1):
            target = test_solver.solve(blank_board, timeout = 1)     
        assert(len(target)==1)

        # Generate a query corresponding to the target
        target, query = target[0], generate_query_from_solution(target[0], puzzle_specs['mask_cells'])

        # Verify whether the query is unique or not
        if puzzle_specs['unq']:
            target_set = solve_puzzle_gss(block_shape, query, job_id)
            if (target_set.shape[0] == 1):
                assert (target_set[0] == target).all()
                return {'target': target, 'query': query}
        else:
            return {'target': target, 'query': query}
    return None


def puzzle_generator(job_id, puzzle_spec_list):
    
    tic = time.time()
    
#     job_id, puzzle_spec_list = job_puzzle_spec['core_ind'], job_puzzle_spec['puzzle_spec_list']
    
    dump_file = os.path.join(args.data_dir, 'dump_{}.pkl'.format(job_id))
    log_file = os.path.join(args.data_dir, 'LOG_{}'.format(job_id))
    log_out = open(log_file, "a")
    
    
    def update_dump(puzzles):
        all_puzzles = []
        if os.path.exists(dump_file):
            with open(dump_file,'rb') as fh:
                all_puzzles = pickle.load(fh)
        with open(dump_file,'wb') as fh:
            pickle.dump(all_puzzles+puzzles, fh)
    
    puzzles, dump_count = [], 0
    for puzzle_spec in puzzle_spec_list:
        bs, mask_cells = puzzle_spec['board_size'], puzzle_spec['mask_cells']
        num_cells, block_shape = bs*bs, block_shapes[bs]
        puzzle = generate_puzzle(puzzle_spec, job_id)
        if puzzle:
            puzzle['spec'] = puzzle_spec
            puzzles.append(puzzle)
            dump_count+=1
            log_out.write("Found puzzle with specs {} \n".format(puzzle_spec))
        else:
            log_out.write("Failed to find puzzle with specs {} \n".format(puzzle_spec))
        log_out.flush()
        if dump_count == args.puzzle_dump:
            update_dump(puzzles)
            puzzles, dump_count = [], 0
            
    update_dump(puzzles)
            
    log_out.close()
            
job_list = [(x['core_ind'], x['puzzle_spec_list']) for x in jobs_puzzle_spec_list]

np.random.seed(42)
random.seed(42)
with multiprocessing.Pool(processes=args.num_cores) as pool:
    results = pool.starmap(puzzle_generator, job_list)
    
    
all_puzzles = get_puzzle_counts(args.mask_low-5, args.mask_high+5)
print(all_puzzles)


# Write puzzles to save path sorted by mask-cells
if args.save_path!='':
    bs_puzzles = []
    for name in glob.glob(args.data_dir+'/dump_*'):
        with open(name,'rb') as fh:
            dat = pickle.load(fh)
            for x in dat:
                bs, mask_cells = x['spec']['board_size'], x['spec']['mask_cells']
                if bs!=args.board_size:
                    continue
                if (mask_cells>=args.mask_low) and (mask_cells<=args.mask_high):
                    puzzle = {'query': np.copy(x['query']), 
                              'target': np.copy(x['target']), 
                              'target_set': np.expand_dims(np.copy(x['target']),axis = 0),
                              'count': 1
                             }
                    bs_puzzles.append(puzzle)
    sorted_puzzles = sorted(bs_puzzles, key = lambda x: np.sum(x['query']==0))
    print(f'Writing to {args.save_path}')
    with open(args.save_path,'wb') as f:
        pickle.dump(sorted_puzzles, f)

