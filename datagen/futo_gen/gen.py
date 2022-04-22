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
import glob


## Sample usage: python gen.py --board-size 8 --num-puzzles 50 --mask-low 30 --mask-high 70 --save-path temp-puzzles.pkl

parser = argparse.ArgumentParser()

parser.add_argument('--board-size',type=int, help='Board Size of futoshikis to be generated')
parser.add_argument('--num-puzzles',type=int, help='Numer of puzzles')
parser.add_argument('--mask-low',type=int, help='Min percentage of mask boxes')
parser.add_argument('--mask-high',type=int, help='Max percentage of mask boxes')

parser.add_argument('--num-cores',type=int, help='cores to be used', default=1)
parser.add_argument('--data-dir',type=str, help='Data directory to dump worker logs in', default='worker-temp')
parser.add_argument('--save-path',type=str, help='Pickle file in which puzzles are saved', default='')

args = parser.parse_args()
os.makedirs(args.data_dir,exist_ok=True)

class Futo_Solver:
    
    def __init__(self,board_size, num_required_solutions = 1,log_file=None):
        self.board_size = board_size
        self.solutions = []
        self.num_required_solutions = min(num_required_solutions,100)
        self.log_file = log_file
        
    def find_empty_locations(self,arr): 
        ind = list(zip(*np.nonzero(arr==0)))
        return ind

    def find_suitable_values(self,arr,row,col):
        unallowed_values = set(arr[row])
        unallowed_values = unallowed_values.union(set(arr[:,col]))
        return_list = list(set(range(1,self.board_size+1)).difference(unallowed_values))

#         np.random.seed(dt.now().microsecond)
        
        np.random.shuffle(return_list)
        return return_list
    
    def check_validity_of_partial_solution(self,grid,constraints):
        for x in range(len(grid)):
            row = grid[x][grid[x] != 0]
            if len(row) != len(set(row)):
                return False
            
            col = grid[:,x][grid[:,x] != 0]
            if len(col)!=len(set(col)):
                return False
            
        if constraints is None:
            return True
        #
        for this_constraint in constraints:
            left_ind,right_ind = this_constraint
            x_left, y_left = left_ind // self.board_size, left_ind % self.board_size
            x_right, y_right = right_ind // self.board_size, right_ind % self.board_size
            if grid[x_left,y_left] != 0 and \
                grid[x_right, y_right] != 0 and \
                grid[x_left,y_left] > grid[x_right, y_right]:
                    return False
        #
        return True
        
       
    # Takes a partially filled-in grid and attempts to assign values to 
    # all unassigned locations in such a way to meet the requirements 
    # for Futo solution (non-duplication across rows, columns) 
    # return codes: 1 for successful finding of required solutions, 0 for not enough solutions, -1 for timeout
    def solve_futo(self,arr,empty_locations, start_ind , timeout, start_time, constraints=None, optimize=False): 
        
        if optimize and not self.check_validity_of_partial_solution(arr,constraints):
            return 0
        #
        if (start_ind == len(empty_locations)):
            if self.check_validity_of_partial_solution(arr,constraints):
                self.solutions.append(deepcopy(arr))
            if len(self.solutions)>= self.num_required_solutions:
                return 1
            return 0


        if (start_ind == 0) and optimize:
            #entry into the recursion. Order the empty locations in the increasing order of suitable values
            num_suitable_values_dict = {}
            for row,col in empty_locations:
                num_suitable_values_dict[(row,col)] = len(self.find_suitable_values(arr,row,col))
            #
            empty_locations.sort(key= lambda x: num_suitable_values_dict[x])
            #Pdb().set_trace()
            print('#iterations:',np.product(np.array(list(num_suitable_values_dict.values())).astype('float')),
                 file = self.log_file)
        #
        row,col = empty_locations[start_ind]

        # consider all digits 
        suitable_values = self.find_suitable_values(arr,row,col)
        for num in suitable_values: 
            if timeout > 0 and (time.time() - start_time > timeout):
                return -1
            
            # make tentative assignment 
            arr[row][col]=num 
            
            # return, if success, ya! 
            self.solve_futo(arr,empty_locations, start_ind + 1, timeout, start_time, constraints, optimize)
            
            # failure, unmake & try again 
            arr[row][col] = 0
            
            if len(self.solutions)>= self.num_required_solutions:
                return 1
        
        return 0

    
# Given a solved board, returns a query
def futo_query_from_solution(solution,mask_pct = 0.6):
    query = deepcopy(solution)
    l = len(query)
    board_size = int(math.sqrt(l))
    mask_essential = np.random.choice(l,int(mask_pct*l), replace=False)
    query[mask_essential]=0
    while len(set(range(board_size+1)).difference(set(np.unique(query)))) >= 2:
        query = deepcopy(solution)
        mask_essential = np.random.choice(l,int(mask_pct*l), replace=False)
        query[mask_essential]=0
    
    return query

# Given a solved board, returns a few less-than constraints
def get_lt_edges(target_in, num):
    
    board_size = int(math.sqrt(target_in.shape[0]))
    target = target_in.tolist() + [-1]
    
    def xy_ind(i,j):
        if min(i,j)>=0 and max(i,j)<board_size:
            return i*board_size + j
        return -1
    
    all_lt_edges = []
    for i,j in product(range(board_size),range(board_size)):
        curr_ind = xy_ind(i,j)
        up_ind, down_ind = xy_ind(i-1,j), xy_ind(i+1,j)
        left_ind, right_ind = xy_ind(i,j-1), xy_ind(i,j+1)
        
        nbrs = [up_ind, down_ind, left_ind, right_ind]
        
        # if current position is less than a neighbour, add lt edge: boundary cases are covered by adding -1 at end
        all_lt_edges += [(curr_ind, nbr_ind) for nbr_ind in nbrs if target[curr_ind] < target[nbr_ind]]
    lt_edges = np.array(random.sample(all_lt_edges,num))
    return lt_edges
  
# Given size of board, generates a futo example
# last argument is needed to pad lt-edges 
def get_futo_pt(futo_size, mask_pct, pad_lt_num, unique=False, timeout=10,log_file = None):
    futo_maker = Futo_Solver(futo_size)
    
    blank_futo = np.zeros((futo_size,futo_size))
    blank_loc = futo_maker.find_empty_locations(blank_futo)
    futo_maker.solve_futo(blank_futo, blank_loc, 0, 2, time.time())
    
    target = futo_maker.solutions[0].flatten()
    query = futo_query_from_solution(target,mask_pct)
    lt_edges_unpad = get_lt_edges(target, 2*futo_size)    
    status = 0
    success = True
    round_num = 1
    if unique:
        time_left = True
        success = False
        start_time = time.time()
        
        while time_left and not success:
            futo_checker = Futo_Solver(futo_size,num_required_solutions = 2,log_file = log_file)
            query_copy = deepcopy(query.reshape(futo_size, futo_size))
            blank_loc = futo_checker.find_empty_locations(query_copy)
            #start_time = time.time()

            status = futo_checker.solve_futo(query_copy,blank_loc,0,
                                         timeout=timeout,start_time = start_time,
                                         constraints=lt_edges_unpad,
                                        optimize=True)

            if status == -1:
                #Timeout. Check how many solutions u found
                if len(futo_checker.solutions) == 1:
                    #print("Partial Success: Could not find 2 solutions in given time")
                    if all(futo_checker.solutions[0].flatten() == target):
                        success  = True
                else:
                    #print("Could not find any solution!")
                    pass
            elif (status == 0):
                if len(futo_checker.solutions) == 1:
                    #print("Success!")
                    success = True
                else:
                    print("NA case: should not be here. #solutions found:", len(futo_checker.solutions), file = log_file)
            else:
                pass
                #print("Failure: Found 2 solutions")
            #
            #Try with another masking if you have not consumed even half the time.
            #In the next version, we will try to increase the number of constraints to make it unique.
            time_left = (time.time() - start_time ) < 0.5*timeout
            if time_left and not success:
                round_num += 1
                #print("Try again with another random masking. Round",round_num)
                #Pdb().set_trace()
                query = futo_query_from_solution(target,mask_pct)
                num_constraints = 2*futo_size
                lt_edges_unpad = get_lt_edges(target, num_constraints)    
      
    #
    if success:
        lt_edges = np.zeros((pad_lt_num,2))
        lt_edges[:lt_edges_unpad.shape[0]] = lt_edges_unpad
        futo_dict = {
            'lt_edges': lt_edges.astype(int), 
            'target': target.astype(int),
            'query': query.astype(int),
            'n': futo_size,
            'target_set': target.copy().reshape(1,target.shape[0]).astype(int),
            'num_lt_edges': 2*futo_size,
            'count': 1,
            'status': status,
        }
        return futo_dict, status, round_num
    
    return None,status, round_num

def generate_futo_data(samples_per_bucket, board_size, start_mask, end_mask, read_only = False):
    # Choose from train, val and test

    log_file = f'{args.data_dir}/futo_bs-{board_size}_mask-{start_mask}to{end_mask}.LOG'
    partial_data_file_template = 'futo_bs-{}_mask-{}.pkl'

    mask_list = [i for i in range(start_mask,end_mask+1)]

    #all_puzzles = []
    timeout_base = 10
    size_base = 5

    fh = open(log_file, 'a')
    print("__________________________________________________", file = fh)
    all_puzzles_list = []
    for mask_pct in mask_list:
        this_ofile = os.path.join(args.data_dir,partial_data_file_template.format(board_size, mask_pct))
        timeout = timeout_base*(board_size/size_base)**2
        this_puzzle_list = []
        if os.path.exists(this_ofile):
            print("EXISTS: Mask PCT: {}, SIZE: {}, TIMEOUT: {}".format(mask_pct, board_size,round(timeout)), file = fh)
            fh.flush()
            with open(this_ofile,'rb') as this_ofh:
                this_puzzle_list = pickle.load(this_ofh)
        #
        if read_only:
            all_puzzles_list.extend(this_puzzle_list)
            continue
        else:
            start_time  = time.time()
            print("Mask PCT: {}, SIZE: {}, TIMEOUT: {}".format(mask_pct, board_size,round(timeout)), file = fh)
            max_lt_edges = 2*board_size

            partial_success = 0
            success_rounds = defaultdict(int)
            failure_rounds = defaultdict(int)
            this_total_timeout = timeout*(samples_per_bucket - len(this_puzzle_list))
            time_left = True
            #for _ in range(len(this_puzzle_list),samples_per_bucket):
            while (len(this_puzzle_list) < samples_per_bucket) and time_left:
                time_left = ((time.time() - start_time) < this_total_timeout)
                this_puzzle, this_status, this_round_num = get_futo_pt(board_size, mask_pct/100, 
                                                            max_lt_edges, unique=True, timeout=timeout, log_file = fh)
                if this_puzzle is not None:
                    this_puzzle_list.append(this_puzzle)
                    partial_success += int(this_status == -1)            
                    success_rounds[this_round_num] += 1
                else:
                    failure_rounds[this_round_num] += 1
            #
            print("Mask Pct: {}; Futo Size: {}; Found: {}/{}. Partial Success: {}. Time Taken: {} secs".format(mask_pct, 
                                    board_size, len(this_puzzle_list),
                                    samples_per_bucket,partial_success, round(time.time() - start_time,2)), file =fh)
            #
            ntrials = sum(failure_rounds.values())
            ntrials_successful = sum(success_rounds.values())
            print("\t\t\tSuccess rounds. Total successful trials: {}".format(ntrials_successful),"Dict: ", success_rounds, file = fh)
            print("\t\t\tFailure rounds: Total failed Trials: {}".format(ntrials), "Dict: ", failure_rounds, file = fh)
            sys.stdout.flush()
            fh.flush()
            with open(this_ofile,'wb') as ofh:
                pickle.dump(this_puzzle_list,ofh)
            all_puzzles_list.extend(this_puzzle_list)
            
    print('#############################', file = fh)
    fh.close()    
#     if read_only:
    return all_puzzles_list
    
    
    
mask_range = (args.mask_low,args.mask_high)
job_list = []
mask_list = [i for i in range(mask_range[0],mask_range[1]+1)]
samples_per_bucket = args.num_puzzles//len(mask_list) + 1
print(f'Samples per bucket: {samples_per_bucket}')

for this_mask in mask_list:
    kargs = (samples_per_bucket,args.board_size, this_mask, this_mask)
    job_list.append(kargs)
            
            
np.random.seed(42)
random.seed(42)
with multiprocessing.Pool(processes=args.num_cores) as pool:
    results = pool.starmap(generate_futo_data, job_list)
    
all_data = [x for sublist in results for x in sublist]
       
    
# Write puzzles to save path sorted by mask-cells
if args.save_path!='':
    k_puzzles = []
    
    sorted_puzzles = sorted(all_data, key = lambda x: np.sum(x['query']==0))
       
    print('Mask buckets:', Counter([round((x['query']==0).mean()*100) for x in sorted_puzzles]))
    print(f'Writing to {args.save_path}')
    with open(args.save_path,'wb') as f:
        pickle.dump(sorted_puzzles, f)                
        
        
        
  