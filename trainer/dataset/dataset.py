#! /usr/bin/env python3

import numpy as np
import itertools
from torch.utils.data.dataset import Dataset
import torch
import jacinle.random as random
import pickle

import math
from IPython.core.debugger import Pdb
import copy
from jacinle.logging import get_logger, set_output_file

from torch.distributions.categorical import Categorical
from . import futoshiki_data as fd
from . import gcp_data as gd
from . import utils 
TRAIN = 0
DEV = 1
TEST = 2

logger = get_logger(__file__)

__all__ = [
        'FutoshikiDataset','GCPDataset','NLMDataset'
]


class FutoshikiDataset(Dataset):
    """The dataset for futoshiki tasks."""
    def __init__(self,
                             epoch_size,
                             data_size = -1,
                             train_dev_test = TRAIN,
                             data_file = None,
                             args=None):
        super().__init__()
        self.args = args
        self._epoch_size = epoch_size
        self.mode = train_dev_test
        #self._n = 81
        print("In constructor.  {}".format(args.task))
        if train_dev_test == TRAIN:
            mode = 'train'
        elif train_dev_test ==  DEV:
            mode = 'val'
        elif train_dev_test == TEST:
            mode = 'test'

        
        self.outfile = data_file 
        #
        logger.info("data file : {}".format(self.outfile))
        #Pdb().set_trace()
        with open(self.outfile,"rb") as f:
            self.dataset = pickle.load(f)
        
        logger.info("Original datasize: {}".format(len(self.dataset)))
        if data_size != -1:
            self.dataset= self.dataset[:data_size]
        #
        logger.info("Datasize after subselection: {}".format(len(self.dataset)))
        np.random.seed(args.seed)
        self.max_count = 1 
        for i,data in enumerate(self.dataset):
            data['query'] = (data['query']).astype(int)
            if len(data["target_set"])>self.max_count:
                self.dataset[i]["target_set"] = data["target_set"][:self.max_count]
                self.dataset[i]["count"]=self.max_count
            if 'count'  in data:
                this_count = data['count']
            else:
                this_count = data['target_set'].shape[0]
                self.dataset[i]['count'] = this_count
        
        self.max_count += 1
        self.reset_sampler() 
        self.class_graph_cache = {}
        self.kwargs = {}

    def reset_sampler(self):
        self.sampler = Categorical(probs = torch.tensor([1.0 for _ in self.dataset]).float())


    def pad_set(self,target_set):
        pad_counter = self.max_count - len(target_set)
        if pad_counter < 0:
            return target_set[:self.max_count]

        return_set = list(target_set)
        return_set.extend([target_set[-1] for _ in range(pad_counter)])
        return np.array(return_set)
        
    def __getitem__(self, item):
        if self.mode==TRAIN:
            ind = self.sampler.sample().item()
        else:
            ind = item%len(self.dataset)

        data = self.dataset[ind]
        
        data["target"] = data["target_set"][0]
        data["target_set"] = self.pad_set(data["target_set"])
        data['lt_edges'] = data['lt_edges'][:data['num_lt_edges']] 
        data['n'] = data['query'].shape[0]
        data['qid'] = np.array([ind])
        if 'mask' not in data:
            data['mask'] = np.array([1 for _ in range(data['count'])] + [0 for _ in range(data['target_set'].shape[0] - data['count'])]) 
       
        if 'graph' not in data:
            puzzle = (torch.tensor(data['query']), torch.tensor(data['lt_edges'][:data['num_lt_edges']]))
            
            #two options: 
            #A: diff, lt, gt, contain, may_contain
            #B: diff, lt, gt, contain, may_contain, belongs, lt, gt, class_diff
            #Option B triggered by args.cell2cluster_msg_passing_steps > 0 
            data['graph'] = fd.get_hetero_graph(puzzle, self.args,self.kwargs)
        #
            
        return_dict = dict([(k,v) for k,v in data.items()])
        return_dict['graph'] = copy.deepcopy(data['graph'])
        if not self.args.binary_model: 
            return_dict['class_graph'] = copy.deepcopy(self.get_class_graph(data['n']))
        return return_dict

    def get_class_graph(self,n):
        if n not in self.class_graph_cache:
            g = fd.get_class_graph(n,self.args.diff_edges_in_class_graph)
            g.nodes['cluster'].data['id'] = torch.arange(g.number_of_nodes('cluster')).long()
            self.class_graph_cache[n] = g
            
        return self.class_graph_cache[n]
        
    def __len__(self):
        if self.mode==TRAIN:
            return self._epoch_size
        else:
            return len(self.dataset)
       
class GCPDataset(Dataset):
    """The dataset for gcp tasks."""
    def __init__(self,
                             epoch_size,
                             data_size = -1,
                             train_dev_test = TRAIN,
                             data_file = None,
                             args=None):
        super().__init__()
        self.args = args
        self._epoch_size = epoch_size
        self.mode = train_dev_test
        print("In constructor.  {}".format(args.task))
        if train_dev_test == TRAIN:
            mode = 'train'
        elif train_dev_test ==  DEV:
            mode = 'val'
        elif train_dev_test == TEST:
            mode = 'test'

        self.outfile = data_file 
        #
        logger.info("data file : {}".format(self.outfile))
        with open(self.outfile,"rb") as f:
            self.dataset = pickle.load(f)
        
        logger.info("Original datasize: {}".format(len(self.dataset)))
        if data_size != -1:
            self.dataset= self.dataset[:data_size]
        #
        logger.info(" datasize after subselection: {}".format(len(self.dataset)))
        if self.args.task_is_sudoku:
            #Adapt the sudoku dataset to gcp dataset
            logger.info("Convert sudoku dataset to gcp dataset")
            self.dataset = utils.convert_sudoku_to_gcp(self.dataset)
        
        np.random.seed(args.seed)
        self.max_count = 1 
        
        for i,data in enumerate(self.dataset):
            if len(data["target_set"])>self.max_count:
                self.dataset[i]["target_set"] = data["target_set"][:self.max_count]
                self.dataset[i]["count"]=self.max_count
            if 'count'  in data:
                this_count = data['count']
            else:
                this_count = data['target_set'].shape[0]
                self.dataset[i]['count'] = this_count
        
        self.max_count += 1
        self.reset_sampler() 
        self.class_graph_cache = {}


    def reset_sampler(self):
        self.sampler = Categorical(probs = torch.tensor([1.0 for _ in self.dataset]).float())


    def pad_set(self,target_set):
        pad_counter = self.max_count - len(target_set)
        if pad_counter < 0:
            return target_set[:self.max_count]

        return_set = list(target_set)
        return_set.extend([target_set[-1] for _ in range(pad_counter)])
        return np.array(return_set)

    def __getitem__(self, item):
        if self.mode==TRAIN:
            ind = self.sampler.sample().item()
        else:
            ind = item%len(self.dataset)

        data = self.dataset[ind]
        data["target"] = data["target_set"][0]
        data["target_set"] = self.pad_set(data["target_set"])
        data['n'] = data['query'].shape[0]
        data['qid'] = np.array([ind])
        data['is_ambiguous'] = data['target_set'].shape[0] > 1
        if 'mask' not in data:
            data['mask'] = np.array([1 for _ in range(data['count'])] + [0 for _ in range(data['target_set'].shape[0] - data['count'])]) 
      
        if 'graph' not in data:
            puzzle = (torch.tensor(data['query'][:data['num_nodes']]), torch.tensor(data['edges'][:data['num_edges']]))
            data['graph'] = gd.get_hetero_graph(puzzle, data['chromatic_num'], self.args)


        return_dict = dict([(k,v) for k,v in data.items()])
        return_dict['graph'] = copy.deepcopy(data['graph'])
        if not self.args.binary_model: 
            return_dict['class_graph'] = copy.deepcopy(self.get_class_graph(data['chromatic_num']))
        return return_dict
    
    def get_class_graph(self,n):
        if n not in self.class_graph_cache:
            g = gd.get_class_graph(n,self.args)
            g.nodes['cluster'].data['id'] = torch.arange(g.number_of_nodes('cluster')).long()
            self.class_graph_cache[n] = g
            
        return self.class_graph_cache[n]
     
    def __len__(self):
        if self.mode==TRAIN:
            return self._epoch_size
        else:
            return len(self.dataset)



class NLMDataset(Dataset):
    """The dataset for futoshiki and Sudoku tasks."""

    def __init__(self,
                 epoch_size,
                 data_size=-1,
                 train_dev_test=TRAIN,
                 data_file=None,
                 task = 'futoshiki',
                 args=None):
        super().__init__()
        self.args = args
        self._epoch_size = epoch_size
        self.mode = train_dev_test
        self.task = task
        #self.relations = self.get_relations()
        if task == 'futoshiki':
            print("In NLM Futoshiki constructor")
        elif task == 'sudoku':
            print("In NLM Sudoku constructor")
        elif task == 'gcp':
            print(" IN NLM GCP constructor")

        if train_dev_test == TRAIN:
            mode = 'train'
        elif train_dev_test == DEV:
            mode = 'val'
        elif train_dev_test == TEST:
            mode = 'test'

        outfile = data_file
        #
        logger.info("data file : {}".format(outfile))
        # Pdb().set_trace()
        with open(outfile, "rb") as f:
            self.dataset = pickle.load(f)

        #
        if data_size != -1:
            self.dataset= self.dataset[:data_size]
        #
        logger.info("Datasize after subselection: {}".format(len(self.dataset)))
        if self.args.task_is_sudoku:
            #Adapt the sudoku dataset to gcp dataset
            logger.info("Convert sudoku dataset to gcp dataset")
            self.dataset = utils.convert_sudoku_to_gcp(self.dataset)

        np.random.seed(args.seed)
        self.max_count = 1 
        for i,data in enumerate(self.dataset):
            data['query'] = (data['query']).astype(int)
            if self.task in ['gcp','sudoku']:
                data['num_bin_nodes'] = data['num_nodes']*data['chromatic_num']
                data['query'] = data['query'][:data['num_nodes']]
                data['target'] = data['target'][:data['num_nodes']]
                data['target_set'] = data['target_set'][:,:data['num_nodes']]
            elif self.task  == 'futoshiki':
                #Pdb().set_trace()
                data['num_bin_nodes'] = int(round(float(data['n'])**(3.0/1.0)))
                data['num_nodes'] = data['n']*data['n']
                data['chromatic_num'] = int(round(float(data['n'])**(1.0/1.0)))

            if len(data["target_set"])>self.max_count:
                self.dataset[i]["target_set"] = data["target_set"][:self.max_count]
                self.dataset[i]["count"]=self.max_count
            if 'count'  in data:
                this_count = data['count']
            else:
                this_count = data['target_set'].shape[0]
                self.dataset[i]['count'] = this_count
        
        self.max_count += 1
        self.reset_sampler() 
        

    def reset_sampler(self):
        self.sampler = Categorical(probs = torch.tensor([1.0 for _ in self.dataset]).float())
    
    def get_lt_relation(self,data):
        #n = int(data['n'])
        n = int(round((int(data['n']))**(1/2.0)))
        n2 = int(n**2)
        n3 = int(n**3)
        lt_relation = np.zeros((n3, n3))
        for l,r in data['lt_edges']:
            l_row, l_col = int(l/n), int(l % n) 
            r_row, r_col = int(r/n), int(r % n)
            l_cells_ind_start = l_row*n2 + l_col*n
            r_cells_ind_start = r_row*n2 + l_col*n
            lt_relation[l_cells_ind_start:(l_cells_ind_start+n)][:,r_cells_ind_start:(r_cells_ind_start+n)] = 1.0
        #
        lt_relation = np.expand_dims(lt_relation,axis=-1)
        return lt_relation.astype(int)

    def get_class_lt_relation(self,data):
        #n = int(data['n'])
        n = int(round((float(data['n']))**(1/2.0)))
        n2 = int(n**2)
        n3 = int(n**3)
        upper_triang = np.triu(np.ones((n,n)),k=1)
        #Pdb().set_trace()
        lt_relation = np.zeros((n3, n3))
        for row in range(n):
            for col in range(n):
                start_ind = row*n2 + col*n
                end_ind = start_ind + n
                lt_relation[start_ind:end_ind][:,start_ind: end_ind] = upper_triang
        #Pdb().set_trace()
        lt_relation = np.expand_dims(lt_relation,axis=-1)
        return lt_relation
      
    def get_same_grid_relation(self,data):
        n = int(round((float(data['n']))**(1/2.0)))
        n2 = int(n**2)
        n3 = int(n**3)
        a,b = utils.block_shape_dict[n]         
        grid_relation = np.zeros((n3, n3))
        for x0 in range(0,n,a):
            for y0 in range(0,n,b):
                #print(x0,y0)
                for z in range(n):
                    x_range = np.arange(x0,x0+a)
                    y_range = np.arange(y0, y0+b)
                    block = (n2*np.expand_dims(x_range,1) + n*np.expand_dims(y_range,0) + z).flatten()
                    rows,cols = zip(*list(itertools.product(block,block)))
                    grid_relation[rows,cols] = 1
                    #grid_relation[block][:,block] = 1
        
        grid_relation[np.arange(n3),np.arange(n3)] = 0
        #
        return grid_relation

    def get_gcp_relations(self,data):
        #num nodes = data['num_nodes'] * data['chromatic_num']
        num_bin_nodes = data['num_nodes']*data['chromatic_num']
        k= data['chromatic_num']
        #Pdb().set_trace()
        relations = np.zeros((num_bin_nodes, num_bin_nodes, 2))
        num_nodes = data['num_nodes']
        for this_node in range(num_nodes):
            this_node_indices = np.arange(this_node*k, this_node*k + k)
            rows,cols = zip(*list(itertools.product(this_node_indices,this_node_indices)))
            relations[rows,cols,0] = 1
            #relations[this_node_indices][:,this_node_indices,0] = 1
        #        
        for l,r in data['edges'][:data['num_edges']]:
            l_cells_ind_start = l*k
            #Pdb().set_trace()
            r_cells_ind_start = r*k
            rows = np.arange(l_cells_ind_start, l_cells_ind_start + k)
            cols = np.arange(r_cells_ind_start, r_cells_ind_start + k)
            relations[rows,cols,1] = 1
            relations[cols,rows,1] = 1
            #relations[l_cells_ind_start:(l_cells_ind_start+k)][:,r_cells_ind_start:(r_cells_ind_start+k),1] = 1.0
            #relations[r_cells_ind_start:(r_cells_ind_start+k)][:,l_cells_ind_start:(l_cells_ind_start+k),1] = 1.0

        #
        np.fill_diagonal(relations[:,:,0],0)
        np.fill_diagonal(relations[:,:,1],0)
        return relations.astype(int)


    def get_relations(self,data, task):
        if task == 'gcp':
            return self.get_gcp_relations(data)

        n = int(round((float(data['n']))**(1/2.0)))
        n2 = int(n**2)
        n3 = int(n**3)
        relations = np.zeros((n3, n3, 2))

        for x in range(n3):
            row = int(x/n2)
            col = int((x % n2)/n)
            num = int(x % n2) % n

            for y in range(n):
                # cell constraints
                relations[x][row*n2+col*n+y][0] = 1

                # column constraints
                relations[x][y*n2+col*n+num][1] = 1

                # row constraints
                relations[x][row*n2+y*n+num][1] = 1
            #
            relations[x][x][0], relations[x][x][1] = 0,0
        
        #
        if task == 'futoshiki':
            class_lt_relation = self.get_class_lt_relation(data)
            return np.concatenate([relations, class_lt_relation], axis=-1).astype(int)
        
        if task == 'sudoku':
            grid_relation = self.get_same_grid_relation(data)
            relations[:,:,1] = ((relations[:,:,1] + grid_relation) > 0).astype(float)
            return relations.astype(int)

    def pad_set(self, target_set):
        pad_counter = self.max_count - len(target_set)
        if pad_counter < 0:
            return target_set[:self.max_count]

        return_set = list(target_set)
        return_set.extend([target_set[-1] for _ in range(pad_counter)])
        return np.array(return_set)
    
    def __getitem__(self, item):
        # Pdb().set_trace()
        if self.mode==TRAIN and (not self.args.controlled_batching):
            ind = self.sampler.sample().item()
        else:
            ind = item%len(self.dataset)


        data = self.dataset[ind]

        data["target"] = data["target_set"][0]
        data['is_ambiguous'] = data['count'] > 1
        data["target_set"] = self.pad_set(data["target_set"])
        if 'lt_edges' in data:
            data['lt_edges'] = data['lt_edges'][:data['num_lt_edges']] 
        data['n'] = data['query'].shape[0]
        data['qid'] = np.array([ind])
        if 'mask' not in data:
            data['mask'] = np.array([1 for _ in range(data['count'])] + [0 for _ in range(data['target_set'].shape[0] - data['count'])]) 
        # 
        if 'nlm_relations' not in data:
            #Pdb().set_trace()
            rel = self.get_relations(data, self.task)
            if self.task == 'futoshiki': 
                lt_rel = self.get_lt_relation(data)
                rel = np.concatenate([rel,lt_rel], axis=-1)
                
            data["nlm_relations"] = rel
            data['nlm_query'] = self.convert_to_unary_predicates(data['query'], data['num_nodes'], data['chromatic_num']) 
            data['nlm_target'] = self.convert_to_unary_predicates(data['target'], data['num_nodes'], data['chromatic_num'])[:,0]
            target_set_list = []
            for this_target in data['target_set']:
                target_set_list.append(self.convert_to_unary_predicates(this_target,data['num_nodes'],data['chromatic_num'])[:,0])
            #
            #Pdb().set_trace()
            data['nlm_target_set'] = np.stack(target_set_list, axis=0)
            #if self.task == 'sudoku':
            #    data['num_nodes'] = data['n'] 
        return data

    def convert_to_unary_predicates(self,query,n,k):
        #n = number of nodes
        #k = number of classes
        predicate = np.zeros((n*k,2))
        #Pdb().set_trace()
        nz_indices = query.nonzero()[0]
        for posn_ind in nz_indices:
            value = query[posn_ind] - 1 # -1 because values in the query start from 1 and not 0.
            ind = posn_ind*k + value
            start_ind = posn_ind*k
            predicate[ind,0] = 1
            predicate[start_ind:(start_ind+k),1] = 1
        return predicate.astype(int)
    
    """
    def convert_to_unary_predicates(self,query,n):
        #n = number of nodes
        #k = number of classes
        n = int(round((float(query.shape[0]))**(1/2.0)))
        n2 = int(n**2)
        n3 = int(n**3)
        predicate = np.zeros((n3,2))
        #Pdb().set_trace()
        nz_indices = query.nonzero()[0]
        for posn_ind in nz_indices:
            row = int(posn_ind / n)
            col = posn_ind % n
            value = query[posn_ind] - 1 # -1 because values in the query start from 1 and not 0.
            ind = row*n2 + col*n  + value 
            start_ind = row*n2 + col*n
            predicate[ind,0] = 1
            predicate[start_ind: (start_ind+n),1] = 1
        return predicate.astype(int)
    """

    def sample_epoch_indices(self):
        if self.mode != TRAIN:
            indices = list(np.arange(len(self.dataset))) 
            lengths = [x['num_bin_nodes'] for x in self.dataset]
        else:
            indices = list(self.sampler.sample(sample_shape = ((self._epoch_size,))).numpy())
            lengths = [self.dataset[ind]['num_bin_nodes'] for ind in indices]
        
        return indices, lengths

    def __len__(self):
        if self.mode == TRAIN:
            return self._epoch_size
        else:
            return len(self.dataset)


