import math
import pickle
from IPython.core.debugger import Pdb
import copy
import collections
import functools
import os
import json
from collections import Counter 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import random as py_random
import jacinle.random as random
import jacinle.io as io
import jactorch.nn as jacnn

from jactorch.utils.meta import as_tensor, as_float, as_cpu
from difflogic.cli import format_args
from difflogic.thutils_rl import binary_accuracy, instance_accuracy

from jacinle.logging import get_logger, set_output_file
from jacinle.utils.container import GView
from jacinle.utils.meter import GroupMeters
from jactorch.data.dataloader import JacDataLoader
from jactorch.utils.meta import as_cuda, as_numpy, as_tensor
import time
import datetime

import dgl
from .rrn.futoshiki_cluster_gg import FutoshikiGGCNN 
import utils

logger = get_logger(__file__)

class FutoshikiRRCN(nn.Module):
    def __init__(self, args):
        super(FutoshikiRRCN, self).__init__()
        self.args = args
        self.num_steps = args.msg_passing_steps
        self.futo_solver = FutoshikiGGCNN(args)

        self.wt_base_loss = torch.nn.CrossEntropyLoss(reduction='none')
        
        def loss_aggregator(pred, target, target_mask):
            # if pred and target have same dimension then simply compute loss
            # pred.shape == BS x 10 x 81 x  32
            #target.shape = BS X Target size x 81
            #if pred.dim()==self.args.sudoku_num_steps*target.dim():
            #    return self.base_loss(pred, target)
            #Pdb().set_trace() 
            # if pred and target are not of same dimension then compute loss wrt each element in target set
            # return a (batchsize x targetset size) vector 
            #target.shape = batch_size x target size x num_variables
            #pred.shape: batch_size x num variables
            #Pdb().set_trace()
            batch_size, target_size, num_variables = target.size()
            num_steps = pred.size(-1)
            pred = pred.unsqueeze(-1).expand(*pred.size(),target_size)
            target= target.transpose(1,2).unsqueeze(-1).expand(batch_size, num_variables, target_size,num_steps).transpose(-1,-2) 
            #target= torch.stack([target.transpose(1,2)]*num_steps,dim=-1).transpose(-1,-2) 
            loss_tensor = self.wt_base_loss(pred, target.long())
             
            loss_tensor = loss_tensor.mean(dim=list(range(1,loss_tensor.dim()-1)))*target_mask.float()
            
            #shape = batch_size x target_size
            if self.args.min_loss:
                #return has shape: batch_size 
                #loss_tensor  = loss_tensor.masked_fill((1-target_mask.byte()),float('inf')).min(dim=1)[0]
                loss_tensor  = loss_tensor.masked_fill((target_mask<1),float('inf')).min(dim=1)[0]
            #
            return loss_tensor
        self.loss_func = nn.CrossEntropyLoss()
        self.loss = loss_aggregator

        #used in the case of solution multiplicity: when the targetset is incomplete and the forward pass may find a valid solution.
        #In that case, we add the found solution to the targetset
        #This is set externally in the trainer/train.py
        self.add_to_targetset = False


    def forward(self, feed_dict, return_loss_matrix = False, can_break = False,get_all_steps=False):
        feed_dict = GView(feed_dict)
        #print(feed_dict['qid'].squeeze())
        batch_size = feed_dict['query'].size(0)
        bg = feed_dict['bg']
        board_size = int(math.sqrt(bg.nodes('cell').shape[0]//batch_size))

        #logits : of shape : args.sudoku_num_steps x batchsize*(max_num_nodes) x (1+max_num_colours) if training
        #logits: of shape : batch_size* max_num_nodes x max_num_colours if not training
        ret_dict, logits, rrcn_time, embedding_ortho_loss  = self.futo_solver(feed_dict, batch_size, self.training, get_all_steps = get_all_steps)
        
        time_solve = rrcn_time
        t0 = time.time()

        if self.training or get_all_steps:
            logits = logits.transpose(1,2)
            logits = logits.transpose(0,2)
        else:
            logits = logits.unsqueeze(-1)
        
        #shape of logits now : BS*81 x 10 x 32 if self.training ,  otherwise BS*81 x 10 x 1
        logits = logits.view(-1, board_size*board_size, logits.size(-2), logits.size(-1)) 
        #shape of logits now : BS x  81 x 10 x 32(1)
        logits = logits.transpose(1,2)
        #shape of logits now : BS x  10 x 81 x 32(1)
        #pred = logits[:,:,:,-1].argmax(dim=1)
        pred = logits 

        if self.training or self.add_to_targetset:
            this_meters, ia_output_dict = instance_accuracy(feed_dict.target.float(), pred, return_float= False, feed_dict=feed_dict, task = self.args.task, args=self.args)
           
            #dont worry about the reward. It is not used here.
            reward,new_targets = ia_output_dict['reward'], ia_output_dict['new_targets'] 
            
            if self.add_to_targetset:
                utils.add_missing_target(feed_dict,new_targets,reward)

            monitors = dict()
            target = feed_dict.target.float()
            count = None
            loss_matrix = None
            if self.args.min_loss or 'weights' in feed_dict or return_loss_matrix:
                loss_matrix = self.loss(logits, feed_dict.target_set,feed_dict.mask)
            else:
                loss_matrix = self.loss(logits, target.unsqueeze(1),feed_dict.mask[:,0].unsqueeze(-1))
            
            if 'weights' in feed_dict:
                loss = (feed_dict.weights*loss_matrix).sum()/feed_dict.weights.sum()
            else:
                loss = loss_matrix.mean()
            
            embedding_ortho_loss = self.args.embedding_ortho_loss_factor*embedding_ortho_loss

            if torch.is_tensor(embedding_ortho_loss):
                loss += embedding_ortho_loss.squeeze()
                monitors['ortho_loss'] = embedding_ortho_loss.squeeze()
                
            t1 = time.time()
            time_remaining = t1 - t0
            monitors['time_solve'], monitors['time_remaining'] =  torch.tensor([time_solve]), torch.tensor([time_remaining])
            monitors.update(this_meters)
            
            ret_dict.update(dict(pred=pred,reward=reward,loss_matrix = loss_matrix))
            return loss, monitors, ret_dict
        else:
            ret_dict.update(dict(pred=pred))
            return ret_dict
 
