"""

okuNN module based on RRN for solving sudoku puzzles
"""

#import sys,os
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .rrcn_bin import RRCN
import torch
from torch import nn
from IPython.core.debugger import Pdb 
import pickle
import os
import torch.nn.functional as F
import math
import time
from .layer_norm_lstm import LayerNormLSTMCell
from jacinle.logging import get_logger, set_output_file

logger = get_logger(__file__)

def get_padded_tensor(cell_outputs, num_nodes_list, num_colours_list):
#     Pdb().set_trace()
    
    max_num_nodes = num_nodes_list.max()
    max_num_colours = num_colours_list.max()
    
    cell_outputs_list = []
    start_at = 0
    end_at = 0
    for i,(this_num_nodes, this_num_colours) in enumerate(zip(num_nodes_list, num_colours_list)):
        end_at += this_num_nodes.item()*this_num_colours.item()

        unpad_cell = cell_outputs[:,start_at:end_at,:].reshape(cell_outputs.shape[0], this_num_colours.item(), this_num_nodes.item(), cell_outputs.shape[-1])
        pad_cell = F.pad(unpad_cell,(0,0,0,max_num_nodes - this_num_nodes,0,max_num_colours - this_num_colours))

        cell_outputs_list.append(pad_cell)
        start_at = end_at
    return torch.stack(cell_outputs_list,dim=1)


# def get_padded_tensor(cell_outputs, num_nodes_list_temp, num_colours_list):
#     num_nodes_list = num_nodes_list_temp*num_colours_list.max()
# #     Pdb().set_trace()
#     max_num_nodes = num_nodes_list.max()
#     cell_outputs_list = []
#     start_at = 0
#     end_at = 0
#     for i,this_num_nodes in enumerate(num_nodes_list):
#         end_at += this_num_nodes.item()
#         cell_outputs_list.append(F.pad(cell_outputs[:,start_at:end_at,:],(0,0,0,max_num_nodes - this_num_nodes)))
#         start_at = end_at
#     return torch.stack(cell_outputs_list,dim=1)


class Futoshiki_BIN_CNN(nn.Module):
    def __init__(self,
            args):
        super(Futoshiki_BIN_CNN, self).__init__()
        self.args = args
        self.num_steps = args.msg_passing_steps
        

        hidden_dim = self.args.hidden_dim 
        self.digit_embed = nn.Embedding(3,hidden_dim)
        
        #2 msg passing layers: one for each of the edge types
        self.msg_layers = nn.ModuleDict()
        
        edge_types = ['intra_diff', 'inter_diff', 'intra_lt', 'intra_gt']
        if not args.share_lt_edges:
            edge_types += ['inter_lt', 'inter_gt']

        if self.args.share_all_msg_passing_mlp:
            logger.info("In BInary model. Sharing all msg passing mlps")
            mlp = nn.Sequential(
                nn.Linear(2*hidden_dim+self.args.edge_embeddings_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for edge_type in edge_types:
                self.msg_layers[edge_type] = mlp
        else:
            for edge_type in edge_types:
                self.msg_layers[edge_type] = nn.Sequential(
                    nn.Linear(2*hidden_dim+self.args.edge_embeddings_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
        
        cell_lstm_ip_size = (len(edge_types) + 1)*hidden_dim
        lstm_cell = nn.LSTMCell
        if self.args.layer_norm:
            logger.info("Futoshiki: Using layer normalization in lstm")
            lstm_cell = LayerNormLSTMCell 

        self.cell_lstm = lstm_cell(cell_lstm_ip_size, hidden_dim, bias=False)
        cell_upd_fxn = self.cell_update_func_lstm

        
        self.embedding2score = nn.Sequential(
                #nn.Linear(hidden,hidden_dim, bias = 0),
                nn.Linear(hidden_dim, 1, bias = 0)
                )
   
        kwargs = {'score_function': self.embedding2score}
        self.rrcn = RRCN(self.msg_layers, cell_upd_fxn, self.num_steps, self.args,kwargs)
   
   
    def forward(self, feed_dict, batch_size, is_training=True, get_all_steps=False):
        
        g = feed_dict['bg']
        board_size = round((g.nodes('cell').shape[0]//batch_size)**(1/3))
        
        #cell embeddings
        cell_input_list = []
        
        cell_idx = g.nodes['cell'].data['q'].long()
        cell_idx += 3*(cell_idx==-1).long()
        cell_input_list.append(self.digit_embed(cell_idx))
        cell_input = torch.cat(cell_input_list, dim = -1)
        
        cell_x = cell_input

        g.nodes['cell'].data['x'] = cell_x
        g.nodes['cell'].data['h'] = cell_x
        g.nodes['cell'].data['rnn_h'] = torch.zeros_like(cell_x, dtype=torch.float)
        g.nodes['cell'].data['rnn_c'] = torch.zeros_like(cell_x, dtype=torch.float)
            
        cell_outputs, rrcn_time,cell_scores = self.rrcn(g, is_training or get_all_steps)
            
        this_steps, hidden_dim = cell_outputs.size(0), cell_outputs.size(-1)
        cell_save = cell_outputs.detach().clone()
       
        if torch.is_tensor(cell_scores):
            logits = cell_scores
        else:
            logits = self.embedding2score(cell_outputs).squeeze(-1) 

        ret_dict = {'cell_embed': cell_save,
                'cell_input': cell_input,
                'cell_x': cell_x
               }
        return ret_dict, logits, rrcn_time
        
        
    def cell_update_func_mlp(self, nodes):
        
        x, h, m = nodes.data['x'], nodes.data['h'], nodes.data['m']
        new_h = self.cell_mlp_before_lstm(torch.cat([x,h,m], -1))
        
        return {'h': new_h}

    def cell_update_func_lstm(self, nodes):
        x, h, m, c = nodes.data['x'], nodes.data['rnn_h'], nodes.data['m'],nodes.data['rnn_c']
        new_h, new_c = self.cell_lstm(torch.cat([x, m], -1), (h, c))
        return {'h': new_h, 'rnn_c': new_c, 'rnn_h': new_h}
                                          
                                          
