
from .rrcn_bin import RRCN
from torch import nn
import torch
from IPython.core.debugger import Pdb 
import pickle
import os
import torch.nn.functional as F
import math
import time
from .layer_norm_lstm import LayerNormLSTMCell
from jacinle.logging import get_logger, set_output_file

logger = get_logger(__file__)

def get_padded_tensor(cell_outputs, num_nodes_list, num_colours_list, args):
#     Pdb().set_trace()
    
    max_num_nodes = num_nodes_list.max()
    max_num_colours = num_colours_list.max()
    
    if args.logk:
        max_levels = math.floor(math.log2(max_num_colours)) + 1
    else:
        max_levels = max_num_colours
    
    cell_outputs_list = []
    start_at = 0
    end_at = 0
    for i,(this_num_nodes, this_num_colours) in enumerate(zip(num_nodes_list, num_colours_list)):
        this_levels = max_levels
        end_at += this_num_nodes.item()*this_levels.item()

#         Pdb().set_trace()
        unpad_cell = cell_outputs[:,start_at:end_at,:].reshape(cell_outputs.shape[0], this_levels.item(), this_num_nodes.item(), cell_outputs.shape[-1])
        pad_cell = F.pad(unpad_cell,(0,0,0,max_num_nodes - this_num_nodes,0,max_levels - this_levels))

        cell_outputs_list.append(pad_cell)
        start_at = end_at
    return torch.stack(cell_outputs_list,dim=1)


class GCP_BIN_CNN(nn.Module):
    def __init__(self,
            args):
        super(GCP_BIN_CNN, self).__init__()
        self.args = args
        self.num_steps = args.msg_passing_steps        
        

        hidden_dim = self.args.hidden_dim 
        
        self.digit_embed = nn.Embedding(3,hidden_dim)
       
        #2 msg passing layers: one for each of the edge types
        self.msg_layers = nn.ModuleDict()
        
        #intra_diff: edges in the input graph puzzle
        #inter_diff: edges added between binary nodes that correspond to the same variable node in the input puzzle
        edge_types = ['intra_diff', 'inter_diff']

        for edge_type in edge_types:
            self.msg_layers[edge_type] = nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),
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
            logger.info("GCP: Using layer normalization in lstm")
            lstm_cell = LayerNormLSTMCell 

        self.cell_lstm = lstm_cell(cell_lstm_ip_size, hidden_dim, bias=False)
        cell_upd_fxn = self.cell_update_func_lstm

        
        self.embedding2score = nn.Sequential(
                #nn.Linear(hidden,hidden_dim, bias = 0),
                nn.Linear(hidden_dim, 1, bias = 0)
                )
    
        kwargs = {'score_function': self.embedding2score}
        self.rrcn = RRCN(self.msg_layers, cell_upd_fxn, self.num_steps, self.args, kwargs)
  
    def forward(self, g, batch_size, num_colours_list, num_nodes_list = None, is_training=True, get_all_steps=False):
        #num_colours should correspond to max chromatic number 
        #labels = g.ndata.pop('a')
        #print('Base: ',self.digit_embed.weight.data[2,:4], self.row_embed.weight.data[2,:4])
        #input_digits = self.digit_embed(g.ndata['q'])
        if isinstance(num_colours_list,int):
            num_colours_list = torch.zeros(batch_size,device=g.nodes['cluster'].data['id'].device).long() + num_colours_list
        
        if num_nodes_list is None:
            num_nodes_list = torch.zeros(batch_size, device = g.nodes['cluster'].data['id'].device).long() + g.number_of_nodes('cell')//batch_size

        num_colours = num_colours_list.max()
        mask_of_num_colours = torch.arange(num_colours+1, device = num_nodes_list.device)[None,:] <= num_colours_list[:,None]
        mask_of_num_nodes = torch.arange(num_nodes_list.max(), device = num_nodes_list.device)[None,:] < num_nodes_list[:,None] 

        max_num_nodes = num_nodes_list.max() 
        
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
        
        cell_outputs = get_padded_tensor(cell_outputs,num_nodes_list, num_colours_list, self.args)
        cell_save = cell_outputs.detach().clone()
        
        #logits = (cell_outputs@self.logit_mat).sum(dim=-1)
        #Pdb().set_trace()
        if torch.is_tensor(cell_scores):
            logits = cell_scores
        else:
            logits = self.embedding2score(cell_outputs).squeeze(-1) 
        #logits shape is: this_steps, batch_size, max_num_nodes, (num_colours+1)
        
        if self.args.logk:
            max_levels = math.floor(math.log2(num_colours.item())) + 1
        else:
            max_levels = num_colours.item()
        
        logits = logits.view(this_steps,batch_size*max_num_nodes*max_levels)

        ret_dict = {'cell_embed': cell_save,
                'cell_input': cell_input,
                'cell_x': cell_x
               }
        return ret_dict, logits, rrcn_time
        
    def cell_update_func_lstm(self, nodes):
        x, h, m, c = nodes.data['x'], nodes.data['rnn_h'], nodes.data['m'],nodes.data['rnn_c']
        new_h, new_c = self.cell_lstm(torch.cat([x, m], -1), (h, c))
        return {'h': new_h, 'rnn_c': new_c, 'rnn_h': new_h}
