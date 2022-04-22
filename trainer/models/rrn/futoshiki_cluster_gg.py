"""
Futoshiki  module based on RRN for solving futoshiki puzzles
"""


from .rrcn import RRCN
from torch import nn
import torch
from IPython.core.debugger import Pdb 
import pickle
import os
import torch.nn.functional as F
import math
import time
from .layer_norm_lstm import LayerNormLSTMCell
from . import embedding_generator as eg
from jacinle.logging import get_logger, set_output_file

logger = get_logger(__file__)

def calc_logits(output_embeddings, batch_size, board_size, cell_outputs, cluster_outputs):
        
    #cell_outputs = self.output_transformation(cell_outputs)
    #return cell_outputs 
    #shape of cell_outputs: num_steps x sum of num nodes x h
    #shape of cluster_outputs: num_steps x sum of num colour nodes x h
    this_steps, hidden_dim  = cell_outputs.size(0), cell_outputs.size(-1)

    cell_outputs = cell_outputs.view(this_steps, batch_size, board_size*board_size, hidden_dim)
    cluster_outputs = cluster_outputs.view(this_steps, batch_size,board_size+1, hidden_dim)

    #make them of the same dim, but first move cell and cluster dims to the end
    cell_outputs = cell_outputs.transpose(2,3)
    cluster_outputs  = cluster_outputs.transpose(2,3)

    #cell shape: #steps x  BS x h x max_nodes
    #cluster shape: #steps x BS x h x (max_colours+1)

    cluster_outputs = output_embeddings.view(
                    batch_size, (board_size+1),hidden_dim).transpose(-1,-2).unsqueeze(0).expand_as(cluster_outputs)

    cell_outputs = cell_outputs.unsqueeze(-1).expand(this_steps, batch_size, hidden_dim, board_size*board_size, (board_size + 1))
    cluster_outputs = cluster_outputs.unsqueeze(-2).expand_as(cell_outputs)
    logits = (cell_outputs*cluster_outputs).sum(dim=2)

    #logits shape is: this_steps, batch_size, max_num_nodes, (num_colours+1)
    logits = logits.view(this_steps,batch_size*board_size*board_size,(board_size + 1))
    
    return logits

    
class FutoshikiGGCNN(nn.Module):
    def __init__(self,
            args):
        super(FutoshikiGGCNN, self).__init__()
        self.args = args
        self.num_steps = args.msg_passing_steps
        hidden_dim = self.args.hidden_dim 
       
        self.msg_layers = nn.ModuleDict()
        edge_types = ['contains','lt','gt']
        edge_types_for_class_graph = ['lt','gt']
        if self.args.diff_edges_in_class_graph:
            edge_types_for_class_graph.append('diff')

        edge_types.append('may_contain')
        edge_types = edge_types + ['diff']
        

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
            logger.info("Futoshiki: Using layer normalization in lstm")
            lstm_cell = LayerNormLSTMCell 

        self.cell_lstm = lstm_cell(cell_lstm_ip_size, hidden_dim, bias=False)
        cell_upd_fxn = self.cell_update_func_lstm
        
        self.rrcn = RRCN(self.msg_layers, cell_upd_fxn, self.num_steps, self.args)
        self.embed_module = eg.get_class_embedding_generator(self,{'edge_types': edge_types_for_class_graph})
            
   
    
    def forward(self, feed_dict, batch_size, is_training=True, get_all_steps=False):
        #num_colours should correspond to max chromatic number 
        g = feed_dict['bg']
        board_size = int(math.sqrt(g.nodes('cell').shape[0]//batch_size))
        cluster_input_list = [] 
        embed_orthogonality_loss= 0
        embed_gen_output = self.embed_module(feed_dict, board_size, batch_size, is_training)
        input_embeddings = embed_gen_output['input_embeddings']
        output_embeddings = embed_gen_output['output_embeddings']
        embed_orthogonality_loss = embed_gen_output['gol']

        lookup_at =  g.nodes['cluster'].data['id'].long() + (board_size + 1)*g.nodes['cluster'].data['sno']
        cluster_input_list.append(F.embedding(lookup_at,input_embeddings))
        
        cluster_input = torch.cat(cluster_input_list, dim = -1)
        cluster_x = cluster_input
        #cell embeddings
        cell_embeddings_lookup = input_embeddings
        #empty cell init by avg
        cell_embeddings_lookup  = cell_embeddings_lookup.reshape(batch_size,(board_size+1),-1)
        avg_of_embeddings = cell_embeddings_lookup[:,1:,:].mean(dim=1).unsqueeze(1)
        cell_embeddings_lookup = torch.cat(
                        [avg_of_embeddings, cell_embeddings_lookup[:,1:,:]],
                        dim=1).reshape(batch_size*(board_size+1),-1)
        
        #now lookup noise at: g.nodes['cell'].data['q'] + (num_colours + 1)*g.nodes['cell'].data.pop('sno')
        lookup_at =  g.nodes['cell'].data['q'].long() + (board_size + 1)*g.nodes['cell'].data.pop('sno')
        cell_x = cell_embeddings_lookup[lookup_at]
        cell_input = cell_x
        g.nodes['cell'].data['x'] = cell_x
        g.nodes['cell'].data['h'] = cell_x
        g.nodes['cell'].data['rnn_h'] = torch.zeros_like(cell_x, dtype=torch.float)
        g.nodes['cell'].data['rnn_c'] = torch.zeros_like(cell_x, dtype=torch.float)
        
        #cluster_x is used to compute attention wts while recieving msgs along may_contain edges 
        g.nodes['cluster'].data['h'] = cluster_x
        
#         Pdb().set_trace()
        cell_outputs, cluster_outputs, rrcn_time = self.rrcn(g, is_training or get_all_steps)
        cluster_save = cluster_outputs.detach().clone()
        cell_save = cell_outputs.detach().clone()

        logits = calc_logits(output_embeddings, batch_size, board_size, cell_outputs, cluster_outputs)
        ret_dict = {'cluster_embed': cluster_save,
                'cell_embed': cell_save,
                'cluster_input': cluster_input,
                'cell_input': cell_input,
                'cluster_x': cluster_x,
                'cell_x': cell_x
               }
        ret_dict.update(embed_gen_output)
        return ret_dict, logits, rrcn_time, embed_orthogonality_loss
        

    def cell_update_func_lstm(self, nodes):
        x, h, m, c = nodes.data['x'], nodes.data['rnn_h'], nodes.data['m'],nodes.data['rnn_c']
        new_h, new_c = self.cell_lstm(torch.cat([x, m], -1), (h, c))
        return {'h': new_h, 'rnn_c': new_c, 'rnn_h': new_h}
