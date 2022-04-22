from torch import nn
import torch
from IPython.core.debugger import Pdb 
import pickle
import os
import torch.nn.functional as F
import math
import time
from .layer_norm_lstm import LayerNormLSTMCell
from .rrcn import ValueRRN 
from jacinle.logging import get_logger, set_output_file
from . import rrn_utils
import copy

logger = get_logger(__file__)

def get_class_embedding_generator(cls,kwargs=None):
    logger.info("Create GNN for initial embeddings")
    return GNNEmbeddingGenerator(cls,kwargs)
    
class FixedPermutation(nn.Module):
    def __init__(self,args):
        super(FixedPermutation,self).__init__()
        self.args = args
        self.fixed_embeddings = nn.Parameter(torch.randn(1+self.args.max_num_embeddings,self.args.hidden_dim))
    
    def forward(self,batch_size, board_size, is_training):
        digit_embeddings = self.fixed_embeddings 
        digit_embeddings, ortho_loss, self.idx = rrn_utils.get_permutation(digit_embeddings, batch_size, board_size, should_permute_gen_embed = is_training) 
        
        return digit_embeddings, ortho_loss 
 
class GNNEmbeddingGenerator(nn.Module):
    def __init__(self,cls,kwargs):
        super(GNNEmbeddingGenerator,self).__init__()

        self.args = cls.args
        self.init_generator = FixedPermutation(self.args)
        edge_type_list = kwargs['edge_types']
        
        self.msg_layers = nn.ModuleDict([(k,cls.msg_layers[k]) for k in edge_type_list])
        #Pdb().set_trace()
        if not cls.args.share_mp_wts_in_gen:
            logger.info("Clone the msg passing wts in generator")
            for k in self.msg_layers:
                self.msg_layers[k] = copy.deepcopy(self.msg_layers[k])
        elif not cls.args.share_diff_edges_in_gen:
            if 'diff' in self.msg_layers:
                self.msg_layers['diff'] = copy.deepcopy(self.msg_layers['diff'])


        lstm_cell = nn.LSTMCell
        if self.args.layer_norm:
            logger.info("Using layer normalization in lstm")
            lstm_cell = LayerNormLSTMCell 
        hidden_dim = self.args.hidden_dim
        lstm_ip_dim = (1 + len(edge_type_list))*hidden_dim

        self.node_lstm = lstm_cell(lstm_ip_dim, hidden_dim, bias=False)

        node_update_func = self.node_update_func
        self.class_rrcn = ValueRRN(  
                self.msg_layers,
                node_update_func,
                self.args.class_rrn_num_steps,
                node_name = 'cluster',
                args=self.args)
        
        assert self.args.output_embedding_generator in ['linear','eye']
        self.input2output_embed_transformation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if self.args.output_embedding_generator  == 'eye':
            self.input2output_embed_transformation.weight.data = torch.eye(hidden_dim, hidden_dim)
            self.input2output_embed_transformation.weight.requires_grad = False

    def node_update_func(self, nodes):
        #Pdb().set_trace()
        x, h, m, c = nodes.data['x'], nodes.data['rnn_h'], nodes.data['m'],nodes.data['rnn_c']
        new_h, new_c = self.node_lstm(torch.cat([x, m], -1), (h, c))
        return {'h': new_h, 'rnn_c': new_c, 'rnn_h': new_h}


    def forward(self, feed_dict, board_size, batch_size,  is_training = True, num_classes_list=None):
        lstm_embeddings, orthogonality_loss = self.init_generator(batch_size, board_size, is_training)
         
        g = feed_dict['bcg']
        lookup_at = g.nodes['cluster'].data['id'].long() + (board_size + 1)*g.nodes['cluster'].data['sno']
        cluster_x = F.embedding(lookup_at,lstm_embeddings)
         
        g.nodes['cluster'].data['x'] = cluster_x
        g.nodes['cluster'].data['h'] = cluster_x
        g.nodes['cluster'].data['rnn_h'] = cluster_x 
        g.nodes['cluster'].data['rnn_c'] = cluster_x 
        
        input_embeddings, rrcn_time = self.class_rrcn(g,True)
        step_input_embeddings = input_embeddings.clone().detach()
        hidden_dim = input_embeddings.shape[-1]
        if num_classes_list is not None:
            cluster_outputs = rrn_utils.get_padded_tensor(input_embeddings,num_classes_list +1)       
        else:
            cluster_outputs = input_embeddings.view(input_embeddings.shape[0], batch_size,board_size+1, hidden_dim)

        #cluster shape: #steps x BS x h x 10
        #mask_of_num_colours.shape: #BS x (max_num_colours + 1)
        #Pdb().set_trace()
        #cluster_outputs = cluster_outputs.transpose(2,3)
        orthogonality_loss += rrn_utils.get_avg_dot_product(cluster_outputs[:,:,1:,:].transpose(-1,-2)).mean()
        #Pdb().set_trace()
        #input_embeddings = input_embeddings[-1]
        cluster_outputs = cluster_outputs[-1,:,1:,:]
        #cluster_outputs.shape = BS x num_classes x hidden_dim
        output_embeddings = self.input2output_embed_transformation(cluster_outputs)
        orthogonality_loss += rrn_utils.get_avg_dot_product(output_embeddings.transpose(-1,-2)).mean()

        lstm_embeddings = lstm_embeddings.view(batch_size, (board_size+1), hidden_dim)
        input_embeddings = torch.cat([lstm_embeddings[:,0:1,:], cluster_outputs], dim = 1).view(-1,hidden_dim)
        output_embeddings = torch.cat([lstm_embeddings[:,0:1,:], output_embeddings], dim = 1).view(-1,hidden_dim) 
        lstm_embeddings = lstm_embeddings.view(-1, hidden_dim)

        return {'gol': orthogonality_loss,
                'lstm_embeddings': lstm_embeddings, 
                'input_embeddings': input_embeddings.squeeze(), 
                'output_embeddings': output_embeddings.squeeze(),
                'step_input_embeddings': step_input_embeddings,
                'lstm_idx': self.init_generator.idx,
               }
