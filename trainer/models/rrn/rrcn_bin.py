"""
Recurrent Relational Network(RRCN) module

References:
- Recurrent Relational Networks
- Paper: https://arxiv.org/abs/1711.08028
- Original Code: https://github.com/rasmusbergpalm/recurrent-relational-networks
"""

import torch
from torch import nn
import dgl.function as fn
import torch.nn.functional as F
import time
import numpy as np

from IPython.core.debugger import Pdb

from jacinle.logging import get_logger, set_output_file
logger = get_logger(__file__)


class RRCNGatLayer(nn.Module):
    def __init__(self, msg_layers, cell_node_update_func, args = None):
        super(RRCNGatLayer, self).__init__()
        self.args = args
        self.msg_layers = msg_layers
        self.cell_node_update_func = cell_node_update_func
        
        #dropout
        self.drop_edges = nn.Dropout(args.edge_dropout)

        self.etype2msg_fn = {'intra_diff': self.get_msg_intra_diff,'inter_diff': self.get_msg_inter_diff, 'intra_lt': self.get_msg_intra_lt,'inter_lt': self.get_msg_inter_lt,'intra_gt': self.get_msg_intra_gt,'inter_gt': self.get_msg_inter_gt}

        self.reduce_func_mean = fn.mean('m','msg')
        self.reduce_func_sum = fn.sum('m','msg') 
        self.message_func_same = fn.copy_edge('m','m')
        self.etype_dict = {}
        self.etype_dict.update(dict.fromkeys(list(self.etype2msg_fn.keys()), (self.message_func_same, self.reduce_func_mean)))

        #attention
        if (len(self.args.attention_edges) > 0):
            self.node_attn_mlps = nn.ModuleDict()
            self.attention_fn = nn.ModuleDict()
            for edge in self.args.attention_edges:
                assert edge in self.msg_layers
                node_fc = nn.Linear(self.args.hidden_dim, self.args.attn_dim, bias = False)
                attn_fc = nn.Linear(2*self.args.attn_dim,1,bias=False)
                self.node_attn_mlps[edge] = node_fc
                self.attention_fn[edge] = attn_fc
                self.etype_dict[edge] = (getattr(self,'message_func_attn_{}'.format(edge)), getattr(self,'reduce_func_attn_{}'.format(edge)))
            self.reset_parameters()
        
        keys = list(self.etype2msg_fn.keys())
        for key in keys:
            if key not in self.msg_layers:
                del self.etype_dict[key]
                del self.etype2msg_fn[key]
        
        self.edge_features = None
        if self.args.edge_embeddings_dim > 0:
            self.edge_features = nn.ParameterDict()
            for k in self.msg_layers:
                self.edge_features[k] = nn.Parameter(torch.randn(self.args.edge_embeddings_dim))
    
        logger.info("INitialized binary RRCNGAT Layer") 
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for etype in self.node_attn_mlps:
            nn.init.xavier_normal_(self.node_attn_mlps[etype].weight, gain=gain)
            nn.init.xavier_normal_(self.attention_fn[etype].weight, gain=gain)
    
    def get_msg(self,edges,etype,ef):
        cat_list = [edges.src['h'], edges.dst['h']]
        if ef is not None:
            cat_list.append(ef.unsqueeze(0).expand(edges.src['h'].size(0),-1))
        e = torch.cat(cat_list, -1)
        e = self.msg_layers[etype](e)
        e = self.drop_edges(e)
        return {'m': e}
    
    def get_msg_inter_diff(self,edges):
        ef = None
        if self.edge_features is not None:
            ef = self.edge_features['inter_diff']
        return self.get_msg(edges, 'inter_diff',ef)

    def get_msg_intra_diff(self,edges):
        ef = None
        if self.edge_features is not None:
            ef = self.edge_features['intra_diff']
        return self.get_msg(edges, 'intra_diff',ef)
    
    def get_msg_inter_lt(self,edges):
        ef = None
        if self.edge_features is not None:
            ef = self.edge_features['inter_lt']
        return self.get_msg(edges, 'inter_lt',ef)

    def get_msg_intra_lt(self,edges):
        ef = None
        if self.edge_features is not None:
            ef = self.edge_features['intra_lt']
        return self.get_msg(edges, 'intra_lt',ef)
    
    def get_msg_inter_gt(self,edges):
        ef = None
        if self.edge_features is not None:
            ef = self.edge_features['inter_gt']
        return self.get_msg(edges, 'inter_gt',ef)

    def get_msg_intra_gt(self,edges):
        ef = None
        if self.edge_features is not None:
            ef = self.edge_features['intra_gt']
        return self.get_msg(edges, 'intra_gt',ef)
    
   
    def compute_attention_wts(self,edges,etype):
        z = torch.cat([edges.src[etype], edges.dst[etype]], dim=1)
        a = self.attention_fn[etype](z)
        return {'attn_score_{}'.format(etype): F.leaky_relu(a)}

    def message_func_attn(self, edges, etype):
        key = 'attn_score_{}'.format(etype)
        return { 'm': edges.data['m'], key : edges.data[key]}
   
    def reduce_func_attn(self, nodes, etype):
        key = 'attn_score_{}'.format(etype)
        alpha = F.softmax(nodes.mailbox[key], dim=1)
        m = torch.sum(alpha * nodes.mailbox['m'], dim=1)
        return {'msg': m}


    def message_func_attn_inter_diff(self, edges):
        return self.message_func_attn(edges, 'inter_diff')

    def reduce_func_attn_inter_diff(self, nodes):
        return self.reduce_func_attn(nodes, 'inter_diff')

    def message_func_attn_intra_diff(self, edges):
        return self.message_func_attn(edges, 'intra_diff')

    def reduce_func_attn_intra_diff(self, nodes):
        return self.reduce_func_attn(nodes, 'intra_diff')

    def forward(self, g,step_no):
        self.step_no = step_no
        t0 = time.time()
        
        for etype in self.msg_layers.keys():
            g.apply_edges(self.etype2msg_fn[etype],etype=etype)

        #attention:
        #Pdb().set_trace()
        for etype in self.args.attention_edges:
            g[etype].ndata[etype] = self.node_attn_mlps[etype](g[etype].ndata['h'])
            g.apply_edges(lambda x: self.compute_attention_wts(x,etype),etype=etype)
            #self.etype_dict[etype] = (lambda x: self.message_func_attn(x,etype), lambda x: self.reduce_func_attn(x,etype))

        g.multi_update_all(etype_dict = self.etype_dict, cross_reducer  = 'stack')
        
        rrcn_time = time.time() - t0
        
        g.nodes['cell'].data['m'] = torch.cat([g.nodes['cell'].data['msg'][:,dim_no,:] for dim_no in range(g.nodes['cell'].data['msg'].size(1))],dim=-1)
        g.apply_nodes(self.cell_node_update,ntype='cell')
        
        return rrcn_time

    def cell_node_update(self, nodes):
        return self.cell_node_update_func(nodes)

    
class RRCN(nn.Module):
    def __init__(self,
                 msg_layers,
                 cell_node_update_func,
                 num_steps,
                 args = None,
                 kwargs = {}):
        super(RRCN, self).__init__()
        self.args = args
        self.num_steps = num_steps
        self.rrcn_layer = RRCNGatLayer(msg_layers, cell_node_update_func, args)

    def forward(self, g, get_all_outputs=True):
        cell_outputs = []
        cell_scores = [] 
        rrcn_time = 0
        
        step_no, prev_pred, patience = 0, None, 0
        while True:
#         for step_no in range(self.num_steps):
            rrcn_time+=self.rrcn_layer(g,step_no)
            if get_all_outputs:
                cell_outputs.append(g.nodes['cell'].data['h'])
            # 
            step_no+=1
            if step_no==self.num_steps:
                break
        
        if get_all_outputs:
            cell_outputs = torch.stack(cell_outputs, 0)  # num_steps x n_nodes x h_dim
        else:
            cell_outputs = g.nodes['cell'].data['h'].unsqueeze(0) # 1 x n_nodes  x h_dim

        return cell_outputs, rrcn_time, cell_scores



