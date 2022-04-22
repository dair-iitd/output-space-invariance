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

class ValueRRNLayer(nn.Module):
    def __init__(self, msg_layers, node_update_func,node_name = 'cluster', args = None):
        super(ValueRRNLayer, self).__init__()
        self.args = args
        self.msg_layers = msg_layers
        self.node_update_func = node_update_func
        self.node_name = node_name 
        #dropout
        self.drop_edges = nn.Dropout(args.edge_dropout)
        
        self.etype2msg_fn = {'diff': self.get_msg_diff,
                                'lt': self.get_msg_lt,
                                'gt': self.get_msg_gt
                               }

        self.reduce_func_mean = fn.mean('m','msg')
        self.message_func_same = fn.copy_edge('m','m')
        self.etype_dict = {}
        self.etype_dict.update(dict.fromkeys(['diff','lt','gt'], (self.message_func_same, self.reduce_func_mean)))
        keys = ['diff','lt','gt']
        for key in keys:
            if key not in self.msg_layers:
                del self.etype_dict[key]
                del self.etype2msg_fn[key]
        #
        logger.info("INitialized Value RRCNGAT Layer...fingers crossed Eyes closed!") 

    def get_msg(self,edges,etype):
        e = torch.cat([edges.src['h'], edges.dst['h']], -1)
        e = self.msg_layers[etype](e)
        e = self.drop_edges(e)
        return {'m': e}

    def get_msg_lt(self,edges):
        return self.get_msg(edges,'lt')
    
    def get_msg_gt(self,edges):
        return self.get_msg(edges,'gt')
    
    def get_msg_diff(self,edges):
        return self.get_msg(edges,'diff')
   
    def forward(self, g,step_no):
        #Pdb().set_trace()
        self.step_no = step_no
        t0 = time.time()
        for etype in self.msg_layers.keys():
            g.apply_edges(self.etype2msg_fn[etype],etype=etype)
        
        g.multi_update_all(etype_dict = self.etype_dict,
                            cross_reducer  = 'stack')
        rrcn_time = time.time() - t0
        
        g.nodes[self.node_name].data['m'] = torch.cat([g.nodes[self.node_name].data['msg'][:,dim_no,:] for dim_no in range(g.nodes[self.node_name].data['msg'].size(1))],dim=-1)
        g.apply_nodes(self.node_update,ntype=self.node_name)
        return rrcn_time

    def node_update(self, nodes):
        return self.node_update_func(nodes)


class RRCNGatLayer(nn.Module):
    def __init__(self, msg_layers, cell_node_update_func, args = None):
        super(RRCNGatLayer, self).__init__()
        self.args = args
        self.msg_layers = msg_layers
        self.cell_node_update_func = cell_node_update_func
        #dropout
        self.drop_edges = nn.Dropout(args.edge_dropout)

        #GAT Parameters
        self.cell_fc = nn.Linear(self.args.hidden_dim, self.args.attn_dim, bias = False)
        self.cluster_fc = nn.Linear(self.args.hidden_dim, self.args.attn_dim, bias = False)
        self.cluster2cell_attn_fc = nn.Linear(2*self.args.attn_dim,1,bias=False)
       
        if (len(self.args.attention_edges) > 0):
            #GAT Parameters for diff edges.
            logger.info("RRCN: Attention on diff edges as well:")
            self.cell_diff_fc = nn.Linear(self.args.hidden_dim, self.args.attn_dim, bias = False)
            self.cell2cell_diff_attn_fc = nn.Linear(2*self.args.attn_dim,1,bias=False)
        # 
        self.reset_parameters() 
        
        self.etype2msg_fn = {'diff': self.get_msg_diff, 
                                 'lt': self.get_msg_lt, 'gt': self.get_msg_gt,
                                     'contains': self.get_msg_contains,'may_contain': self.get_msg_may_contain                            }

        self.reduce_func_mean = fn.mean('m','msg')
        self.message_func_same = fn.copy_edge('m','m')
        self.etype_dict = {}
        self.etype_dict.update(dict.fromkeys(['diff','lt','gt','contains'], (self.message_func_same, self.reduce_func_mean)))
        self.etype_dict.update(dict.fromkeys(['may_contain'], (self.message_func_across, self.reduce_func_attn)))
        if (len(self.args.attention_edges) > 0):
            self.etype_dict.update(dict.fromkeys(['diff'], (self.message_func_across, self.reduce_func_attn)))
        #
        keys = ['diff','lt','gt','contains','may_contain']
        for key in keys:
            if key not in self.msg_layers:
                del self.etype_dict[key]
                del self.etype2msg_fn[key]
        #

        logger.info("INitialized RRCNGAT Layer...fingers crossed Eyes closed!") 

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.cluster_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.cell_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.cluster2cell_attn_fc.weight, gain=gain)
        #
        if (len(self.args.attention_edges) > 0):
            nn.init.xavier_normal_(self.cell_diff_fc.weight, gain=gain)
            nn.init.xavier_normal_(self.cell2cell_diff_attn_fc.weight, gain=gain)

    def cluster2cell_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.cluster2cell_attn_fc(z2)
        return {'as': F.leaky_relu(a)}

    def cell2cell_diff_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z1'], edges.dst['z1']], dim=1)
        a = self.cell2cell_diff_attn_fc(z2)
        return {'as': F.leaky_relu(a)}

    def message_func_across(self, edges):
        # message UDF for equation (3) & (4)
        return { 'm': edges.data['m'], 'as': edges.data['as']}
   
    def reduce_func_attn(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        #Pdb().set_trace()
        alpha = F.softmax(nodes.mailbox['as'], dim=1)
        # equation (4)
        m = torch.sum(alpha * nodes.mailbox['m'], dim=1)
        return {'msg': m}

    def get_msg(self,edges,etype):
        e = torch.cat([edges.src['h'], edges.dst['h']], -1)
        e = self.msg_layers[etype](e)
        e = self.drop_edges(e)
        return {'m': e}

    def get_msg_lt(self,edges):
        return self.get_msg(edges,'lt')
    
    def get_msg_gt(self,edges):
        return self.get_msg(edges,'gt')

    def get_msg_diff(self,edges):
        return self.get_msg(edges,'diff')
        
    def get_msg_contains(self,edges):
        return self.get_msg(edges, 'contains')

    
    def get_msg_may_contain(self,edges):
        return self.get_msg(edges, 'may_contain')

    
    def forward(self, g,step_no):
        #Pdb().set_trace()
        self.step_no = step_no
        t0 = time.time()
        for etype in self.msg_layers.keys():
            g.apply_edges(self.etype2msg_fn[etype],etype=etype)

        #go to attn space for msgs along diff
        g.nodes['cell'].data['z'] = self.cell_fc(g.nodes['cell'].data['h'])
        g.nodes['cluster'].data['z'] = self.cluster_fc(g.nodes['cluster'].data['h'])

        #if attn across diff edges, then go to attn space
        if len(self.args.attention_edges) > 0:
            g.nodes['cell'].data['z1'] = self.cell_diff_fc(g.nodes['cell'].data['h'])
            g.apply_edges(self.cell2cell_diff_attention,etype='diff')

        g.apply_edges(self.cluster2cell_attention,etype='may_contain')
        g.multi_update_all(etype_dict = self.etype_dict,
                            cross_reducer  = 'stack')
        rrcn_time = time.time() - t0
        
        g.nodes['cell'].data['m'] = torch.cat([g.nodes['cell'].data['msg'][:,dim_no,:] for dim_no in range(g.nodes['cell'].data['msg'].size(1))],dim=-1)
        g.apply_nodes(self.cell_node_update,ntype='cell')
    
        return rrcn_time

    def cell_node_update(self, nodes):
        return self.cell_node_update_func(nodes)



class ValueRRN(nn.Module):
    def __init__(self,
                msg_layers,
                node_update_func,
                num_steps,
                node_name = 'cluster',
                args=None):
        super(ValueRRN,self).__init__()
        self.args = args
        self.num_steps = num_steps
        self.node_name = node_name
        self.rrcn_layer = ValueRRNLayer(msg_layers,node_update_func,node_name, args)
      
    def forward(self,g,get_all_outputs):
        rrcn_time = 0
        cluster_outputs = []
        if get_all_outputs:
            cluster_outputs.append(g.nodes[self.node_name].data['h'])
        for step_no in range(self.num_steps):
            rrcn_time+=self.rrcn_layer(g,step_no)
            if get_all_outputs:
                cluster_outputs.append(g.nodes[self.node_name].data['h'])
        
        if get_all_outputs:
#             Pdb().set_trace()
            cluster_outputs = torch.stack(cluster_outputs, 0)  # num_steps x n_nodes x h_dim
        else:
            cluster_outputs =  g.nodes[self.node_name].data['h'].unsqueeze(0) 
        
        return cluster_outputs, rrcn_time 


class RRCN(nn.Module):
    def __init__(self,
                 msg_layers,
                 cell_node_update_func,
                 num_steps,
                 args = None):
        super(RRCN, self).__init__()
        self.args = args
        self.num_steps = num_steps
        self.rrcn_layer = RRCNGatLayer(msg_layers, cell_node_update_func, args)

    def forward(self, g, get_all_outputs=True):
        cell_outputs = []
        cluster_outputs = []
        
        rrcn_time = 0
        
        step_no, prev_pred, patience = 0, None, 0
        while True:
#         for step_no in range(self.num_steps):
            rrcn_time+=self.rrcn_layer(g,step_no)
            if get_all_outputs:
                cell_outputs.append(g.nodes['cell'].data['h'])
                cluster_outputs.append(g.nodes['cluster'].data['h'])
             
            step_no+=1
            if step_no==self.num_steps:
                break
                
        
        if get_all_outputs:
            cell_outputs = torch.stack(cell_outputs, 0)  # num_steps x n_nodes x h_dim
            cluster_outputs = torch.stack(cluster_outputs, 0)  # num_steps x n_nodes x h_dim
        else:
            cell_outputs = g.nodes['cell'].data['h'].unsqueeze(0) # 1 x n_nodes  x h_dim
            cluster_outputs =  g.nodes['cluster'].data['h'].unsqueeze(0) 
            #outputs = g.ndata['h']  # n_nodes x h_dim
        return cell_outputs, cluster_outputs, rrcn_time 



