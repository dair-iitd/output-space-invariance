import torch

import torch.nn.functional as F

def get_padded_tensor(cell_outputs, num_nodes_list):
#     Pdb().set_trace()
    max_num_nodes = num_nodes_list.max()
    cell_outputs_list = []
    start_at = 0
    end_at = 0
    for i,this_num_nodes in enumerate(num_nodes_list):
        end_at += this_num_nodes.item()
        cell_outputs_list.append(F.pad(cell_outputs[:,start_at:end_at,:],(0,0,0,max_num_nodes - this_num_nodes)))
        start_at = end_at
    return torch.stack(cell_outputs_list,dim=1)




def get_permutation(digit_embeddings, batch_size, board_size, should_permute_gen_embed, sort = False):
    #embedding of digit 0 at 0th index. 
    ortho_loss = get_avg_dot_product(digit_embeddings[1:].transpose(0,1).unsqueeze(0)).mean()
    if should_permute_gen_embed:    
        p = torch.ones(batch_size,digit_embeddings.shape[0] - 1)
        idx = torch.cat([torch.zeros(batch_size,1).long(), p.multinomial(num_samples=board_size)+1],dim=-1)
        if sort:
            idx = idx.sort(dim=-1)[0]
        #
        idx = idx.flatten()
    else:
        idx = torch.arange(board_size+1).repeat(batch_size)
    digit_embeddings = digit_embeddings[idx]
    return digit_embeddings, ortho_loss, idx 
 



def get_avg_dot_product(cluster_embeddings,mask_of_num_colours=None):
    epsilon = 1e-6 
    dotp = torch.matmul(cluster_embeddings.transpose(-1,-2),cluster_embeddings).abs()
    #dotp.shape = #steps x BS x max_num_colours x max_num_colours
    norms = torch.diagonal(dotp, dim1 = -2, dim2 = -1)
    #norms.shape = #steps x BS x max_num_colours
    norms = norms + (norms == 0).float()*epsilon
    ndotp = (dotp/norms.unsqueeze(-1).sqrt().detach())/(norms.unsqueeze(-2).sqrt().detach())
    
    #ndotp = torch.abs(ndotp)*torch.matmul(mask_of_num_colours.unsqueeze(-1), mask_of_num_colours.unsqueeze(1))
    if mask_of_num_colours is None:
        num_dotp = (ndotp.size(-1)*(ndotp.size(-1) -1))
    else:
        num_dotp= (mask_of_num_colours.sum(dim=-1)*(mask_of_num_colours.sum(dim=-1) - 1 )).sum()
    #
    avg_dotp = (ndotp.sum(dim=(-1,-2)) - torch.diagonal(ndotp,dim1=-1, dim2=-2).sum(dim=-1))/num_dotp
    return avg_dotp
  

def apply_attention_wts(self, lstm_emb, query_emb, board_size, batch_size, is_training):
    hidden_dim = lstm_emb.size(-1)
    orthogonality_loss = get_avg_dot_product(lstm_emb[:,1:].transpose(1,2)).mean() + get_avg_dot_product(query_emb[:,1:].transpose(1,2)).mean()
    curr_temp = 1e-8        # low temperature during testing
    if is_training:
        curr_temp = self.temperature
        self.temperature *= self.args.cooling_factor
    
    query_emb = self.gnn2lstm_embed_transformation(query_emb)
    attention_wts = torch.exp(torch.log_softmax(torch.matmul(query_emb[:,1:], lstm_emb[:,1:].transpose(-1,-2))/curr_temp, dim = -1))
    lstm_emb_exp = lstm_emb[:,1:].unsqueeze(1).expand(-1,board_size,-1,-1)
    
    input_embeddings = (lstm_emb_exp*attention_wts.unsqueeze(-1)).sum(dim=2)
    
    output_embeddings = self.input2output_embed_transformation(input_embeddings)
    orthogonality_loss += get_avg_dot_product(output_embeddings[:,1:].transpose(1,2)).mean()
    
    input_embeddings = torch.cat([lstm_emb[:,0:1,:], input_embeddings], dim = 1)
    output_embeddings = torch.cat([lstm_emb[:,0:1,:], output_embeddings], dim = 1)
    
    return {'gol': orthogonality_loss, 
            'input_embeddings': input_embeddings.view(-1,hidden_dim), 
            'output_embeddings': output_embeddings.view(-1,hidden_dim),
            'attention_wts': attention_wts
           }

