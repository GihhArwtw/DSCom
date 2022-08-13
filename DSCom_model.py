'''
=============================================
          Package Use for Model.
=============================================
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# Unfortunately, GATConv automatically apply softmax to attention weights, and the softmax operation is implemented deep in the source code of "MessagePassing".
# Thus, to get attention before softmax operation, we have to rewrite GAT layer.

'''
==============================================
       Definition of GAT Layer.
==============================================
'''


class SpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
            assert not torch.isnan(grad_values).any()
            
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
            assert not torch.isnan(grad_b).any()
        
        return None, grad_values, None, grad_b


class Spmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpmmFunction.apply(indices, values, shape, b)


class singleGATLayer(nn.Module):
    def __init__(self, in_features, out_features, slope=0.2, dropout=0.0, concat=True):
        super(singleGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.slope = slope
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(out_features, in_features)))   # W.t() is the true parameter matrix
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.slope)
        self.spmm = Spmm()
        
        self.alpha = None


    def forward(self, input, edge_index, get_att=False, add_self_loop=False):
        device = torch.device('cuda' if input.is_cuda else 'cpu')

        h = F.linear(input, self.W, bias=None)
        num_nodes = h.size()[0]
        
        # Add self loop.
        if add_self_loop:
        # Unfortunately, this takes way too long. 
        # To avoid repetitive computations, we add self loop in the DSCom_train.py
            self_loop_set = set()
            for x in edge_index:
                if (x[0]==x[1]):
                    self_loop_set.add(x[0])
                
            self_loops = []
            for x in range(num_nodes):
                if not (x in self_loop_set):
                    self_loops.append([x,x])
            self_loops = torch.tensor(self_loops, dtype=torch.long).to(device)
            edge_index = torch.cat((edge_index,self_loops), axis=0)
            
        edge = edge_index.t()
        
        # sigmoid on attention.
        alpha = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        alpha = torch.exp(-self.leakyrelu(self.a.t().mm(alpha).squeeze()))
        alpha = torch.clamp(alpha, max=5, min=-5)
        assert not torch.isnan(alpha).any()
        
        self.alpha = alpha
        self.new_edge = edge_index
        # attention before softmax (but applied sigmoid)

        ## alpha_sp = torch.sparse_coo_tensor(edge, alpha, torch.Size([num_nodes, num_nodes]))
        rowsum = self.spmm(edge, alpha, torch.Size([num_nodes, num_nodes]), torch.ones(size=(num_nodes,1), device=device))
        zeros = 1e-8*torch.ones(rowsum.shape).to(device)
        rowsum = torch.where(rowsum>0, rowsum, zeros)
        # in case rowsum is 0, which will generate NaN in torch.div
        
        alpha = self.dropout(alpha)
        
        ## alpha_sp = torch.sparse_coo_tensor(edge, alpha, torch.Size([num_nodes, num_nodes]))
        h_prime = self.spmm(edge, alpha, torch.Size([num_nodes, num_nodes]), h)
        assert not torch.isnan(h_prime).any()
        
        h_prime = h_prime.div(rowsum)
        assert not torch.isnan(h_prime).any()

        if self.concat:
            h_prime = F.elu(h_prime)
        
        if get_att:
            return h_prime, (self.new_edge.t(), self.alpha.t())
        else:
            return h_prime


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=1, slope=0.2, dropout=0.0, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.slope = slope
        self.dropout = dropout
        self.concat = concat
        
        self.attentions = [singleGATLayer(in_features, out_features, slope, dropout, concat=True) for _ in range(heads)]
        for i, att in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), att)
        
            
    def forward(self, x, edge_index, get_att=False):
        if get_att:
            
            out = None
            alpha = None
            new_edge = None
            for att in self.attentions:
                y, (e, a) = att(x, edge_index, get_att)
                if out is None:
                    out, new_edge, alpha = y, e, a.unsqueeze(dim=0)
                else:
                    out = torch.cat([out, y], dim=1)
                    alpha = torch.cat([alpha, a.unsqueeze(dim=0)], dim=0)

            return out, (new_edge, alpha)
            
        else:
            
            x = torch.cat([att(x, edge_index, get_att) for att in self.attentions], dim=1)
            return x



'''
==============================================
       Definition of DSCom Model.
==============================================
'''

class DSCom(nn.Module):
    def __init__(self, num_features, num_out, dropout=0.6):
        super(DSCom, self).__init__()
        self.in_features = num_features
        self.out_features = num_out
        self.dropout = dropout
        
        self.hidden1 = 8
        self.in_head = 8
        self.hidden2 = 4
        self.out_head = 1
        
        self.gat1 = GATLayer(num_features, self.hidden1, heads=self.in_head, dropout=dropout)
        self.gat2 = GATLayer(self.hidden1*self.in_head, self.hidden2, concat=False, heads=self.out_head, dropout=dropout)
        self.mlp = nn.Linear(self.hidden2*self.out_head, num_out)
        # self.mlp = nn.Linear(self.hidden1*self.in_head, num_out)


    def forward(self, x, edge_index):        
        x, alpha = self.gat1(x, edge_index, get_att=True)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)
        x = F.elu(x)
        
        return x, alpha  ## F.softmax(x, dim=1), alpha



from torch_geometric.nn import GATConv

class DSCom_pyg(nn.Module):
    def __init__(self, num_features, num_out, dropout=0.6):
        super(DSCom_pyg, self).__init__()
        self.in_features = num_features
        self.out_features = num_out
        self.dropout = dropout
        
        self.hidden1 = 8
        self.in_head = 8
        # self.hidden2 = 8
        # self.out_head = 1
        
        self.conv1 = GATConv(num_features, self.hidden1, heads=self.in_head, concat=False, dropout=0.6, bias=True)
        # self.conv2 = GATConv(self.hidden1*self.in_head, self.hidden2, concat=False, heads=self.out_head, dropout=0.6)
        # self.mlp = nn.Linear(self.hidden2*self.out_head, num_out)
        self.mlp = nn.Linear(self.hidden1, num_out)

    def forward(self, x, edge_index):
        x, alpha = self.conv1(x, edge_index.t(), return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.conv2(x, edge_index.t())
        # x = F.elu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)
        x = F.elu(x)
        
        return x, alpha


'''
============================================
            Negative Sampling.
============================================
'''

def negative_sampling(dataset, num_nodes, neg_spl_ratio=5, random_seed=0):
    import random
    random.seed(random_seed)
    
    samples = None    
    
    for batch in dataset:

        batch_size = len(batch[1])
        w = batch[0]
        negs_ = []
        
        for _ in range(batch_size*neg_spl_ratio):
            u = w
            while (u==w or u in negs_):
                u = random.randint(0,num_nodes-1)
            negs_ = negs_ + [u]
            
        new_batch = batch + [negs_]
        
        if (samples is None):
            samples = [new_batch]
        else:
            samples = samples + [new_batch]
        
    return samples
