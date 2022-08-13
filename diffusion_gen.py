import argparse
import os

'''
================================================
                    Arguments
================================================
'''

def str2bool(v):
    return v.lower() in ('true')

class GenOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.inited = False
        return
        
    def init(self):
        self.parser.add_argument('--dir', type=str, default='./_experiments', help='the path where checkpoints and logs are stored.\n  Defaultly sets to \'./dscom_expr\'.')
        self.parser.add_argument('--name', type=str, default='PIC_test', help='name of the experiment.\n  Decides where to store the diffusion dataset generated.\n  Defaultly sets to \'PIC_test.\'')
        self.parser.add_argument('--random_seed', type=int, default=20220727, help='the random seed to be used in the diffusion dataset generation process.')
        self.parser.add_argument('--model', type=str, default='PIC', help='the diffusion model used to simulate the information spread in reality.\n  Defaultly set to \'PIC\'.\n  Must be one of the following: \'PIC\', \'IC\', \'LT\'.')
        
        self.parser.add_argument('--dataset_node', type=str, default="./dataset_tbu/node_features.txt", help='the dir of the txt-file of node features.\n  Defaultly sets to "./dataset_tbu/node_features.txt".')
        self.parser.add_argument('--dataset_edge', type=str, default="./dataset_tbu/edges.txt", help='the dir of the txt-file of edges.\n  Defaultly sets to "./dataset_tbu/edges.txt".')
        self.parser.add_argument('--directed', type=str2bool, default=True, help='whether the network is directed or undirected. If it is directed, set to True. If not, set to false.\n  Defaultly set to True.')
        
        self.parser.add_argument('--num_chains', type=int, default=5000, help='the number of the diffusion pairs in the diffusion dataset.\n  Note that for LT, it usually requires more diffusion pairs to achieve better performance.\n  Defaultly set to 5000.')
        self.parser.add_argument('--window_len', type=int, default=5, help='the length of the sliding window on each diffusion chain.\n  We use the nodes inside the window to generate diffusion pairs.\n  Defaultly set to 5.')
        self.parser.add_argument('--v_isRandom', type=str2bool, default='True', help='whether v in PIC is generated randomly.\n  Defaultly set to True.\n  If set to False, v would be read from the file \'PIC_para.py\'.')
        self.parser.add_argument('--W_isRandom', type=str2bool, default='True', help='whether W in PIC is generated randomly.\n  Defaultly set to True.\n  If set to False, W would be read from the file \'PIC_para.py\'.')
        self.parser.add_argument('--v_len', type=int, default=8, help='the length of the vector v in PIC.\n  Required only when both v_isRandom and W_isRandom is set to True.')
        self.parser.add_argument('--scalar', type=float, default=10, help='the scalar factor in PIC diffusion model.')
        self.parser.add_argument('--offset', type=float, default=-53, help='the offset in PIC diffusion model.')
        
        return
        
    def parse(self, save=True):
        if not self.inited:
            self.init()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        print('\n------------ Options -------------')
        for key, value in sorted(args.items()):
            print('%s: %s' % (str(key), str(value)))
        print('-------------- End ---------------')

        if not os.path.exists(self.opt.dir):
            os.makedirs(self.opt.dir)
            
        expr_dir = os.path.join(self.opt.dir, self.opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
            
        tmp_dir = os.path.join(expr_dir, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            
        log_dir = os.path.join(expr_dir, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        if save:
            file_name = os.path.join(expr_dir, 'gen_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for key, value in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(key), str(value)))
                opt_file.write('-------------- End ----------------\n')
        
        return self.opt



def sigmoid(x):
    return 1./ ( 1. + np.exp(-x) )

def softmax(x):
    return np.exp(x) / np.exp(x).sum()

# probabilities that an edge will be picked during diffusion under IC Model. (i.e. 1/degree)
def deg_pr(adjacency,in_adjacency, num_nodes):
    probabilities = []
    for u in range(num_nodes):
        x = np.zeros(len(adjacency[u]))
        for i in range(len(adjacency[u])):
            x[i] = 1./ len(in_adjacency[adjacency[u][i]])
        probabilities = probabilities + [x]
    return probabilities

# probabilities that an edge will be picked during diffusion under Parameterized IC Model
def PIC_pr(adjacency, node_feature, v, W):
    probabilities = []
    for u in range(len(node_feature)):
        h_u = node_feature[u]
        x = np.empty(len(adjacency[u]))
        
        for i in range(len(adjacency[u])):
            h_w = node_feature[adjacency[u][i]]
            x[i] = v.transpose() @ np.tanh(W @ np.append(h_u,h_w))
            
        # x = softmax(opt.scalar * x + opt.offset).tolist()
        x = sigmoid(opt.scalar * x + opt.offset).tolist()
    
        probabilities = probabilities + [x]
        
    return probabilities



# diffusion simulation.
# In fact, "parameterized LT" can also be tested, i.e. use the probabilities generated by PIC_pr and set model to 'LT'.
def diffusion(adjacency, probabilities, num_nodes, window, seeds, random_seed=0, model="IC"):
    import random
    import numpy as np
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    import queue
    q = queue.Queue()
    prev = -np.ones(num_nodes,dtype=int)
    
    samples = None
    for seed in seeds:
        q.put_nowait(seed)

    ## try:
    if (model=="IC" or model=='PIC'):
        visited = np.zeros(num_nodes,dtype=bool)
        for seed in seeds:
            visited[seed] = True
        
        while (not q.empty()):
            u = q.get_nowait()
            x = probabilities[u]

            for i in range(len(adjacency[u])):
                r = random.uniform(0,1)
                node = adjacency[u][i]
                
                if ((not visited[node]) and r<x[i]):
                    q.put_nowait(node)  # Activate node.
                    prev[node] = u
                    tmp = 1
                    pos_ = [node]
                    v = node
                    
                    while (prev[v]>-1 and tmp<window):
                        v = prev[v]
                        pos_ = pos_ + [v]
                        tmp = tmp+1
                    
                    if (samples is None):
                        samples = [pos_]
                    else:
                        samples += [pos_]
                    
                visited[node] = True
                
    
    elif model=="LT":
        
        thr = np.random.uniform(0,1,size=num_nodes)
        weight = np.zeros(num_nodes,dtype=float)
        lt = queue.Queue()
        tmp = queue.Queue()
        prev_st = -np.ones(num_nodes,dtype=int)
        prev_en = -np.ones(num_nodes,dtype=int)
        
        for seed in seeds:
            weight[seed] = thr[seed]+1e-6
        
        while (not q.empty()):
            u = q.get_nowait()
            x = probabilities[u]

            for i in range(len(adjacency[u])):
                node = adjacency[u][i]
                
                if (weight[node]>thr[node] and prev[node]<0):
                    q.put_nowait(node)  # Activate node.
                    pos_ = [node]

                    visited = np.zeros(num_nodes,dtype=bool)
                    while (not lt.empty()):
                        lt.get_nowait()
                    while (not tmp.empty()):
                        tmp.get_nowait()
                        
                    prev_st[node] = 0
                        
                    for j in range(len(adjacency[node])):
                        v = adjacency[node][j]
                        if (weight[v]>thr[v]):
                            lt.put_nowait(v)
                            tmp.put_nowait(2)
                            visited[v] = True
                            pos_ += [v]
                                
                    prev_en[node] = len(pos_)
                    
                    w = 2
                    while ((not lt.empty()) and w<window):
                        v = lt.get_nowait()
                        w = tmp.get_nowait()
                        if (prev[v]>-1):
                            ngh = samples[prev[v]][1:]
                            
                            for y in ngh:
                                if (not visited[y]):
                                    lt.put_nowait(y)
                                    tmp.put_nowait(w+1)
                                    visited[y] = True
                                    pos_ += [y]
                                    
                    if samples is None:
                        prev[node] = 0
                        samples = [pos_]
                    else:
                        prev[node] = len(samples)
                        samples += [pos_]
                    
    else:
        raise ValueError("argument \'model\' is neither \"LT\" nor \"IC\".")

    ## except ValueError as e:
    ##     print(repr(e))
    
    return samples



'''
============================================
                    MAIN
============================================
'''
    
if __name__ == '__main__':

    opt = GenOptions().parse()

    import numpy as np
    import networkx as nx
    import random
    np.random.seed(opt.random_seed)
    random.seed(opt.random_seed)
    
    path = opt.dir+"/"+opt.name+"/tmp/"
    
    # Loading dataset.
    node_feature = np.loadtxt(open(opt.dataset_node),dtype=int,delimiter=" ",skiprows=0)
    edges = np.loadtxt(open(opt.dataset_edge),dtype=int,delimiter=" ",skiprows=0)
    
    if opt.directed:
        adjacency = nx.to_dict_of_lists(nx.from_edgelist(edges, create_using=nx.DiGraph()))
        edges_ch = edges.t()
        edges_ch = np.array([edges_ch[1],edges_ch[0]]).t()
        in_adjacency = nx.to_dict_of_lists(nx.from_edgelist(edges_ch, create_using=nx.DiGraph()))
    else:
        adjacency = nx.to_dict_of_lists(nx.from_edgelist(edges))
        in_adjacency = adjacency
    
    num_nodes = len(node_feature)
    
    # in case some nodes are not connected to any other nodes.
    for i in range(num_nodes):
        if not (i in adjacency):
            adjacency.update({i:[]})
            
    
    # Append the degree and 1/degree into the feature.
    deg = np.ndarray((num_nodes,2))
    deg_avg = 0
    for node in range(num_nodes):
        if (len(adjacency[node])==0):
            deg[node][0] = 0
            deg[node][1] = 10.    # in fact, "inf"
        else:
            deg[node][0] = len(adjacency[node])
            deg[node][1] = 1./len(adjacency[node])
        
        deg_avg += deg[node][0]
        
    node_feature = np.concatenate((node_feature,deg),axis=1)
    deg_avg /= float(num_nodes)
    
    
    # Normalize node features.
    feature_max = np.max(node_feature,axis=0)
    feature_min = np.min(node_feature,axis=0)
    node_feature = node_feature-feature_min
    std = feature_max-feature_min
    std = np.where(std>0, std, np.ones(std.shape))
    node_feature = node_feature/std
    
    num_features = len(node_feature[0])
    
    # Save preprocessed node features.
    np.save(path+"tmp_nodes.npy",node_feature)
    np.savetxt(path+"tmp_nodes.txt",node_feature, fmt='%.6f', delimiter=' ', newline='\n')
    
    
    # Preprocess edges.
    edge_index = set()
    for x in edges:
        edge_index.add(x[0]*num_nodes+x[1])
    
    edges_inverse = []
    for x in edges:
        if not (x[1]*num_nodes+x[0] in edge_index):
            edges_inverse.append([x[1],x[0]])
    edges = np.concatenate((edges,edges_inverse), axis=0)
    
    print("Node: {}\t\t Preprocessed Edges: {}".format(num_nodes, len(edges)))
    
    np.save(path+"tmp_edges.npy",edges)
    np.savetxt(path+"tmp_edges.txt",edges)

    
    # Loading diffusion model.
    if (opt.model=='IC' or opt.model=='LT'):
        probabilities = deg_pr(adjacency, in_adjacency, num_nodes)
        
    else:
        lenv = opt.v_len
        if (not opt.W_isRandom):
            from PIC_para import W
            W = np.array(W)
            lenv = len(W)
        else:
            W = np.random.random((lenv, 2*num_features))
        
        if (not opt.v_isRandom):
            from PIC_para import v
            v = np.array(v)
        else:
            v = np.random.random(lenv)
            
        np.save(path+"v.npy",v)
        np.save(path+"W.npy",W)
            
        probabilities = PIC_pr(adjacency, node_feature, v, W)
        
        
    if opt.model=='LT':
        for i in range(len(probabilities)):
            probabilities[i] = probabilities[i] * random.uniform(0.9,1)
            
    # save weighted edges for IM methods requiring diffusion models, i.e. all methods so far, e.g. IMM, SSA, MAIM, etc.
    '''
    weight = []
    for [u,v] in edges:
        x = adjacency[u].index(v)
        w = probabilities[u][x]
        if opt.model=='LT':
            w = w * random.uniform(0.9,1)
        weight.append(w)
    '''
        
    if opt.model=='LT':
        np.save(path+"tmp_diff_model.npy",np.array([1]))
    else:
        np.save(path+"tmp_diff_model.npy",np.array([-1]))
        
    ## weight = np.concatenate((edges.transpose(),[weight]),axis=0).transpose()
    ## weight = np.concatenate((flag,weight),axis=0)
    ## np.savetxt(path+"tmp_weighted_edges.txt",weight, fmt="%.6e", delimiter=' ', newline='\n')
    
    with open(path+"tmp_weighted_edges.txt", "w") as file:
        for u in range(num_nodes):
            for x in range(len(adjacency[u])):
                w = probabilities[u][x]
                print("{} {} {:.6f}".format(u, adjacency[u][x], w), file=file)
        
    
    # diffusion dataset generation.
    pairs = None
    count = 0
    while (pairs is None):
        num_nodes_ = random.randint(int(deg_avg/2), int(num_nodes/deg_avg)+1)
        nodes_ = []
        while (len(nodes_)<num_nodes_):
            node_ = random.randint(0, num_nodes-1)
            while (node_ in nodes_):
                node_ = random.randint(0, num_nodes-1)
            nodes_.append(node_)
        
        for _ in range(int(deg_avg)+1):
            new_pairs = diffusion(adjacency, probabilities, num_nodes, opt.window_len, nodes_, opt.random_seed, model=opt.model)
            if (new_pairs is None): continue
            if (pairs is None):
                pairs = new_pairs
            else:
                pairs = pairs + new_pairs
        
        count += 1
        if (count>1000):
            raise ValueError('Diffusion Model might be full of 0.')

    while (len(pairs)<opt.num_chains):
        num_nodes_ = random.randint(int(deg_avg/2), int(num_nodes/deg_avg)+1)
        nodes_ = []
        while (len(nodes_)<num_nodes_):
            node_ = random.randint(0, num_nodes-1)
            while (node_ in nodes_):
                node_ = random.randint(0, num_nodes-1)
            nodes_.append(node_)
            
        for _ in range(int(deg_avg)+1):
            new_pairs = diffusion(adjacency, probabilities, num_nodes, opt.window_len, nodes_, opt.random_seed, model=opt.model)
            if (new_pairs is None): continue
            pairs = pairs + new_pairs
        

    import pandas as pd
    df = pd.DataFrame(pairs[:opt.num_chains])
    df.to_csv(path+'pairs.csv',index=False)  


'''
=============================================
     Explanation of the Output Files
=============================================

[tmp_node.npy][tmp_node.txt]
    the node feature with "degree" and "1/degree" feature appended.
    
[v.npy]
    the v parameter in PIC model.
    
[W.npy]
    the W parameter in PIC model.

[pairs.csv]
    For each column, rows with positive indices contains
        several nodes close to the node in row 0 on the 
        diffusion chains.
    Let pairs(i)(0) = u. Then for any j>0 s.t. pairs(i)(j)=v
        is not None, pair(u,v) is a positive sample.
'''
