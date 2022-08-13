import argparse
import os

'''
================================================
                    Arguments
================================================
'''

def str2bool(v):
    return v.lower() in ('true')

class PredOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.inited = False
        return
        
    def init(self):
        self.parser.add_argument('--dir', type=str, default='./_experiments', help='the path where dataset temp files, checkpoints and logs are stored.\n  Must be the same one used when generating diffusion dataset.\n  Defaultly sets to \'./dscom_expr\'.')
        self.parser.add_argument('--name', type=str, default='PIC_test', help='name of the experiment.\n  Must be the same one used when generating diffusion dataset.\n  Defaultly sets to \'PIC_test.\'')
        self.parser.add_argument('--ablation', type=str2bool, default=False, help='whether conduct abalation experiments.\n  Defaultly set to Fasle.')
        
        self.parser.add_argument('--model_under_dir', type=str2bool, default=True, help='whether the pth file of the DSCom model to be used is under path [dir]/[name].\n  Defaultly set to True.')
        self.parser.add_argument('--dscom_model', type=str, default='model_best.pth', help='name of the pth file of the DSCom model to be used.\n  If [model_under_dir] is set to True, then the file must be under the path [dir]/[name].')
        
        self.parser.add_argument('--num_seeds', type=int, default=20, help='the number of seeds selected for influence maximization problem.\n  Defaultly set to 20.')
        self.parser.add_argument('--dynamic', type=str2bool, default=False, help='whether the current dataset is a dynamic graph with its previous status already clustered.\n  Defaultly set to False.')
        self.parser.add_argument('--dynamic_base_under_dir', type=str2bool, default=True, help='whether the npy file of the centroids of all clusters of the previous status already clustered is under path [dir]/[name].\n  Defaultly set to True.')
        self.parser.add_argument('--dynamic_base', type=str, default=None, help='name of the npy file of the centroids of all clusters of the previous status already clustered.\n  If [dynamic_base_under_dir] is set to True, then the file must be under the path [dir]/[name].')
        
        return
        
    def parse(self, save=True):
        if not self.inited:
            self.init()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        
        print('\n--------------------- Options ----------------------')
        for key, value in sorted(args.items()):
            print('%s: %s' % (str(key), str(value)))
        print('----------------------- End ------------------------')

        if not os.path.exists(self.opt.dir):
            raise ValueError("DIR ERROR. The path of dataset does not exist.")
            
        expr_dir = os.path.join(self.opt.dir, self.opt.name)
        if not os.path.exists(expr_dir):
            raise ValueError("NAME ERROR. The path of dataset does not exist.")
        
        tmp_dir = os.path.join(expr_dir, 'tmp')
        if not os.path.exists(tmp_dir):
            raise ValueError("DIR or NAME ERROR. TMP missing. The path of dataset does not exist.")
        
        log_dir = os.path.join(expr_dir, 'log')
        if not os.path.exists(log_dir):
            raise ValueError("DIR or NAME ERROR. LOG missing. The path of dataset does not exist.")
            
        
        if save:
            file_name = os.path.join(expr_dir, 'pred_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('--------------------- Options ----------------------\n')
                for key, value in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(key), str(value)))
                opt_file.write('----------------------- End ------------------------\n')
        
        if self.opt.model_under_dir:
            self.opt.dscom_model = self.opt.dir+'/'+self.opt.name+'/'+self.opt.dscom_model
        
        if (self.opt.dynamic and self.opt.dynamic_base_under_dir):
            self.opt.dynamic_base = self.opt.dir+'/'+self.opt.name+'/'+self.opt.dynamic_base
        
        return self.opt



def Core_k_truss(subgraph,k_min,k_max):
    
    import networkx as nx
    
    if (k_min+1 >= k_max):
        truss = nx.k_truss(subgraph,k_min)
        return truss.nodes
    
    k = int((k_min+k_max)/2)
    truss = nx.k_truss(subgraph,k)
    
    if (len(truss)==0):
        return Core_k_truss(subgraph,k_min,k-1)
    if (len(truss)>0):
        return Core_k_truss(subgraph,k,k_max)



'''
============================================
                   MAIN
============================================
'''



if __name__ == "__main__":
    
    opt = PredOptions().parse()
    
    import numpy as np
    import torch
    import networkx as nx
    import time
    
    path = opt.dir+"/"+opt.name+"/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clstr_time  = time.perf_counter()
    
    # Load dataset: graph information.
    edge_index = np.load(path+"tmp/tmp_edges.npy")
    node_feature = np.load(path+"tmp/tmp_nodes.npy")
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
    node_feature = torch.Tensor(node_feature).to(device)
    
    num_nodes = len(node_feature)
    num_features = len(node_feature[0])
    
    print("\n============== DSCOM: START SELECTING ==============\n")
    
    
    
    # =========================================
    #            Prediction Process
    # =========================================
    
    
    
    # To use "DSCom" in DSCom_model: Add self loops.
    # If use "DSCom_pyg", then it is unnecessary.
    self_loop_set = set()
    for x in edge_index:
        if (x[0]==x[1]):
            self_loop_set.add(x[0])
        
    self_loops = []
    for x in range(num_nodes):
        if not (x in self_loop_set):
            self_loops.append([x,x])
    self_loops = torch.tensor(self_loops, dtype=torch.long).to(device)
    new_edge = torch.cat((edge_index,self_loops), axis=0)
    
    
    # Loading Trained Model.
    from DSCom_model import DSCom, DSCom_pyg
    
    model = torch.load(opt.dscom_model, map_location=device)
    model.eval()
    
    out_emb, (new_edge, alpha) = model(node_feature, edge_index)
    out_emb = out_emb.detach().cpu().numpy()
    
    print("\nModel Loading Fin.")
    
    # For DSCom only (not using pyG package)
    alpha = torch.sigmoid(alpha)
    alpha_judge = alpha>1.
    assert not alpha_judge.any()
    
    alpha = alpha.t()
    
    if (not opt.dynamic):
        new_adjacency = np.zeros((num_nodes, num_nodes))
        new_edge = new_edge.long()
        
        for i in range(len(new_edge[0])):
            new_adjacency[int(new_edge[0][i])][int(new_edge[1][i])] = alpha[i].mean()
            
        with open(path+"tmp/tmp_rel.txt", "w") as file:
            for i in range(len(new_edge[0])):
                print("{} {} {:.8f}".format(int(new_edge[0][i]),int(new_edge[1][i]),alpha[i].mean()),file=file)
            
        new_adjacency = 0.5 * ( new_adjacency + new_adjacency.transpose() )
        # to make sure it is symmetric.
        
        
        # Spectral Clustering.
        from sklearn.cluster import spectral_clustering
        
        clusters = spectral_clustering(new_adjacency, n_clusters = opt.num_seeds)
        centroids = []
        comm_nodes = []

        for i in range(opt.num_seeds):
            comm_nodes.append([x for x in range(len(out_emb)) if clusters[x]==i])
            clu_feature = np.mean(out_emb[comm_nodes[-1]], axis=0)
            centroids.append(clu_feature)
        ## print(centroids)
        
        centroids = np.array(centroids)
        np.save(path+f"{opt.num_seeds}_centroids.npy",centroids)
        
    else:
        
        centroids = np.load(opt.dynamic_base)
        comm_nodes = [[] for _ in range(opt.num_seeds)]
        
        for i in range(num_nodes):
            dist = centroids - out_emb[i]
            dist = [np.linalg.norm(x) for x in dist]
            cluster = dist.index(min(dist))
            comm_nodes[cluster].append(i)
    
    
    
    # Separate each community.
    communities = []
    for i in range(opt.num_seeds):
        sub_nodes = comm_nodes[i]
        x = [x for x in range(len(new_edge)) if (new_edge[x][0] in sub_nodes and new_edge[x][1] in sub_nodes)]
        sub_edges = new_edge[x]
        sub_alpha = alpha[x]
        communities = communities + [[sub_nodes, sub_edges, sub_alpha]]
    
    print("Commuity Division Fin. Start selecting seeds.")
    
    
    
    # Seed selection in each community.
    clstr_time = time.perf_counter()-clstr_time
    
    seeds_mxdg = []      # choose the node with maximal degeree
    seeds_mxct = []      # choose the node with maximal betweenness-centrality
    seeds_core = []      # k-core
    seeds_pgRk = []      # PageRank
    time_mxdg = clstr_time
    time_mxct = clstr_time
    time_core = clstr_time
    time_pgRk = clstr_time
    
    
    for i in range(opt.num_seeds):
    
        pre_time = time.perf_counter()
        
        # community subgraph construction.
        subgraph = nx.Graph()
        subgraph.add_nodes_from(communities[i][0])
        sub_edges = communities[i][1]
        sub_alpha = communities[i][2]
        for x in range(len(sub_edges)):
            subgraph.add_edge(int(sub_edges[x][0]),int(sub_edges[x][1]),weight=int(sub_alpha[x].mean()))
        
        pre_time = time.perf_counter()-pre_time
        subtime = time.perf_counter()
        
        
        # by max-degree
        core_num = []
        indices = []
        for j in subgraph.nodes:
            core_num.append(subgraph.degree[j])
            indices.append(j)
        core = indices[core_num.index(max(core_num))]
        seeds_mxdg.append(core)
        
        time_mxdg += (time.perf_counter()-subtime+pre_time)
        subtime = time.perf_counter()
        
        
        # by max-closeness-centrality
        core_num = nx.algorithms.centrality.closeness_centrality(subgraph)
        core = max(core_num, key=core_num.get)
        seeds_mxct.append(core)
        
        time_mxct += (time.perf_counter()-subtime+pre_time)
        subtime = time.perf_counter() 
        
        
        # by pageRank
        core_num = nx.pagerank(subgraph, alpha=0.5)
        core = max(core_num, key=core_num.get)
        seeds_pgRk.append(core)
        
        time_pgRk += (time.perf_counter()-subtime+pre_time)
        subtime = time.perf_counter() 
        
        
        # by k-core
        for e in subgraph.edges:
            if (e[0]==e[1]):
                subgraph.remove_edge(e[0],e[1])
                
        core_num = nx.core_number(subgraph)
        core = max(core_num, key=core_num.get)
        seeds_core.append(core)
        
        time_core += (time.perf_counter()-subtime+pre_time)
        subtime = time.perf_counter()
        
        print(f"Seed selection in Community {i} Fin.")
    
    
    
    # save the results.
    time_all = np.array([time_mxdg, time_mxct, time_core, time_pgRk])
    
    print("Number of Seeds: {}".format(opt.num_seeds))
    print("\nSeeds:")
    print("Max_Degree:     {}".format(seeds_mxdg))
    print("Max_Centrality: {}".format(seeds_mxct))
    print("k-core:         {}".format(seeds_core))
    print("pageRank:       {}".format(seeds_pgRk))
    print("\nTime:")
    print("Spec Clustering:{}".format(clstr_time))
    print("Max_Degree:     {}".format(time_mxdg))
    print("Max_Centrality: {}".format(time_mxct))
    print("k-core:         {}".format(time_core))
    print("pageRank:       {}".format(time_pgRk))
    
    with open(path+f'log/{opt.num_seeds}_dscom.txt',"w") as file:
        print("Number of Seeds: {}".format(opt.num_seeds), file=file)
        print("\nSeeds:", file=file)
        print("Max_Degree:     {}".format(seeds_mxdg), file=file)
        print("Max_Centrality: {}".format(seeds_mxct), file=file)
        print("k-core:         {}".format(seeds_core), file=file)
        print("pageRank:       {}".format(seeds_pgRk), file=file)
        print("\nTime:", file=file)
        print("Max_Degree:     {}".format(time_mxdg), file=file)
        print("Max_Centrality: {}".format(time_mxct), file=file)
        print("k-core:         {}".format(time_core), file=file)
        print("pageRank:       {}".format(time_pgRk), file=file)
        
    seeds_mxdg = np.array(seeds_mxdg)
    seeds_mxct = np.array(seeds_mxct)
    seeds_core = np.array(seeds_core)
    seeds_pgRk = np.array(seeds_pgRk)
    np.save(path+f'log/{opt.num_seeds}_dscom_mxdg.npy',seeds_mxdg)
    np.save(path+f'log/{opt.num_seeds}_dscom_mxct.npy',seeds_mxct)
    np.save(path+f'log/{opt.num_seeds}_dscom_core.npy',seeds_core)
    np.save(path+f'log/{opt.num_seeds}_dscom_pgRk.npy',seeds_pgRk)
    np.save(path+f'log/{opt.num_seeds}_dscom_time.npy',time_all)
    
    
    print("====== DSCom Seed Selection Fin. ======")
    
    
    
    # =========================================
    #           Ablation Experiments
    # =========================================
    
    
    
    if opt.ablation:
        print("\n\n======== DSCom Ablation Studies =======")
        print("\nNumber of Seeds: {}".format(opt.num_seeds))
        print("\nSeeds:")
        
        
        from sklearn.cluster import k_means
        
        
        # w.o. Community Division, i.e. GAT -> out_emb -> k-means
        centroid, label, _ = k_means(out_emb, opt.num_seeds, init='k-means++')
        
        seeds_gatk = []
        for i in range(opt.num_seeds):
            cluster = [x for x in range(len(label)) if label[x]==i]
            dist = out_emb[cluster] - centroid[i]
            dist = [np.linalg.norm(x) for x in dist]
            core = cluster[dist.index(min(dist))]
            seeds_gatk.append(core)
            
        print("GATK    (GAT -> out -> k-means):   {}".format(seeds_gatk))
        seeds_gatk = np.array(seeds_gatk)
        np.save(path+f'log/abl_{opt.num_seeds}_gatk.npy',seeds_gatk)
        
        
        # w.o. Community Division, i.e. GAT -> atten -> IMM
        # to test how good we are at relation learning.
        from tools import readGraph_direct
        from comp_algos.IMM.IMM import IMM
        
        with open(path+"tmp/tmp_rel.txt", "w") as file:
            for i in range(len(new_edge[0])):
                print("{} {} {:.8f}".format(int(new_edge[0][i]),int(new_edge[1][i]),alpha[i].mean()),file=file)
        
        graph = readGraph_direct(path+"tmp/tmp_rel.txt")
        seeds_rl_imm = IMM(graph, opt.num_seeds, 0.1, 3)
        
        print("RL-IMM  (GAT -> attention -> IMM): {}".format(seeds_rl_imm))
        seeds_rl_imm = np.array(list(seeds_rl_imm))
        np.save(path+f'log/abl_{opt.num_seeds}_rl_imm.npy',seeds_rl_imm)
                
        
        # w.o. Relation Learning, i.e. Spectral Clustering -> pageRank
        adj = np.zeros((num_nodes,num_nodes))
        for x in edge_index:
            adj[x[0],x[1]] = 1
           
        adj = 0.5 * ( adj + adj.transpose() )
        from sklearn.cluster import spectral_clustering
        pure_cls = spectral_clustering(adj, n_clusters = opt.num_seeds)
        
        seeds_spec_pr = []
        for i in range(opt.num_seeds):
            subgraph = nx.Graph()
            subgraph.add_nodes_from([x for x in range(num_nodes) if pure_cls[x]==i])
            
            x = [x for x in range(len(edge_index)) if (edge_index[x][0] in sub_nodes and edge_index[x][1] in sub_nodes)]
            sub_edges = edge_index[x]
            for x in range(len(sub_edges)):
                subgraph.add_edge(int(sub_edges[x][0]),int(sub_edges[x][1]))
            
            core_num = nx.pagerank(subgraph, alpha=0.5)
            core = max(core_num, key=core_num.get)
            seeds_spec_pr.append(core)
        
        print("Spec-PR (Spec Clstr -> pageRank):  {}".format(seeds_spec_pr))
        
        
        with open(path+f'log/abl_{opt.num_seeds}_dscom.txt', "w") as file:
            print("Number of Seeds: {}".format(opt.num_seeds), file=file)
            print("\nSeeds:", file=file)
            print("GATK    (GAT -> out -> k-means):   {}".format(list(seeds_gatk)), file=file)
            print("RL-IMM  (GAT -> attention -> IMM): {}".format(list(seeds_rl_imm)), file=file)
            print("Spec-PR (Spec Clstr -> pageRank):  {}".format(seeds_spec_pr), file=file)
        
        seeds_spec_pr = np.array(seeds_spec_pr)
        np.save(path+f'log/abl_{opt.num_seeds}_spec_pr.npy',seeds_spec_pr)
        
        print("===== DSCom Ablation Studies Fin. =====")
        