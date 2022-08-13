import argparse
import os

'''
================================================
                    Arguments
================================================
'''

def str2bool(v):
    return v.lower() in ('true')

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.inited = False
        return
        
    def init(self):
        self.parser.add_argument('--dir', type=str, default='./_experiments', help='the path where dataset temp files, checkpoints and logs are stored.\n  Must be the same one used when generating diffusion dataset.\n  Defaultly sets to \'./dscom_expr\'.')
        self.parser.add_argument('--name', type=str, default='PIC_test', help='name of the experiment.\n  Must be the same one used when generating diffusion dataset.\n  Defaultly sets to \'PIC_test.\'')
        self.parser.add_argument('--random_seed', type=int, default=20220727, help='the random seed to be used in the diffusion dataset generation process.')
        
        self.parser.add_argument('--continue_train', type=str2bool, default=False, help='whether continue the training of a pretrained model.\n  Defaultly set to False.')
        self.parser.add_argument('--pretrained_model', type=str, default=None, help='the path of the pretrained model.\n  Must be the path of a \'.pth\' file.')
        
        self.parser.add_argument('--learning_rate', type=float, default=0.01, help='the learning rate when training DSCom.\n  Defaultly set to 0.01.')
        self.parser.add_argument('--dropout', type=float, default=0.6, help='the dropout rate when training DSCom.\n  Defaultly set to 0.6.')
        self.parser.add_argument('--num_epoch', type=int, default=1000, help='the number of training epoches.\n  Defaultly set to 10000.')
        self.parser.add_argument('--num_out_emb', type=int, default=6, help='the length of the output embedding, which is a vector.\n  Defaultly set to 6.')
        self.parser.add_argument('--neg_spl_ratio', type=int, default=5, help='the negative sampling ratio, i.e. #(negative samples)/#(positive samples).\n  Defaultly set to 5.')
        ## self.parser.add_argument('--minibatch', type=str2bool, default=True, help='whether apply mini-batch training or not.\n  Defaultly set to True.\n  Recommend using mini-batch training, especially when the grah is massive with millions or even billions of nodes or edges.')
        
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
            raise ValueError("DIR or NAME ERROR. The path of dataset does not exist.")
            
        if save:
            file_name = os.path.join(expr_dir, 'train_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('--------------------- Options ----------------------\n')
                for key, value in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(key), str(value)))
                opt_file.write('----------------------- End ------------------------\n')
        
        return self.opt


'''
============================================
                   MAIN
============================================
'''


if __name__ == "__main__":
    
    opt = TrainOptions().parse()
    
    import numpy as np
    
    path = opt.dir+"/"+opt.name+"/"
    
    
    # Load dataset: graph information.
    edge_index = np.load(path+"tmp/tmp_edges.npy")
    node_feature = np.load(path+"tmp/tmp_nodes.npy")
    
    num_nodes = len(node_feature)
    num_features = len(node_feature[0])
    
    print("Node: {}\t\t Preprocessed Edges: {}".format(num_nodes, len(edge_index)))
    
    
    # Load dataset: diffusion dataset.
    import pandas as pd
    df = pd.read_csv(path+'tmp/pairs.csv')
    df = df.values.tolist()
    dataset = None
    
    for line in df:
        pos_ = []
        
        for i in line[1:]:
            if np.isnan(i):  break
            if pos_ is None:
                pos_ = [int(i)]
            else:
                pos_ = pos_ + [int(i)]
        
        if dataset is None:
            dataset = [[line[0], pos_]]
        else:
            dataset = dataset + [[line[0], pos_]]
    
    # =========================================
    #            Training Process
    # =========================================
    
    
    import torch
    ## from torch.nn.utils import clip_grad_norm_
    ## import dgl
    ## import dgl.nn as dglnn
    from DSCom_model import DSCom, DSCom_pyg, negative_sampling
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dscom_time = time.perf_counter()
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
    node_feature = torch.Tensor(node_feature).to(device)
    
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
    edge_index = torch.cat((edge_index,self_loops), axis=0)
    
    # 
    '''
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.NodeDataLoader(
        g, train_nids, sampler,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)
    '''
    
    
    # DSCom Model Definition.
    
    model = DSCom(num_features, opt.num_out_emb, opt.dropout).to(device)
    ## model = DSCom(num_features, opt.num_out_emb, opt.dropout).to(device)
    if opt.continue_train:
        model = torch.load(opt.pretrained_model, map_location=device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=5e-4)

    best_loss = float("inf")
    best_model = None
    
    print("\n=============== DSCOM: START TRAINING ==============\n")
    with open(path+"train_log.txt","w") as file:
        print("\n=============== DSCOM: START TRAINING ==============\n", file=file)
    
    # Training.
    ## with torch.autograd.set_detect_anomaly(True):
    for epoch in range(opt.num_epoch):
        model.train()
        optimizer.zero_grad()
        out_emb, alpha = model(node_feature, edge_index)
        
        train_dataset = negative_sampling(dataset, num_nodes, neg_spl_ratio=opt.neg_spl_ratio, random_seed=opt.random_seed)
        
        loss = torch.tensor(0, dtype=torch.float)
        for batch in train_dataset:
            curr = torch.tensor(batch[0], dtype=torch.long)
            pos = torch.tensor(batch[1], dtype=torch.long)
            neg = torch.tensor(batch[2], dtype=torch.long)
            
            loss = torch.add(loss, torch.sum( - torch.log( torch.sigmoid( torch.clamp(out_emb[pos]@out_emb[curr], max=5, min=-5) ) ) ) )         
            '''
            if torch.isnan(loss) or torch.isinf(loss):
                print("========\n\tNan/Inf Loss Encountered in Positive Samples.")
                print(pos)
                print(out_emb[pos])
                print(torch.clamp(out_emb[pos]@out_emb[curr],min=-5,max=5))
                print(torch.sigmoid( torch.clamp(out_emb[pos]@out_emb[curr],min=-5,max=5) ) )
            '''    
            
            loss = torch.add(loss, torch.sum( - torch.log( 1. - torch.sigmoid( torch.clamp(out_emb[neg]@out_emb[curr], max=5, min=-5) ) ) ) )
            '''
            if torch.isnan(loss) or torch.isinf(loss):
                print("========\n\tNan/Inf Loss Encountered in Negative Sampling.")
                print(torch.clamp(out_emb[neg]@out_emb[curr],min=-5,max=5))
                print( 1. - torch.sigmoid( torch.clamp(out_emb[neg]@out_emb[curr],min=-5,max=5) ) )
            '''
            
        if (epoch%100 == 0):
            print("Epoch {}\t Loss: {:.8f}".format(epoch, loss))
            with open(path+"train_log.txt","a") as file:
                print("Epoch {}\t Loss: {:.8f}".format(epoch, loss), file=file)
            torch.save(best_model, path+"model_best.pth")
            if (epoch%500 == 0):
                torch.save(model, path+"model_"+str(epoch)+".pth")
        
        if (best_loss>loss):
            best_loss = loss
            best_model = model
                
        loss.backward()
        optimizer.step()
                
        
    torch.save(best_model, path+"model_best.pth")
    print("\nBEST MODEL - Loss: {:.8}".format(best_loss))
    with open(path+"train_log.txt","a") as file:
        print("\nBEST MODEL - Loss: {:.8}".format(best_loss), file=file)
    
    dscom_time = time.perf_counter() - dscom_time
    print("\nTRAINING TIME: {}".format(dscom_time))
    with open(path+"train_log.txt","a") as file:
        print("\nTRAINING TIME: {}".format(dscom_time), file=file)
    
    print("\n============ DSCOM: TRAINING FINISHED. ===========\n")
    with open(path+"train_log.txt","a") as file:
        print("\n============ DSCOM: TRAINING FINISHED. ===========\n", file=file)
    