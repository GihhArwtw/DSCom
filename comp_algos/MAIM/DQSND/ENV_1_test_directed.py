import time
import math
from random import random
from copy import deepcopy
import networkx as nx
import os
import psutil
p = 0.001
r = 1 - p


class Env:
    ''' The General ENV, namely the graph
    Input: file -- address of the graph data
    Input: budget -- size of budget
    Comment: the structure in which we run our model
    '''
    def __init__(self, file, budget = 50):
        process = psutil.Process(os.getpid())
        start_M = process.memory_info().rss  # in bytes
        self.netInput = []
        self.G = []
        self.graph ={}
        self.seed = set()
        self.budget = budget
        self.graph,self.G = read_from_txt(file)
        self.graph_ = deepcopy(self.graph)
        # Virtual graph
        # Node first tested on it
        # If accepted, replace graph with graph_
        print("Cock")
        start = time.time()
        self.list = strength_list(self.graph,self.G,self.budget)
        self.maxGain = self.list[0][1]
        print(time.time()-start)
        print("Cock")
        end_M = process.memory_info().rss  # in bytes
        print("Memory used",end_M - start_M, "bytes", flush=True)  # in bytes
        #print(self.graph)
    def update_graph(self):
        '''
        Replace graph by graph_
        '''
        temp = self.graph
        self.graph = self.graph_
        del(temp)

    def backup_graph(self):
        '''
        Update the graph_
        '''
        self.graph_ = deepcopy(self.graph)

    def list(self):
        '''
        Output: list of nodes ordered by degree with marginal gain of each
        '''
        return self.list

    def steps(self, step, A, T):
        '''
        Make action, steps to next stage
        Input: step -- index of node stepped
        Input: A -- bool, if this node is really added into seed set
        Input: T -- bool, if we are trainning
        Output: R -- Reward, marginal gain of a node, later used as machine Reward
        '''
        node = {self.list[step][0]}

        # No need to cal gain if not training
        if T == 1:
            R =  IC(self.graph, self.seed | node) - IC(self.graph, self.seed)
        else:
            R = 0

        # If this action is really made
        # 1. Add the node into seed set
        # 2. Update the graph (Collided Nodes)
        if A == 1:
            self.seed = self.seed | node
            self.update_graph()

        return R

    def reward(self,seed):
        '''
        Run IC and gain of whole seed set on graph
        '''
        return IC(self.graph,seed)

    def node2feat(self,step):
        '''
        Covert a node index to its attributes: individual influence and collision
        backup graph in this stage, since it's the first step to work
        Input: step -- index of current node in self.list
        Output (private): self.netInput
        1. influence ratio
        2. collision ratio (self not included)
        Output: influence -- influence of individual node
        Output: coll -- collision of the node with seed set
        '''
        # Preparation
        # Update the graph_
        node = {self.list[step][0]}
        self.backup_graph()

        influence,coll,inf,touch = collision(self.graph,self.graph_,node, 0)
        if influence == 1:
            influence = 1.5
        self.netInput = [touch/(influence - 1),inf/(influence-1),coll/(influence-1)]
        return influence,coll

def read_from_txt(filename):
    """
    From TXT file read node pairs and assign active probability to edge by random
    Input: filename -- TXT file address
    Input: p -- weight of each edge
    Output: g, a graph in dict
    snap, hundreds of snapshots
    """
    g = {}
    G = []

    with open(filename) as f:
        lines = f.readlines()[5:]
        for line in lines:
            line = line.replace('\t',' ')
            e = [int(s) for s in line.replace('\n', '').split(' ')]
            #e = [int(s) for s in line.replace('\n', '').split(' ')]

            if e[0] in g.keys():
                if e[1] in g[e[0]].keys():
                    x = g[e[0]][e[1]]
                    g[e[0]][e[1]] = 1 + (x-1)*r
                else:
                    g[e[0]][e[1]] = p
            else:
                g[e[0]] = {"On": 0}
                g[e[0]][e[1]] = p

            if e[1] in g.keys():
                pass
            else:
                g[e[1]] = {"On": 0}
                g[e[1]][e[0]] = 0

    return g,G

def collision(graph,graph_,Node, R):
    """
    calculate a new node's collison with seed set
    :param g: the graph
    :param A: new seed
    :param R: if this collison should be recorded
    :return: influence
    """

    res = 0.0
    coll = 0.0
    inf = 0.0
    touch = 0.0
    IC_ITERATION = 100

    # if R ==0, record everything on graph_, the backup graph
    if R == 0:
        g = graph_
    else:
        g = graph

    # recorded influence and collision of individual node
    for _ in range(IC_ITERATION):
        total_activated_nodes = set()
        for u in Node:
            l1 = {u}
            l2 = set()
            failed_nodes = set()
            activated_nodes = {u}
            while len(l1):
                for v in l1:
                    for w, weight in g[v].items():
                        r = random()
                        # each node has only one chance to be activated and should only be actived once
                        if w not in failed_nodes and w not in activated_nodes and r < weight:
                            l2.add(w)
                            g[w]["On"] += -1
                            activated_nodes.add(w)
                        else:
                            failed_nodes.add(w)
                l1 = l2
                l2 = set()
            total_activated_nodes.update(activated_nodes)
            res += len(total_activated_nodes)

    # record "On" nodes, the potentially activated nodes
    for _ in range(10):
        total_activated_nodes = set()
        for u in Node:
            l1 = {u}
            l2 = set()
            failed_nodes = set()
            activated_nodes = {u}
            while len(l1):
                for v in l1:
                    for w, weight in g[v].items():
                        r = random()
                        # each node has only one chance to be activated and should only be actived once
                        if w not in failed_nodes and w not in activated_nodes and r < weight:
                            l2.add(w)
                            if g[w]["On"] <= -IC_ITERATION * 1.2:
                                coll+= 1
                            if g[w]["On"] <= -IC_ITERATION * 0.5 and g[w]["On"] >= -IC_ITERATION:
                                inf += 1
                            if g[w]["On"] >= -IC_ITERATION * 0.5 and g[w]["On"] <= 0:
                                touch += 1
                            activated_nodes.add(w)
                        else:
                            failed_nodes.add(w)
                l1 = l2
                l2 = set()
            total_activated_nodes.update(activated_nodes)

    end = time.time()
    #print("time cost on coll: ",end - start)
    return res / IC_ITERATION, (coll) /10, inf/10, touch/10 #to include itself


def IC(g, A, IC_ITERATION = 10):
    """
    calculate seed set's influence in IC（Independent Cascade）model
    :param g: a graph in dict
    :param A: seed nodes in set
    :return: influence
    """
    res = 0.0
    for _ in range(IC_ITERATION):
        total_activated_nodes = set()
        for u in A:
            l1 = {u}
            l2 = set()
            failed_nodes = set()
            total_activated_nodes.update({u})
            while len(l1):
                for v in l1:
                    for w, weight in g[v].items():
                        r = random()
                        if w not in total_activated_nodes and r < weight:
                            l2.add(w)
                            total_activated_nodes.update({w})
                l1 = l2
                l2 = set()
        res += len(total_activated_nodes)
    return res / IC_ITERATION


def R_first_v(G, v):
    inf = 0
    for adj in G:
        temp = set({v})
        s_temp = set({v})
        while s_temp:
            a = set()
            for i in s_temp:
                if not(i in adj):
                    continue
                for j in adj[i]:
                    if not(j in temp):
                        temp.add(j)
                        a.add(j)
            s_temp = a
        inf += len(temp)
    return inf/snap_num

def strength_list(g,G,budget):
    '''
    Input: budget -- size of seed set
    Return: A -- budget*50 nodes, ordered by degree and individual influence attached
    '''
    #max_increment = {i:R_first_v(G,i) for i in g.keys()}
    max_increment = {i:IC(g, {i}) for i in g.keys()}
    t = sorted(max_increment.items(), key=lambda x:x[1], reverse=True)
    A = t[0:budget*20]
    return A;





#test = Env( r"C:\Users\siwei\OneDrive\Paper\DATA\Cit-HepPh.txt",0)
#S =
#print(IC(test.graph,S))
#print(test.list)
#print(test.graph)
#print(CELF.CELF_plus_plus(test.graph, 300))
#test = Env(r"C:\Users\siwei\OneDrive\CNA\大创\complex_network_course-master\Homework_4\DBLP.txt")
#print(CELF.CELF_plus_plus(test.graph, 300))
