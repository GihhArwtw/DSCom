import random
import time
import math
from copy import deepcopy

import networkx as nx
import random
from degreeDiscount import degreeDiscountIC2
from randomHeuristic import randomHeuristic
from func import CELF


class Env:
    def __init__(
    self,
    file,
    ratio = 0.1,
    budget = 50,# -*- coding: utf-8 -*-
    ):
        self.budget = budget
        self.seed = set()
        self.graph ={}
        self.graph = read_from_txt(file)
        self.list = strength_list(self.graph,ratio,self.budget)
        self.maxGain = self.list[0][1]
        self.graph_ = deepcopy(self.graph)
        self.netInput = []
        print(self.graph)

    def list(self):
        return self.list

    """
    Action basing on choice
    :param: step, seed number
    :param: A, Add seed or not
    :param: T, traing or not
    :return: g, a graph in dict
    """
    def steps(self, step, A, T):
        start = time.time()
        node = {self.list[step][0]}
        if T == 1:
            R =  IC(self.graph, self.seed | node) - IC(self.graph, self.seed)
        else:
            R = 0

        if A == 1:
            self.seed = self.seed | node
            self.graph = self.graph_
        print("time cost on step: ",time.time() - start)
        return R

    def reward(self,seed):
        return IC(self.graph,seed)

    def node2feat(self,step):
        node = {self.list[step][0]}
        self.graph_ = deepcopy(self.graph)
        influence,coll = collision(self.graph,self.graph_,node, 0)
        self.netInput = [influence/self.maxGain,coll/(influence-1)]
        return influence,coll

    def update_graph(self):
        self.graph = self.graph_

    def backup_graph(self):
        self.graph_ = deepcopy(self.graph)

def read_from_txt(filename):
    """
    From TXT file read node pairs and assign active probability to edge by random
    :param: filename, TXT file name
    :return: g, a graph in dict
    """
    g = {}
    with open(filename) as f:
        lines = f.readlines()[1:]
        for line in lines:
            e = [int(s) for s in line.replace('\n', '').split(' ')]
            r = 0.99

            if e[0] in g.keys():
                if e[1] in g[e[0]].keys():
                    x = g[e[0]][e[1]]
                    g[e[0]][e[1]] = 1 + (x-1)*r
                else:
                    g[e[0]][e[1]] = 1-r
            else:
                g[e[0]] = {"On": 0}
                g[e[0]][e[1]] = r
            if e[1] in g.keys():
                if e[0] in g[e[1]].keys():
                    x = g[e[1]][e[0]]
                    g[e[1]][e[0]] = 1 + (x-1)*r
                else:
                    g[e[1]][e[0]] = 1-r
            else:
                g[e[1]] = {"On": 0}
                g[e[1]][e[0]] = r
    return g

def collision(graph,graph_,Node, R):
    """
    calculate a new node's collison with seed set
    :param g: the graph
    :param A: new seed
    :param R: if this collison should be recorded
    :return: influence
    """
    start = time.time()
    res = 0.0
    coll = 0.0
    On_node = 0
    IC_ITERATION = 100
    if R == 0:
        g = graph_
    else:
        g = graph


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
                        r = random.random()
                        # each node has only one chance to be activated and should only be actived once
                        if w not in failed_nodes and w not in activated_nodes and r < weight:
                            l2.add(w)
                            if g[w]["On"] == -1:
                                coll += 1
                            activated_nodes.add(w)
                        else:
                            failed_nodes.add(w)
                l1 = l2
                l2 = set()
            total_activated_nodes.update(activated_nodes)
            res += len(total_activated_nodes)

    for _ in range(20):
        total_activated_nodes = set()
        for u in Node:
            l1 = {u}
            l2 = set()
            failed_nodes = set()
            activated_nodes = {u}
            while len(l1):
                for v in l1:
                    for w, weight in g[v].items():
                        r = random.random()
                        # each node has only one chance to be activated and should only be actived once
                        if w not in failed_nodes and w not in activated_nodes and r < weight:
                            l2.add(w)
                            if g[w]["On"] == 0:
                                g[w]["On"] = -1
                            else:
                                On_node += 1
                            activated_nodes.add(w)
                        else:
                            failed_nodes.add(w)
                l1 = l2
                l2 = set()
            total_activated_nodes.update(activated_nodes)

    end = time.time()
    #print("time cost on coll: ",end - start)
    return res / IC_ITERATION, (coll) / IC_ITERATION #to include itself


def IC(g, A):
    """
    calculate a node's influence in IC（Independent Cascade）model
    :param g: a graph in dict
    :param A: seed nodes in set
    :return: influence
    """
    res = 0.0
    IC_ITERATION = 300
    for _ in range(IC_ITERATION):
        total_activated_nodes = set()
        for u in A:
            l1 = {u}
            l2 = set()
            failed_nodes = set()
            activated_nodes = {u}
            while len(l1):
                for v in l1:
                    #print(g[v].items(),flush=True)
                    for w, weight in g[v].items():
                        #print(w,weight,flush=True)
                        r = random.random()
                        # each node has only one chance to be activated and should only be actived once
                        if w not in failed_nodes and w not in activated_nodes and r < weight:
                            l2.add(w)
                            activated_nodes.add(w)
                        else:
                            failed_nodes.add(w)
                l1 = l2
                l2 = set()
            total_activated_nodes.update(activated_nodes)
        res += len(total_activated_nodes)
    return res / IC_ITERATION


def strength_list(g,ratio,budget):
    G = nx.Graph()
    with open(r'C:\Users\siwei\Desktop\Paper\influence-maximization-master\graphdata\phy.txt') as f:
        n, m = f.readline().split()
        for line in f:
            u, v = list(map(int, line.split()))
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u,v, weight=1)
            # G.add_edge(u, v, weight=1)

    S =degreeDiscountIC2(G, budget*50)
    print(IC(g,S[0:budget]))
    t = {i:IC(g, {i}) for i in S}
    A = []
    for key in t:
        A.append([key,t[key]])
    return A;




test = Env(r"C:\Users\siwei\Desktop\Paper\Final Code\DATA\phy.txt",10)
#print(test.list)
#print(CELF.CELF_plus_plus(test.graph, 300))
#test = Env(r"C:\Users\siwei\OneDrive\CNA\大创\complex_network_course-master\Homework_4\DBLP.txt")
#print(CELF.CELF_plus_plus(test.graph, 300))
