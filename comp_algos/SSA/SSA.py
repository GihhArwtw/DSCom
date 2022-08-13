import random
import copy
import math

from tools import readGraph_direct, readGraph_undirect, isHappened

def reverseSearch(graph, root):
    """ Generate a RR-set for node *root* with BFS realized with queue structure. """
    
    # Initialization
    searchSet = set() # Set containing the nodes already searched once to avoid revisit the same node multiple times
    queue = []
    queue.append(root)
    searchSet.add(root)
    
    # BFS
    while len(queue) != 0:
        current_node = queue.pop(0) # Pop out the first element in the list
        parentss = graph.get_parentss(current_node)
        for parent in parentss:
            if parent not in searchSet:
                rate = graph.edges[(parent, current_node)] # Propagation probability in IC model
                if isHappened(rate):
                    searchSet.add(parent)
                    queue.append(parent)
    return searchSet


def generateRRset(graph):
    """ Generate one random RR-set. """
    # Uniformly sample one node from the node set
    selectedNode = random.randint(0,len(graph.nodes)-1)
    
    # Sample a RR-set for selected node
    RRset = reverseSearch(graph, selectedNode)
    return RRset


def RRCollection(graph, theta):
    """ Sample theta times random RR-sets. """
    R = []
    for i in range(theta):
        RR = generateRRset(graph)
        R.append(RR)
    return R


def estimate(graph, seeds, R):
    """ 
    Estimate the influence of a seed set with generated RR-sets, R.
    Return fraction of RR-sets covered by node set seeds.
    """
    counter = 0
    for RR in R:
        if len(seeds & RR) != 0:
            counter += 1
    influence = counter / len(R)
    return influence


def compute(graph, seeds, N=2000):
    """ Estimate the influence spread with Monte-Carlo approach, repetition time N. """
    influence = 0
    for i in range(N):
        queue = []
        queue.extend(seeds)
        checked = copy.deepcopy(seeds)
        while len(queue) != 0:
            current_node = queue.pop(0)
            children = graph.get_children(current_node)
            for child in children:
                if child not in checked:
                    rate = graph.edges[(current_node, child)]
                    if isHappened(rate):
                        checked.add(child)
                        queue.append(child)
        influence += len(checked)
    influence = influence/N
    return influence


def GreedyCoverage(R, k):
    # Initialization
    seeds = set()
    count_dict = {}  # Record the number of uncovered RR-sets will be covered by each node 
    covered = set()  # Record the RR-sets already covered by seed set
    for RR in R:
        for node in RR:
            if count_dict.get(node) is None:
                count_dict[node] = 1
            else:
                count_dict[node] += 1
    
    cover = 0
    
    # Select k nodes with greedy approach
    for i in range(k):
        seed = max(count_dict, key=count_dict.get) # Select the node covers most RR-sets
        cover += count_dict[seed]
        if (count_dict[seed] == 0):  # If all RR-sets have already been covered
            seeds = seeds | set(random.sample(list(set(count_dict.keys()) - seeds), k=k-i))
            break
        seeds.add(seed)
        for j in range(len(R)):  # Update the count_dict and covered after selecting the node
            if (j not in covered) and (seed in R[j]):
                covered.add(j)
                for node in R[j]:
                    count_dict[node] -= 1
                    
    return seeds, cover


"""
==================================
            SSA-fix
==================================
"""

from math import factorial as fac

def GAMMA(epsilon, delta):
    return ( 2. + 2./3.*epsilon ) * math.log( 1./delta) * 1./(epsilon**2)


def estimate_inf(g, seeds, eps, delta, T_max):
    lambda_2 = 1. + (1.+eps)*GAMMA(eps,delta)
    cov = 0
    seeds = set(seeds)
    for t in range(int(T_max)):
        R_j = generateRRset(g)
        cov += min(1, len(R_j & seeds))
        if (cov >= lambda_2):
            return g.n*lambda_2/float(t)
    return -1
        
    
def SSA(graph, num_seeds, EPSILON=0.1, DELTA=0.25):
    # SSA-fix from Revisiting of "Revisiting the Stop-and-Stare Algorithms for Influence Maximization"
    
    from numpy import e
    eps_2 = 1. / (1.-1./e) * EPSILON / 2.
    eps_3 = eps_2
    eps_1 = (1. + 1./(1-1/e-EPSILON) * EPSILON/2. ) / ( 1. + eps_2 ) - 1.
    
    beta = DELTA / 6. * 1. / ( fac(graph.n)/(fac(graph.n-num_seeds)*fac(num_seeds)) )
    N_max = 8. * (1.-1./e) / (2.+2*EPSILON/3.) * GAMMA(EPSILON,beta) * graph.n / num_seeds
    
    i_max = math.ceil( math.log(2.*N_max/GAMMA(eps_3,DELTA/3.)) / math.log(2) )
    lambda_1 = (1+eps_1) * (1+eps_2) * GAMMA(eps_3,DELTA/(3.*i_max))
    
    # Sampling RR-sets
    R = RRCollection(graph, int(lambda_1))
    size = int(lambda_1)
    
    # Greedy algorithm
    while (size<N_max or size<=int(lambda_1)):
        
        R = R + RRCollection(graph, size)
        size = len(R)
        seeds, cover = GreedyCoverage(R,num_seeds)
        inf_hat = cover / size
        
        if (cover >= lambda_1):
            delta_2 = math.exp( -float(cover-1)*(eps_2**2)/(2.*(1+eps_2)) )
            T_max = 2. * len(R) * (1.+eps_2) / (1.-eps_2) * (eps_3**2) / (eps_2**2)
            inf_c = estimate_inf(graph, seeds, eps_2, delta_2, T_max)
            if (inf_hat <= (1+eps_1)*inf_c):
                return seeds
    
    return seeds



if __name__ == '__main__':
    
    from config import GRAPH_PATH, EPSILON, k, DELTA
    
    # Read in graph
    graph = readGraph_direct(GRAPH_PATH)
    
    seeds = SSA(graph, k, EPSILON, DELTA)
    print(seeds)
    
    for i in range(5):
        influence = compute(graph, seeds)
        print(influence)
    

    # R = RRCollection(graph, theta=20000)
    # influence = estimate(graph, seeds, R)
