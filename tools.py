
def read_diff_model(PATH):
    import numpy as np
    temp = np.load(PATH)
    if temp[0]>0:
        return 'LT-based'
    else:
        return 'IC-based'
    
def read_weights(num_nodes, PATH):
    import numpy as np
    temp = np.loadtxt(PATH,delimiter=' ')
    
    probabilities = [[] for _ in range(num_nodes)]
    adjacency = dict()
    for u in range(num_nodes):
        adjacency[u] = []
    
    for i in range(len(temp)):
        adjacency[int(temp[i][0])].append(int(temp[i][1]))
        probabilities[int(temp[i][0])].append(temp[i][2])
        
    return adjacency, probabilities


def evaluate(adjacency, probabilities, seeds, model_base, random_seed=0, N=2000):
    import random
    import numpy as np
    import queue
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    num_nodes = len(adjacency)
    influence = 0

    for _ in range(N):
        q = queue.Queue()
        for seed in seeds:
            q.put_nowait(seed)
        activated = len(seeds)
        
        if (model_base=='IC-based'):
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
                        activated += 1
                    visited[node] = True
                    
        elif (model_base=='LT_based'):
            thr = np.random.uniform(0,1,size=num_nodes)
            weight = np.zeros(num_nodes,dtype=float)
            visited = np.zeros(num_nodes,dtype=bool)
            for seed in seeds:
                weight[seed] = thr[seed]+1e-6
                visited[seed] = True
            
            while (not q.empty()):
                u = q.get_nowait()
                x = probabilities[u]
                r = random.uniform(0.9,1)
                
                for i in range(len(adjacency[u])):
                    node = adjacency[u][i]
                    weight[node] += r*x[i]
                    if ((not visited[node]) and weight[node]>thr[node]):
                        q.put_nowait(node)  # Activate node.
                        activated += 1
                        
        else:
            raise ValueError("argument \'model_base\' is neither \"LT-based\" nor \"IC-based\".")
            
        influence += activated
        
    influence = influence / float(N)
    return influence


import random
class Graph:
    # Set Default
    nodes = None
    edges = None
    children = None
    parentss = None
    
    def __init__(self, nodes, edges, children, parentss):
        self.n = len(nodes)
        self.nodes = nodes
        self.edges = edges
        self.children = children
        self.parentss = parentss
        self.threshold = {}
        self.accumulate_weight = {}
        
    def get_children(self, node):
        # Get the children set of the node.
        return self.children.get(node, set())
    def get_parentss(self, node):
        # Get the parents set of the node.
        return self.parentss.get(node, set())

def readGraph_direct(path):
    # Initialization
    parentss = {}
    children = {}
    edges = {}
    nodes = set()
    
    # Line by line read in edges
    f = open(path, 'r')
    for line in f.readlines():
        line = line.strip()
        if not len(line) or line.startswith('#'):
            continue
        row = line.split()
        src = int(row[0])
        dst = int(row[1])
        nodes.add(src)
        nodes.add(dst)
        if children.get(src) is None:
            children[src] = set()
        if parentss.get(dst) is None:
            parentss[dst] = set()
        edges[(src, dst)] = float(row[2])
        children[src].add(dst)
        parentss[dst].add(src)
    
    # # Set naive IC edge weight
    # for edge in edges:
    #     dst = edge[1]
    #     edges[edge] = 1 / len(parentss[dst])
    return Graph(nodes, edges, children, parentss)

def readGraph_undirect(path):
    parentss = {}
    children = {}
    edges = {}
    nodes = set()
    f = open(path, 'r')
    for line in f.readlines():
        line = line.strip()
        if not len(line) or line.startswith('#'):
            continue
        row = line.split()
        src = int(row[0])
        dst = int(row[1])
        nodes.add(src)
        nodes.add(dst)
        if children.get(src) is None:
            children[src] = set()
        if children.get(dst) is None:
            children[dst] = set()
        if parentss.get(src) is None:
            parentss[src] = set()
        if parentss.get(dst) is None:
            parentss[dst] = set()
        edges[(src, dst)] = 0
        edges[(dst, src)] = 0
        children[src].add(dst)
        children[dst].add(src)
        parentss[src].add(dst)
        parentss[dst].add(src)
    for edge in edges:
        dst = edge[1]
        edges[edge] = 1 / len(parentss[dst])
    return Graph(nodes, edges, children, parentss)

def isHappened(prob):
    """ Flip a coin with probability prob. """
    if prob == 1:
        return True
    if prob == 0:
        return False
    rand = random.random()
    if rand <= prob:
        return True
    else:
        return False
        
def compute(graph, model, seeds, N=2000):
    """ Estimate the influence spread with Monte-Carlo approach, repetition time N. """
    influence = 0
    import copy
    
    if model=='IC-based':
        
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
            
    elif model=='LT-based':
        
        for i in range(N):
            for node in graph.nodes:
                graph.threshold[node] = random.random()
                graph.accumulate_weight[node] = 0
            newly_activate = []
            newly_activate.extend(seeds)
            activated = copy.deepcopy(seeds)
            while len(newly_activate) != 0:
                current_node = newly_activate.pop(0)
                children = graph.get_children(current_node)
                for child in children:
                    if child not in activated:
                        graph.accumulate_weight[child] += graph.edges[(current_node, child)]
                        if graph.accumulate_weight[child] > graph.threshold[child]:
                            activated.add(child)
                            newly_activate.append(child)
            influence += len(activated)
            
    else:
        
        raise ValueError("model is neither IC-based nor LT-based.")
    
    
    influence = influence/N
    return influence