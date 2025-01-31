import time
import random
import networkx as nx
import matplotlib.pyplot as plt

# Class to represent a graph
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    def find(self, parent, i):
        if parent[i] != i:
            parent[i] = self.find(parent, parent[i])
        return parent[i]

    def union(self, parent, rank, x, y):
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x
        else:
            parent[y] = x
            rank[x] += 1

    def KruskalMST(self):
        result = []
        i = 0
        e = 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1:
            if i >= len(self.graph):
                print("Not enough edges to form a spanning tree.")
                return result
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
        return result

# Generate a random graph

def generate_random_graph(vertices, edges):
    g = Graph(vertices)
    for _ in range(edges):
        u, v = random.sample(range(vertices), 2)
        w = random.randint(1, 20)
        g.addEdge(u, v, w)
    return g

# Measure execution time
def measure_time(algorithm, graph):
    start_time = time.time()
    algorithm(graph)
    end_time = time.time()
    return end_time - start_time

# Brute Force MST (Checking all subsets)
def brute_force_mst(graph):
    G = nx.Graph()
    for edge in graph.graph:
        G.add_edge(edge[0], edge[1], weight=edge[2])
    
    min_tree = None
    min_weight = float('inf')
    for edges in nx.connected_components(G):
        subgraph = G.subgraph(edges)
        weight = sum(nx.get_edge_attributes(subgraph, 'weight').values())
        if len(subgraph.edges) == graph.V - 1 and weight < min_weight:
            min_weight = weight
            min_tree = subgraph.edges
    return min_tree

# Backtracking approach to find MST
def backtracking_mst(graph):
    mst = []
    min_weight = [float('inf')]

    def backtrack(index, current_tree, current_weight):
        if len(current_tree) == graph.V - 1:
            if current_weight < min_weight[0]:
                min_weight[0] = current_weight
                mst.clear()
                mst.extend(current_tree)
            return
        if index >= len(graph.graph):
            return
        # Include current edge
        backtrack(index + 1, current_tree + [graph.graph[index]], current_weight + graph.graph[index][2])
        # Exclude current edge
        backtrack(index + 1, current_tree, current_weight)
    
    backtrack(0, [], 0)
    return mst

# Parameters
vertices = 6
edges = 10
n_list = range(1, 10)  # Number of nodes for different graph sizes

# Measure execution time for Kruskal, Brute Force, and Backtracking
kruskal_times = []
bruteforce_times = []
backtracking_times = []

for n in n_list:
    graph = generate_random_graph(n, min(n * 2, (n * (n - 1)) // 2))
    kruskal_times.append(measure_time(lambda g: g.KruskalMST(), graph))
    bruteforce_times.append(measure_time(brute_force_mst, graph))
    backtracking_times.append(measure_time(backtracking_mst, graph))

# Plotting the time complexity graph for Kruskal, Backtracking, and Brute Force
plt.figure(figsize=(6, 6))
plt.plot(n_list, kruskal_times, marker='o', label='Kruskal')
plt.plot(n_list, bruteforce_times, marker='o', label='Brute Force')
plt.plot(n_list, backtracking_times, marker='o', label='Backtracking')
plt.xlabel('Number of Nodes (n)')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity Comparison: Kruskal vs Brute Force vs Backtracking')
plt.legend()
plt.grid(True)
plt.show()