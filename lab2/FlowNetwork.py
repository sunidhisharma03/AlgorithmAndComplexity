import time
import random
import networkx as nx
import matplotlib.pyplot as plt

# Ford-Fulkerson Algorithm using DFS (Backtracking approach)
def ford_fulkerson_backtracking(graph, source, sink):
    def dfs(flow_graph, u, flow, visited):
        if u == sink:
            return flow
        visited.add(u)
        for v in flow_graph[u]:
            if v not in visited and flow_graph[u][v] > 0:  # If capacity available
                min_flow = min(flow, flow_graph[u][v])
                result = dfs(flow_graph, v, min_flow, visited)
                if result > 0:
                    flow_graph[u][v] -= result
                    flow_graph[v][u] += result
                    return result
        return 0

    max_flow = 0
    residual_graph = {u: dict(edges) for u, edges in graph.items()}
    while True:
        visited = set()
        flow = dfs(residual_graph, source, float('inf'), visited)
        if flow == 0:
            break
        max_flow += flow
    return max_flow

# Brute Force approach for Maximum Flow (inefficient)
def max_flow_bruteforce(graph, source, sink):
    nodes = list(graph.keys())
    max_flow = 0
    for _ in range(1000):  # Randomly test paths (not exhaustive, but slow)
        random.shuffle(nodes)
        flow = ford_fulkerson_backtracking(graph, source, sink)
        max_flow = max(max_flow, flow)
    return max_flow

# Function to generate a random flow network
def generate_random_flow_network(n):
    graph = {i: {} for i in range(n)}
    for i in range(n - 1):
        for j in range(i + 1, n):
            if random.random() > 0.5:  # Randomly create edges
                capacity = random.randint(1, 20)
                graph[i][j] = capacity
                graph[j][i] = 0  # Reverse edge with zero initial flow
    return graph

# Function to measure execution time
def measure_time(algorithm, graph, source, sink):
    start_time = time.time()
    algorithm(graph, source, sink)
    end_time = time.time()
    return end_time - start_time

# Parameters for the time complexity graph
n_list = range(5, 20)  # Number of nodes
backtracking_times = []
bruteforce_times = []

for n in n_list:
    graph = generate_random_flow_network(n)
    source, sink = 0, n - 1
    
    # Measure Backtracking time
    backtracking_time = measure_time(ford_fulkerson_backtracking, graph, source, sink)
    backtracking_times.append(backtracking_time)
    
    # Measure Brute Force time
    bruteforce_time = measure_time(max_flow_bruteforce, graph, source, sink)
    bruteforce_times.append(bruteforce_time)

# Plotting the time complexity graph
plt.plot(n_list, backtracking_times, marker='o', label='Ford-Fulkerson (Backtracking)')
plt.plot(n_list, bruteforce_times, marker='o', label='Brute Force')
plt.xlabel('Number of Nodes (n)')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity Comparison: Ford-Fulkerson vs Brute Force')
plt.legend()
plt.grid(True)
plt.show()
