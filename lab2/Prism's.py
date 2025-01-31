import random
import time
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt

# Class to represent a graph
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]

    def minKey(self, key, mstSet):
        min_val = sys.maxsize
        min_index = -1
        for v in range(self.V):
            if key[v] < min_val and not mstSet[v]:
                min_val = key[v]
                min_index = v
        return min_index

    def primMST(self):
        key = [sys.maxsize] * self.V
        parent = [None] * self.V  # Array to store MST
        key[0] = 0  # Start with the first vertex
        mstSet = [False] * self.V
        parent[0] = -1  # Root node

        for _ in range(self.V):
            u = self.minKey(key, mstSet)
            mstSet[u] = True

            for v in range(self.V):
                if self.graph[u][v] > 0 and not mstSet[v] and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u
        return parent

    def bruteForceMST(self):
        edges = [(i, j, self.graph[i][j]) for i in range(self.V) for j in range(i + 1, self.V) if self.graph[i][j] > 0]
        min_weight = sys.maxsize
        best_tree = None
        for edge_comb in itertools.combinations(edges, self.V - 1):
            adj_matrix = np.zeros((self.V, self.V))
            for u, v, weight in edge_comb:
                adj_matrix[u][v] = adj_matrix[v][u] = weight
            if self.isConnected(adj_matrix):
                weight_sum = sum(weight for _, _, weight in edge_comb)
                if weight_sum < min_weight:
                    min_weight = weight_sum
                    best_tree = edge_comb
        return best_tree

    def isConnected(self, adj_matrix):
        visited = [False] * self.V
        self.dfs(0, adj_matrix, visited)
        return all(visited)

    def dfs(self, node, adj_matrix, visited):
        visited[node] = True
        for i in range(self.V):
            if adj_matrix[node][i] > 0 and not visited[i]:
                self.dfs(i, adj_matrix, visited)

    def backtrackingMST(self):
        edges = [(i, j, self.graph[i][j]) for i in range(self.V) for j in range(i + 1, self.V) if self.graph[i][j] > 0]
        min_weight = sys.maxsize
        best_tree = None
        
        def backtrack(start, edge_list, total_weight):
            nonlocal min_weight, best_tree
            if len(edge_list) == self.V - 1:
                if self.isConnectedMatrix(edge_list) and total_weight < min_weight:
                    min_weight = total_weight
                    best_tree = edge_list[:]
                return
            for i in range(start, len(edges)):
                edge_list.append(edges[i])
                backtrack(i + 1, edge_list, total_weight + edges[i][2])
                edge_list.pop()
        
        backtrack(0, [], 0)
        return best_tree
    
    def isConnectedMatrix(self, edge_list):
        adj_matrix = np.zeros((self.V, self.V))
        for u, v, weight in edge_list:
            adj_matrix[u][v] = adj_matrix[v][u] = weight
        return self.isConnected(adj_matrix)

# Generate a random graph
def generateGraph(num_vertices, max_weight=10):
    g = Graph(num_vertices)
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < 0.5:
                weight = random.randint(1, max_weight)
                g.graph[i][j] = g.graph[j][i] = weight
    return g

# Measure execution times
def measureExecutionTimes(max_num_vertices, step, runs=3):
    vertex_counts, prim_times, brute_force_times, backtracking_times = [], [], [], []
    for num_vertices in range(3, max_num_vertices + 1, step):
        prim_total_time = brute_force_total_time = backtracking_total_time = 0
        for _ in range(runs):
            g = generateGraph(num_vertices)
            start_time = time.time()
            g.primMST()
            prim_total_time += (time.time() - start_time)
            if num_vertices <= 8:  # Limit brute-force and backtracking to small graphs
                start_time = time.time()
                g.bruteForceMST()
                brute_force_total_time += (time.time() - start_time)
                start_time = time.time()
                g.backtrackingMST()
                backtracking_total_time += (time.time() - start_time)
        vertex_counts.append(num_vertices)
        prim_times.append(prim_total_time / runs)
        brute_force_times.append(brute_force_total_time / runs if num_vertices <= 8 else 0)
        backtracking_times.append(backtracking_total_time / runs if num_vertices <= 8 else 0)
    return vertex_counts, prim_times, brute_force_times, backtracking_times

# Plot execution times
def plotExecutionTimes(vertex_counts, prim_times, brute_force_times, backtracking_times):
    plt.figure(figsize=(10, 6))
    plt.plot(vertex_counts, prim_times, marker='o', label="Prim's Algorithm", color='blue')
    plt.plot(vertex_counts, brute_force_times, marker='o', label="Brute Force MST", color='red')
    plt.plot(vertex_counts, backtracking_times, marker='o', label="Backtracking MST", color='green')
    plt.title("Execution Time: Prim's Algorithm vs Brute Force vs Backtracking MST")
    plt.xlabel("Number of Vertices")
    plt.ylabel("Average Execution Time (seconds)")
    plt.grid(True)
    plt.legend()
    plt.show()

# Driver code
if __name__ == "__main__":
    max_num_vertices = 10  # Reduce max vertices to avoid long execution time
    step = 1  # Step size
    runs = 3  # Number of runs for averaging execution time
    vertex_counts, prim_times, brute_force_times, backtracking_times = measureExecutionTimes(max_num_vertices, step, runs)
    plotExecutionTimes(vertex_counts, prim_times, brute_force_times, backtracking_times)