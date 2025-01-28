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

    # A utility function to find the vertex with minimum key value
    def minKey(self, key, mstSet):
        min_val = sys.maxsize
        min_index = -1

        for v in range(self.V):
            if key[v] < min_val and not mstSet[v]:
                min_val = key[v]
                min_index = v

        return min_index

    # Prim's Algorithm to find MST
    def primMST(self):
        key = [sys.maxsize] * self.V
        parent = [None] * self.V  # Array to store constructed MST
        key[0] = 0  # Start with the first vertex
        mstSet = [False] * self.V
        parent[0] = -1  # Root node of MST

        for _ in range(self.V):
            u = self.minKey(key, mstSet)
            mstSet[u] = True

            for v in range(self.V):
                if (
                    self.graph[u][v] > 0
                    and not mstSet[v]
                    and key[v] > self.graph[u][v]
                ):
                    key[v] = self.graph[u][v]
                    parent[v] = u

        return parent

    # Brute Force to calculate the MST by evaluating all spanning trees
    def bruteForceMST(self):
        edges = []
        for i in range(self.V):
            for j in range(i + 1, self.V):
                if self.graph[i][j] > 0:
                    edges.append((i, j, self.graph[i][j]))

        min_weight = sys.maxsize
        best_tree = None
        for edge_comb in itertools.combinations(edges, self.V - 1):
            adj_matrix = np.zeros((self.V, self.V))
            for u, v, weight in edge_comb:
                adj_matrix[u][v] = weight
                adj_matrix[v][u] = weight

            if self.isConnected(adj_matrix):
                weight_sum = sum(weight for u, v, weight in edge_comb)
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


# Generate a random graph with vertices and edges
def generateGraph(num_vertices, max_weight=10):
    g = Graph(num_vertices)
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < 0.5:  # Randomly add an edge
                weight = random.randint(1, max_weight)
                g.graph[i][j] = weight
                g.graph[j][i] = weight
    return g

# Measure execution time for both approaches (Prim's vs Brute Force)
def measureExecutionTimes(max_num_vertices, step, runs=10):
    vertex_counts = []
    prim_times = []
    brute_force_times = []

    for num_vertices in range(3, max_num_vertices + 1, step):
        prim_total_time = 0
        brute_force_total_time = 0

        for _ in range(runs):
            g = generateGraph(num_vertices)

            # Measure time for Prim's Algorithm
            start_time = time.time()
            g.primMST()
            end_time = time.time()
            prim_total_time += (end_time - start_time)

            # Measure time for Brute Force MST
            start_time = time.time()
            g.bruteForceMST()
            end_time = time.time()
            brute_force_total_time += (end_time - start_time)

        # Average time over all runs
        prim_times.append(prim_total_time / runs)
        brute_force_times.append(brute_force_total_time / runs)
        vertex_counts.append(num_vertices)

    return vertex_counts, prim_times, brute_force_times

# Plot the execution times
def plotExecutionTimes(vertex_counts, prim_times, brute_force_times):
    plt.figure(figsize=(10, 6))
    plt.plot(vertex_counts, prim_times, marker='o', label="Prim's Algorithm", color='blue')
    plt.plot(vertex_counts, brute_force_times, marker='o', label="Brute Force MST", color='red')
    plt.title("Execution Time: Prim's vs Brute Force MST")
    plt.xlabel("Number of Vertices")
    plt.ylabel("Average Execution Time (seconds)")
    plt.grid(True)
    plt.legend()
    plt.show()

# Driver code
if __name__ == "__main__":
    max_num_vertices = 20  # Maximum number of vertices to test
    step = 2               # Step size for increasing number of vertices
    runs = 5               # Number of runs for averaging the execution time

    # Measure execution times for both approaches
    vertex_counts, prim_times, brute_force_times = measureExecutionTimes(max_num_vertices, step, runs)

    # Plot the results
    plotExecutionTimes(vertex_counts, prim_times, brute_force_times)
