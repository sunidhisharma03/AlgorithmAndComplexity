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
            if i >= len(self.graph):  # Check if there are enough edges
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

# Backtracking Approach for 0/1 Knapsack Problem
def knapsack_backtracking(weights, values, capacity, n):
    def backtrack(index, current_weight, current_value):
        nonlocal max_value
        if current_weight <= capacity and current_value > max_value:
            max_value = current_value
        if index == n or current_weight >= capacity:
            return
        # Include the current item
        backtrack(index + 1, current_weight + weights[index], current_value + values[index])
        # Exclude the current item
        backtrack(index + 1, current_weight, current_value)

    max_value = 0
    backtrack(0, 0, 0)
    return max_value

# Brute Force Approach for 0/1 Knapsack Problem
def knapsack_bruteforce(weights, values, capacity, n):
    max_value = 0
    # Generate all possible subsets
    for i in range(2**n):
        total_weight = 0
        total_value = 0
        for j in range(n):
            if (i >> j) & 1:  # Check if the j-th item is included
                total_weight += weights[j]
                total_value += values[j]
        if total_weight <= capacity and total_value > max_value:
            max_value = total_value
    return max_value

# Function to generate random inputs for Knapsack Problem
def generate_random_knapsack_input(n):
    weights = [random.randint(1, 20) for _ in range(n)]
    values = [random.randint(10, 50) for _ in range(n)]
    capacity = random.randint(20, 50)
    return weights, values, capacity

# Function to measure time for Knapsack algorithms
def measure_knapsack_time(algorithm, weights, values, capacity, n):
    start_time = time.time()
    algorithm(weights, values, capacity, n)
    end_time = time.time()
    return end_time - start_time

# Parameters for the time complexity graph
n_list = range(1, 20)  # Number of items for Knapsack

# Measure Backtracking and Brute Force times
backtracking_times = []
bruteforce_times = []
for n in n_list:
    weights, values, capacity = generate_random_knapsack_input(n)
    # Measure Backtracking time
    backtracking_time = measure_knapsack_time(knapsack_backtracking, weights, values, capacity, n)
    backtracking_times.append(backtracking_time)
    # Measure Brute Force time
    bruteforce_time = measure_knapsack_time(knapsack_bruteforce, weights, values, capacity, n)
    bruteforce_times.append(bruteforce_time)

# Plotting the time complexity graph for Backtracking and Brute Force
plt.figure(figsize=(6, 6))  # Adjusted size for a single plot
plt.plot(n_list, backtracking_times, marker='o', label='Backtracking')
plt.plot(n_list, bruteforce_times, marker='o', label='Brute Force')
plt.xlabel('Number of Items (n)')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity Comparison: Backtracking vs Brute Force')
plt.legend()
plt.grid(True)

plt.show()
