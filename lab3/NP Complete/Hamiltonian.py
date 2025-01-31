import random
import itertools
import time
import matplotlib.pyplot as plt
import sys


# Generate a random adjacency matrix for a graph
def generate_random_graph(num_vertices, edge_probability=0.5):
    graph = [[0] * num_vertices for _ in range(num_vertices)]
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < edge_probability:
                graph[i][j] = graph[j][i] = 1
    return graph


# Check if the current path forms a Hamiltonian cycle
def is_hamiltonian_cycle(graph, path):
    n = len(graph)
    if graph[path[-1]][path[0]] == 0:
        return False  # Last node must connect back to the first node
    visited = set(path)
    if len(visited) != n:
        return False  # All nodes must be visited exactly once
    for i in range(n - 1):
        if graph[path[i]][path[i + 1]] == 0:
            return False  # Adjacent nodes in the path must be connected
    return True


# Brute Force solution to the Hamiltonian Cycle Problem
def hamiltonian_cycle_bruteforce(graph):
    n = len(graph)
    vertices = list(range(n))
    for perm in itertools.permutations(vertices):
        if is_hamiltonian_cycle(graph, perm):
            return perm
    return None


# Backtracking solution to the Hamiltonian Cycle Problem
def hamiltonian_cycle_backtracking(graph):
    n = len(graph)
    path = [-1] * n  # Initialize the path

    def backtrack(v):
        if v == n:
            return is_hamiltonian_cycle(graph, path)

        for vertex in range(1, n):
            if graph[path[v - 1]][vertex] == 1 and vertex not in path[:v]:
                path[v] = vertex
                if backtrack(v + 1):
                    return True
                path[v] = -1  # Backtrack

        return False

    path[0] = 0  # Start from vertex 0
    if backtrack(1):
        return path
    return None


# Dynamic Programming (Held-Karp Algorithm) for Hamiltonian Cycle
def hamiltonian_cycle_dynamic(graph):
    n = len(graph)
    INF = float('inf')

    # DP table: dp[mask][i] = min cost to visit set "mask" ending at vertex i
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start from vertex 0

    # Iterate over all subsets
    for mask in range(1, 1 << n):
        for i in range(n):
            if (mask & (1 << i)) == 0:
                continue
            prev_mask = mask ^ (1 << i)
            for j in range(n):
                if prev_mask & (1 << j) and graph[j][i]:
                    dp[mask][i] = min(dp[mask][i], dp[prev_mask][j] + 1)

    # Find minimum cost cycle
    min_cycle = INF
    for i in range(1, n):
        if graph[i][0]:  # Must connect last node to start node
            min_cycle = min(min_cycle, dp[(1 << n) - 1][i] + 1)

    return min_cycle if min_cycle < INF else None


# Compare performance of Brute Force, Backtracking, and Dynamic Programming
def plot_hamiltonian_cycle_performance():
    num_vertices_list = [5, 6, 7, 8]  # Different sizes of graphs to test
    edge_probability = 0.5  # Probability of edge presence

    brute_times = []
    backtrack_times = []
    dynamic_times = []

    for num_vertices in num_vertices_list:
        graph = generate_random_graph(num_vertices, edge_probability)

        # Brute Force
        start_time = time.time()
        cycle_bruteforce = hamiltonian_cycle_bruteforce(graph)
        brute_times.append(time.time() - start_time)

        # Backtracking
        start_time = time.time()
        cycle_backtrack = hamiltonian_cycle_backtracking(graph)
        backtrack_times.append(time.time() - start_time)

        # Dynamic Programming
        start_time = time.time()
        cycle_dynamic = hamiltonian_cycle_dynamic(graph)
        dynamic_times.append(time.time() - start_time)

        print(f"Vertices: {num_vertices}, Brute Force Time: {brute_times[-1]:.6f}s, Backtracking Time: {backtrack_times[-1]:.6f}s, DP Time: {dynamic_times[-1]:.6f}s")

    # Plotting the time complexity
    plt.figure(figsize=(10, 6))
    plt.plot(num_vertices_list, brute_times, label="Brute Force Time", marker='o')
    plt.plot(num_vertices_list, backtrack_times, label="Backtracking Time", marker='o')
    plt.plot(num_vertices_list, dynamic_times, label="Dynamic Programming Time", marker='o', linestyle="dashed")
    plt.xlabel("Number of Vertices")
    plt.ylabel("Time (seconds)")
    plt.title("Performance Comparison of Hamiltonian Cycle Problem")
    plt.grid(True)
    plt.legend()
    plt.show()


# Run the performance comparison
plot_hamiltonian_cycle_performance()
