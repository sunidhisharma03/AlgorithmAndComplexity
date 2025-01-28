import random
import itertools
import time
import matplotlib.pyplot as plt


# Generate a random adjacency matrix for a graph
def generate_random_graph(num_vertices, edge_probability=0.5):
    graph = [[0] * num_vertices for _ in range(num_vertices)]
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < edge_probability:
                graph[i][j] = graph[j][i] = 1
    return graph


# Check if a subset of vertices forms a clique
def is_clique(graph, subset):
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            if graph[subset[i]][subset[j]] == 0:
                return False
    return True


# Brute Force solution to the Clique Problem
def clique_bruteforce(graph):
    n = len(graph)
    max_clique_size = 0

    # Check all subsets of vertices
    for subset in itertools.chain.from_iterable(itertools.combinations(range(n), r) for r in range(1, n + 1)):
        if is_clique(graph, subset):
            max_clique_size = max(max_clique_size, len(subset))

    return max_clique_size


# Backtracking solution to the Clique Problem
def clique_backtracking(graph):
    n = len(graph)
    max_clique_size = [0]  # Store the maximum clique size globally

    # Recursive backtracking function
    def backtrack(subset, index):
        if is_clique(graph, subset):
            max_clique_size[0] = max(max_clique_size[0], len(subset))
        else:
            return  # Prune the branch if it is not a clique

        # Explore further by adding the next vertices
        for next_vertex in range(index, n):
            backtrack(subset + [next_vertex], next_vertex + 1)

    backtrack([], 0)
    return max_clique_size[0]


# Compare performance of different algorithms
def plot_clique_performance():
    num_vertices_list = [5, 10, 15, 18]  # Number of vertices to test
    edge_probability = 0.5  # Probability of edge presence

    brute_times = []
    backtrack_times = []

    for num_vertices in num_vertices_list:
        graph = generate_random_graph(num_vertices, edge_probability)

        # Brute Force
        start_time = time.time()
        max_clique_brute = clique_bruteforce(graph)
        brute_times.append(time.time() - start_time)

        # Backtracking
        start_time = time.time()
        max_clique_backtrack = clique_backtracking(graph)
        backtrack_times.append(time.time() - start_time)

        print(f"Vertices: {num_vertices}, Max Clique (Brute): {max_clique_brute}, Max Clique (Backtrack): {max_clique_backtrack}")

    # Plotting the time complexity
    plt.figure(figsize=(10, 6))
    plt.plot(num_vertices_list, brute_times, label="Brute Force Time", marker='o')
    plt.plot(num_vertices_list, backtrack_times, label="Backtracking Time", marker='o')
    plt.xlabel("Number of Vertices")
    plt.ylabel("Time (seconds)")
    plt.title("Performance Comparison of Clique Problem")
    plt.grid(True)
    plt.legend()
    plt.show()


# Run the performance comparison
plot_clique_performance()
