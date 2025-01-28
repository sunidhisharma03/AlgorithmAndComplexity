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


# Check if a given set of vertices is a vertex cover
def is_vertex_cover(graph, vertices_cover):
    for i in range(len(graph)):
        for j in range(i + 1, len(graph)):
            if graph[i][j] == 1 and (i not in vertices_cover and j not in vertices_cover):
                return False
    return True


# Brute Force solution to the Vertex Cover Problem
def vertex_cover_bruteforce(graph):
    n = len(graph)
    min_cover_size = n
    
    # Check all subsets of vertices
    for subset in itertools.chain.from_iterable(itertools.combinations(range(n), r) for r in range(1, n + 1)):
        if is_vertex_cover(graph, subset):
            min_cover_size = min(min_cover_size, len(subset))

    return min_cover_size


# Backtracking solution to the Vertex Cover Problem
def vertex_cover_backtracking(graph):
    n = len(graph)
    min_cover_size = [n]  # Store the minimum vertex cover size globally

    # Recursive backtracking function
    def backtrack(subset, index):
        if is_vertex_cover(graph, subset):
            min_cover_size[0] = min(min_cover_size[0], len(subset))
        else:
            return  # Prune the branch if it does not cover all edges

        # Explore further by adding the next vertex
        for next_vertex in range(index, n):
            backtrack(subset + [next_vertex], next_vertex + 1)

    backtrack([], 0)
    return min_cover_size[0]


# Compare performance of different algorithms
def plot_vertex_cover_performance():
    num_vertices_list = [5, 10, 15, 18]  # Number of vertices to test
    edge_probability = 0.5  # Probability of edge presence

    brute_times = []
    backtrack_times = []

    for num_vertices in num_vertices_list:
        graph = generate_random_graph(num_vertices, edge_probability)

        # Brute Force
        start_time = time.time()
        min_cover_brute = vertex_cover_bruteforce(graph)
        brute_times.append(time.time() - start_time)

        # Backtracking
        start_time = time.time()
        min_cover_backtrack = vertex_cover_backtracking(graph)
        backtrack_times.append(time.time() - start_time)

        print(f"Vertices: {num_vertices}, Vertex Cover (Brute): {min_cover_brute}, Vertex Cover (Backtrack): {min_cover_backtrack}")

    # Plotting the time complexity
    plt.figure(figsize=(10, 6))
    plt.plot(num_vertices_list, brute_times, label="Brute Force Time", marker='o')
    plt.plot(num_vertices_list, backtrack_times, label="Backtracking Time", marker='o')
    plt.xlabel("Number of Vertices")
    plt.ylabel("Time (seconds)")
    plt.title("Performance Comparison of Vertex Cover Problem")
    plt.grid(True)
    plt.legend()
    plt.show()


# Run the performance comparison
plot_vertex_cover_performance()
