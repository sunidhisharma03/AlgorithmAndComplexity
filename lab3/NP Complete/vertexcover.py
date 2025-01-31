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


# Optimized Backtracking solution
def vertex_cover_backtracking(graph):
    n = len(graph)
    min_cover_size = [n]  # Store the minimum vertex cover size globally

    # Sort nodes by degree (helps in pruning)
    degrees = [sum(graph[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])

    # Recursive backtracking function
    def backtrack(subset, index):
        if is_vertex_cover(graph, subset):
            min_cover_size[0] = min(min_cover_size[0], len(subset))
            return

        if index == n:
            return  # Prune search if all nodes are considered

        # Explore adding the next vertex
        backtrack(subset + [sorted_vertices[index]], index + 1)

        # Explore without adding the vertex
        backtrack(subset, index + 1)

    backtrack([], 0)
    return min_cover_size[0]


# Greedy Approximation (Heuristic)
def vertex_cover_greedy(graph):
    n = len(graph)
    cover = set()
    remaining_edges = {(i, j) for i in range(n) for j in range(i + 1, n) if graph[i][j]}

    while remaining_edges:
        # Pick the vertex with the highest degree
        degree_count = {v: sum(graph[v]) for v in range(n) if v not in cover}
        max_degree_vertex = max(degree_count, key=degree_count.get)

        # Add to cover
        cover.add(max_degree_vertex)

        # Remove covered edges
        remaining_edges = {(i, j) for i, j in remaining_edges if i != max_degree_vertex and j != max_degree_vertex}

    return len(cover)


# 2-Approximation Algorithm (Edge Matching)
def vertex_cover_2_approx(graph):
    n = len(graph)
    cover = set()
    remaining_edges = {(i, j) for i in range(n) for j in range(i + 1, n) if graph[i][j]}

    while remaining_edges:
        # Pick an arbitrary edge (u, v)
        u, v = next(iter(remaining_edges))
        cover.add(u)
        cover.add(v)

        # Remove all edges touching u or v
        remaining_edges = {(i, j) for i, j in remaining_edges if i != u and i != v and j != u and j != v}

    return len(cover)


# Compare performance of different algorithms
def plot_vertex_cover_performance():
    num_vertices_list = [5, 10, 15, 18]  # Number of vertices to test
    edge_probability = 0.5  # Probability of edge presence

    brute_times = []
    backtrack_times = []
    greedy_times = []
    approx_2_times = []

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

        # Greedy Approximation
        start_time = time.time()
        min_cover_greedy = vertex_cover_greedy(graph)
        greedy_times.append(time.time() - start_time)

        # 2-Approximation Algorithm
        start_time = time.time()
        min_cover_approx_2 = vertex_cover_2_approx(graph)
        approx_2_times.append(time.time() - start_time)

        print(f"Vertices: {num_vertices}, "
              f"Brute: {min_cover_brute}, "
              f"Backtrack: {min_cover_backtrack}, "
              f"Greedy: {min_cover_greedy}, "
              f"2-Approx: {min_cover_approx_2}")

    # Plotting the time complexity
    plt.figure(figsize=(10, 6))
    plt.plot(num_vertices_list, brute_times, label="Brute Force", marker='o')
    plt.plot(num_vertices_list, backtrack_times, label="Backtracking", marker='o')
    plt.plot(num_vertices_list, greedy_times, label="Greedy Heuristic", marker='o', linestyle="dashed")
    plt.plot(num_vertices_list, approx_2_times, label="2-Approximation", marker='o', linestyle="dotted")

    plt.xlabel("Number of Vertices")
    plt.ylabel("Time (seconds)")
    plt.title("Performance Comparison of Vertex Cover Algorithms")
    plt.grid(True)
    plt.legend()
    plt.show()


# Run the performance comparison
plot_vertex_cover_performance()
