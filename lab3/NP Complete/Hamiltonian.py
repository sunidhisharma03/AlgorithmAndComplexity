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


# Check if the current path forms a Hamiltonian cycle
def is_hamiltonian_cycle(graph, path):
    n = len(graph)
    # Check if the last vertex is connected to the first vertex
    if graph[path[-1]][path[0]] == 0:
        return False
    
    # Check if all vertices are visited exactly once
    visited = set(path)
    if len(visited) != n:
        return False
    
    # Check if every adjacent pair of vertices is connected
    for i in range(n - 1):
        if graph[path[i]][path[i + 1]] == 0:
            return False
    
    return True


# Brute Force solution to the Hamiltonian Cycle Problem
def hamiltonian_cycle_bruteforce(graph):
    n = len(graph)
    vertices = list(range(n))
    # Generate all permutations of the vertices
    for perm in itertools.permutations(vertices):
        if is_hamiltonian_cycle(graph, perm):
            return perm
    return None


# Backtracking solution to the Hamiltonian Cycle Problem
def hamiltonian_cycle_backtracking(graph):
    n = len(graph)
    path = [-1] * n  # Initialize the path

    # Recursive function for backtracking
    def backtrack(v):
        if v == n:
            return is_hamiltonian_cycle(graph, path)

        for vertex in range(1, n):
            # Check if vertex can be added to the current path
            if graph[path[v - 1]][vertex] == 1 and vertex not in path[:v]:
                path[v] = vertex
                if backtrack(v + 1):
                    return True
                path[v] = -1  # Backtrack
                
        return False

    # Start the search from the first vertex
    path[0] = 0
    if backtrack(1):
        return path
    return None


# Compare performance of Brute Force and Backtracking approaches
def plot_hamiltonian_cycle_performance():
    num_vertices_list = [5, 6, 7, 8]  # Different sizes of graphs to test
    edge_probability = 0.5  # Probability of edge presence

    brute_times = []
    backtrack_times = []
    
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

        print(f"Vertices: {num_vertices}, Brute Force Time: {brute_times[-1]:.6f}s, Backtracking Time: {backtrack_times[-1]:.6f}s")
    
    # Plotting the time complexity
    plt.figure(figsize=(10, 6))
    plt.plot(num_vertices_list, brute_times, label="Brute Force Time", marker='o')
    plt.plot(num_vertices_list, backtrack_times, label="Backtracking Time", marker='o')
    plt.xlabel("Number of Vertices")
    plt.ylabel("Time (seconds)")
    plt.title("Performance Comparison of Hamiltonian Cycle Problem")
    plt.grid(True)
    plt.legend()
    plt.show()


# Run the performance comparison
plot_hamiltonian_cycle_performance()
