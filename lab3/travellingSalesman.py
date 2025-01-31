import itertools
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappop, heappush

def generate_random_distance_matrix(num_cities, min_distance=10, max_distance=100, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    matrix = np.zeros((num_cities, num_cities), dtype=int)
    for i in range(num_cities):
        for j in range(i+1, num_cities):
            distance = random.randint(min_distance, max_distance)
            matrix[i][j] = distance
            matrix[j][i] = distance
    return matrix.tolist()

def calculate_total_distance(route, distance_matrix):
    total_distance = sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    total_distance += distance_matrix[route[-1]][route[0]]
    return total_distance

# 1. Brute Force TSP
def brute_force_tsp(distance_matrix):
    num_cities = len(distance_matrix)
    cities = list(range(num_cities))
    all_routes = itertools.permutations(cities[1:])
    shortest_route, shortest_distance = None, float('inf')
    for route in all_routes:
        current_route = [0] + list(route)
        current_distance = calculate_total_distance(current_route, distance_matrix)
        if current_distance < shortest_distance:
            shortest_route, shortest_distance = current_route, current_distance
    return shortest_route, shortest_distance

# 2. Dynamic Programming TSP (Held-Karp)
def held_karp(distance_matrix):
    num_cities = len(distance_matrix)
    memo = {}
    def dp(mask, pos):
        if mask == (1 << num_cities) - 1:
            return distance_matrix[pos][0]
        if (mask, pos) in memo:
            return memo[(mask, pos)]
        ans = float('inf')
        for city in range(num_cities):
            if not mask & (1 << city):
                ans = min(ans, distance_matrix[pos][city] + dp(mask | (1 << city), city))
        memo[(mask, pos)] = ans
        return ans
    return dp(1, 0)

# 3. Branch and Bound
def branch_and_bound_tsp(distance_matrix):
    num_cities = len(distance_matrix)
    priority_queue = []
    heappush(priority_queue, (0, [0], set([0])))
    best_distance, best_route = float('inf'), None

    while priority_queue:
        cost, path, visited = heappop(priority_queue)
        if len(path) == num_cities:
            total_cost = cost + distance_matrix[path[-1]][path[0]]
            if total_cost < best_distance:
                best_distance, best_route = total_cost, path + [0]
        else:
            for city in range(num_cities):
                if city not in visited:
                    new_cost = cost + distance_matrix[path[-1]][city]
                    if new_cost < best_distance:
                        heappush(priority_queue, (new_cost, path + [city], visited | {city}))
    return best_route, best_distance

# 4. Backtracking
def backtracking_tsp(distance_matrix):
    num_cities = len(distance_matrix)
    best_route, best_distance = None, float('inf')
    
    def backtrack(path, visited, current_cost):
        nonlocal best_route, best_distance
        if len(path) == num_cities:
            total_cost = current_cost + distance_matrix[path[-1]][path[0]]
            if total_cost < best_distance:
                best_distance, best_route = total_cost, path[:]
            return
        for city in range(num_cities):
            if city not in visited:
                new_cost = current_cost + distance_matrix[path[-1]][city]
                if new_cost < best_distance:
                    visited.add(city)
                    path.append(city)
                    backtrack(path, visited, new_cost)
                    path.pop()
                    visited.remove(city)
    
    backtrack([0], set([0]), 0)
    return best_route + [0], best_distance

# Main Function to Compare Approaches and Plot Time Complexity
def main():
    num_cities_list = [5, 6, 7, 8, 9]
    time_results = {"Brute Force": [], "Dynamic Programming": [], "Branch and Bound": [], "Backtracking": []}
    
    for num_cities in num_cities_list:
        distance_matrix = generate_random_distance_matrix(num_cities, seed=42)
        
        for method_name, method in [("Brute Force", brute_force_tsp), 
                                    ("Dynamic Programming", lambda dm: (None, held_karp(dm))),
                                    ("Branch and Bound", branch_and_bound_tsp),
                                    ("Backtracking", backtracking_tsp)]:
            start_time = time.time()
            method(distance_matrix)
            end_time = time.time()
            time_results[method_name].append(end_time - start_time)
    
    # Plot Time Complexity
    for method_name, times in time_results.items():
        plt.plot(num_cities_list, times, label=method_name)
    plt.xlabel("Number of Cities")
    plt.ylabel("Computation Time (seconds)")
    plt.title("Time Complexity of TSP Approaches")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
