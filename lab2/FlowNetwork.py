import time
import random
import matplotlib.pyplot as plt

# Backtracking Approach
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

# Brute Force Approach
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

# Function to generate random inputs
def generate_random_input(n):
    weights = [random.randint(1, 20) for _ in range(n)]
    values = [random.randint(10, 50) for _ in range(n)]
    capacity = random.randint(20, 50)
    return weights, values, capacity

# Function to measure execution time
def measure_time(algorithm, weights, values, capacity, n):
    start_time = time.time()
    algorithm(weights, values, capacity, n)
    end_time = time.time()
    return end_time - start_time

# Parameters for the time complexity graph
n_list = range(1, 20)  # Number of items
backtracking_times = []
bruteforce_times = []

for n in n_list:
    weights, values, capacity = generate_random_input(n)
    # Measure Backtracking time
    backtracking_time = measure_time(knapsack_backtracking, weights, values, capacity, n)
    backtracking_times.append(backtracking_time)
    # Measure Brute Force time
    bruteforce_time = measure_time(knapsack_bruteforce, weights, values, capacity, n)
    bruteforce_times.append(bruteforce_time)

# Plotting the time complexity graph
plt.plot(n_list, backtracking_times, marker='o', label='Backtracking')
plt.plot(n_list, bruteforce_times, marker='o', label='Brute Force')
plt.xlabel('Number of Items (n)')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity Comparison: Backtracking vs Brute Force')
plt.legend()
plt.grid(True)
plt.show()