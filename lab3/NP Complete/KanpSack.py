import random
import itertools
import time
import matplotlib.pyplot as plt

# Brute force 0/1 Knapsack solution
def knapsack_bruteforce(weights, values, W):
    n = len(weights)
    max_value = 0
    
    # Generate all subsets of items
    for subset in itertools.product([0, 1], repeat=n):
        total_weight = sum(subset[i] * weights[i] for i in range(n))
        total_value = sum(subset[i] * values[i] for i in range(n))
        
        # If weight is within the limit, check if value is the maximum
        if total_weight <= W:
            max_value = max(max_value, total_value)
    
    return max_value

# Dynamic programming 0/1 Knapsack solution
def knapsack_dynamic(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(W + 1):
            if weights[i - 1] <= w:
                # Take maximum of not taking or taking the item
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][W]

# Backtracking 0/1 Knapsack solution
def knapsack_backtracking(weights, values, W):
    n = len(weights)
    max_value = [0]  # Use a list to store the max value as reference

    # Recursive function for backtracking
    def backtrack(index, current_weight, current_value):
        # Update maximum value if within bounds
        if current_weight <= W:
            max_value[0] = max(max_value[0], current_value)

        # Termination condition
        if index == n or current_weight > W:
            return

        # Explore including the current item
        backtrack(index + 1, current_weight + weights[index], current_value + values[index])
        # Explore excluding the current item
        backtrack(index + 1, current_weight, current_value)

    # Start backtracking
    backtrack(0, 0, 0)
    return max_value[0]

# Function to compare performance of all approaches
def plot_knapsack_performance():
    n_values = [5, 10, 15, 20]  # Different numbers of items (limited due to brute force)
    W = 50  # Knapsack capacity
    brute_times = []
    dp_times = []
    backtrack_times = []
    
    for n in n_values:
        # Randomly generate weights and values for n items
        weights = [random.randint(1, 10) for _ in range(n)]
        values = [random.randint(1, 20) for _ in range(n)]
        
        # Brute Force
        start_time = time.time()
        max_value_brute = knapsack_bruteforce(weights, values, W)
        brute_times.append(time.time() - start_time)
        
        # Dynamic Programming
        start_time = time.time()
        max_value_dp = knapsack_dynamic(weights, values, W)
        dp_times.append(time.time() - start_time)
        
        # Backtracking
        start_time = time.time()
        max_value_backtrack = knapsack_backtracking(weights, values, W)
        backtrack_times.append(time.time() - start_time)
        
        print(f"n={n}, Max Value (Brute): {max_value_brute}, Max Value (DP): {max_value_dp}, Max Value (Backtrack): {max_value_backtrack}")
    
    # Plotting the time complexity
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, brute_times, label="Brute Force Time", marker='o')
    plt.plot(n_values, dp_times, label="Dynamic Programming Time", marker='o')
    plt.plot(n_values, backtrack_times, label="Backtracking Time", marker='o')
    plt.xlabel("Number of Items (n)")
    plt.ylabel("Time (seconds)")
    plt.title("Performance Comparison of 0/1 Knapsack Problem")
    plt.grid(True)
    plt.legend()
    plt.show()

# Run the performance comparison
plot_knapsack_performance()
