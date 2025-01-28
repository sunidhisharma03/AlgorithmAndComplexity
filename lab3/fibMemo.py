import time
import sys
import matplotlib.pyplot as plt

# Increase the recursion depth limit
sys.setrecursionlimit(20000)  # Set to a higher value for deep recursion

# Default Recursive Fibonacci function
def fibonacci_recursive(n):
    if n < 0:
        return None
    if n < 2:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

# Memoized Fibonacci function to calculate the nth Fibonacci number
def memoized_fib(n, memo={}):
    # Base case: Fibonacci for 0 and 1 is 1
    if n == 0 or n == 1:
        return 1
    
    # Check if the result is already computed in the memoization dictionary
    if n in memo:
        return memo[n]
    
    # Compute Fibonacci recursively and store in the memo dictionary
    memo[n] = memoized_fib(n-1, memo) + memoized_fib(n-2, memo)
    
    return memo[n]

# Function to time the execution of Fibonacci functions
def time_execution(fib_func, n, iterations=5):
    total_time = 0
    for _ in range(iterations):
        start_time = time.time()
        fib_func(n)
        end_time = time.time()
        total_time += (end_time - start_time)
    return total_time / iterations  # Average time over specified iterations

# List of `n` values from 0 to 100,000, with a step of 5000 for faster plotting
n_values = list(range(0, 38, 1))  # Reduced range to avoid too long runtime

# Lists to store average execution times for recursive and memoized Fibonacci
execution_times_recursive = []
execution_times_memoized = []

# Measure execution times for both Fibonacci functions
for n in n_values:
    # Print the current value of `n` to track the progress
    print(f"Processing n = {n}")
    
    avg_time_recursive = time_execution(fibonacci_recursive, n)
    avg_time_memoized = time_execution(memoized_fib, n)
    
    execution_times_recursive.append(avg_time_recursive)
    execution_times_memoized.append(avg_time_memoized)

# Plotting the graph for both Fibonacci functions
plt.plot(n_values, execution_times_recursive, label="Recursive Fibonacci", color="red")
plt.plot(n_values, execution_times_memoized, label="Memoized Fibonacci", color="blue")
plt.xlabel('n (Fibonacci Index)')
plt.ylabel('Average Time (seconds)')
plt.title('Execution Time of Fibonacci Calculation (Recursive vs Memoized)')
plt.grid(True)
plt.legend()
plt.show()
