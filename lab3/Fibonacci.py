import time
import matplotlib.pyplot as plt

# Sequential Fibonacci generator
def generate_fibonacci_sequential(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fibonacci_series = [0, 1]
    for _ in range(2, n):
        next_term = fibonacci_series[-1] + fibonacci_series[-2]
        fibonacci_series.append(next_term)
    
    return fibonacci_series

# Memoization Fibonacci generator
def fibonacci_memoization(n, memo={}):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    if n not in memo:
        memo[n] = fibonacci_memoization(n - 1, memo) + fibonacci_memoization(n - 2, memo)
    
    return memo[n]

def generate_fibonacci_memoization(n):
    return [fibonacci_memoization(i) for i in range(n)]

# Measure execution time for both methods
def measure_execution_time():
    sizes = [10000, 50000, 80000, 90000]
    sequential_times = []
    memoization_times = []

    for size in sizes:
        # Sequential method
        start_time = time.time()
        generate_fibonacci_sequential(size)
        sequential_times.append(time.time() - start_time)

        # Memoization method
        start_time = time.time()
        generate_fibonacci_memoization(size)
        memoization_times.append(time.time() - start_time)

    return sizes, sequential_times, memoization_times

# Plot the comparison
def plot_comparison(sizes, sequential_times, memoization_times):
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, sequential_times, label="Sequential", marker='o')
    plt.plot(sizes, memoization_times, label="Memoization", marker='o')
    plt.title("Execution Time Comparison: Sequential vs Memoization")
    plt.xlabel("Number of Terms")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
sizes, sequential_times, memoization_times = measure_execution_time()
plot_comparison(sizes, sequential_times, memoization_times)