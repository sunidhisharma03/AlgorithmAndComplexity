import random
import time
import matplotlib.pyplot as plt

# Function to compute the prefix sum array
def compute_prefix_sum(arr):
    n = len(arr)
    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + arr[i - 1]
    return prefix_sum

# Function to get the sum of elements from index l to r using prefix sum array
def get_range_sum(prefix_sum, l, r):
    return prefix_sum[r + 1] - prefix_sum[l]

# Function to compare performance of Prefix Sum and brute force sum computation
def plot_prefix_sum_performance():
    sizes = [100, 1000, 5000, 10000, 20000]  # Different sizes of the array
    brute_times = []
    prefix_times = []
    
    for size in sizes:
        # Generate a random array
        arr = [random.randint(1, 100) for _ in range(size)]
        
        # Brute Force
        start_time = time.time()
        total = sum(arr)  # Sum the entire array
        brute_times.append(time.time() - start_time)
        
        # Prefix Sum
        start_time = time.time()
        prefix_sum = compute_prefix_sum(arr)  # Precompute prefix sums
        range_sum = get_range_sum(prefix_sum, 0, size - 1)  # Query the entire array sum
        prefix_times.append(time.time() - start_time)

        print(f"Size: {size}, Brute Time: {brute_times[-1]:.6f} s, Prefix Time: {prefix_times[-1]:.6f} s")

    # Plotting the time complexity
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, brute_times, label="Brute Force Time", marker='o')
    plt.plot(sizes, prefix_times, label="Prefix Sum Time", marker='o')
    plt.xlabel("Array Size")
    plt.ylabel("Time (seconds)")
    plt.title("Performance Comparison of Prefix Sum and Brute Force")
    plt.grid(True)
    plt.legend()
    plt.show()

# Run the performance comparison
plot_prefix_sum_performance()
