import random
import time
import matplotlib.pyplot as plt

# Miller-Rabin Primality Test
def is_prime_miller_rabin(n, k):
    if n <= 1:
        return "composite"
    if n <= 3:
        return "probably prime"
    if n % 2 == 0:
        return "composite"
    
    s, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        s += 1

    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return "composite"
    
    return "probably prime"

# Fermat Primality Test
def is_prime_fermat(n, k):
    if n <= 1:
        return "composite"
    if n <= 3:
        return "probably prime"
    
    for _ in range(k):
        a = random.randint(2, n - 2)
        if pow(a, n - 1, n) != 1:
            return "composite"
    
    return "probably prime"

# Function to measure execution time
def measure_time(prime_test_func, n, k):
    start_time = time.time()
    prime_test_func(n, k)
    end_time = time.time()
    return end_time - start_time

# Main execution for comparison
n_values = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]  # Sample prime numbers
k = 5  # Number of iterations
miller_rabin_times = []
fermat_times = []

for n in n_values:
    print(f"Testing n = {n}")
    miller_rabin_times.append(measure_time(is_prime_miller_rabin, n, k))
    fermat_times.append(measure_time(is_prime_fermat, n, k))

# Plotting the comparison
plt.figure(figsize=(10, 6))
plt.plot(n_values, miller_rabin_times, label="Miller-Rabin", marker='o')
plt.plot(n_values, fermat_times, label="Fermat", marker='o')
plt.title("Execution Time Comparison: Miller-Rabin vs Fermat Primality Test")
plt.xlabel("n (Tested Number)")
plt.ylabel("Execution Time (seconds)")
plt.legend()
plt.grid(True)
plt.show()
