import time
import itertools
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt  # For plotting

class NQueensVisualizerBacktracking:
    def __init__(self, n):
        self.n = n
        self.solutions = []
        self.solve()

    def solve(self):
        """Find all solutions using backtracking."""
        def is_safe(board, row, col):
            for i in range(row):
                if board[i] == col or abs(board[i] - col) == row - i:
                    return False
            return True

        def backtrack(board, row):
            if row == self.n:
                self.solutions.append(board[:])
                return
            for col in range(self.n):
                if is_safe(board, row, col):
                    board[row] = col
                    backtrack(board, row + 1)
                    board[row] = -1

        board = [-1] * self.n
        backtrack(board, 0)


class NQueensVisualizerBruteForce:
    def __init__(self, n):
        self.n = n
        self.solutions = []
        self.solve()

    def solve(self):
        """Find all solutions using brute force."""
        all_permutations = itertools.permutations(range(self.n))
        for perm in all_permutations:
            if self.is_valid(perm):
                self.solutions.append(list(perm))

    def is_valid(self, perm):
        """Check if the permutation represents a valid N-Queens solution."""
        for i in range(len(perm)):
            for j in range(i + 1, len(perm)):
                if abs(perm[i] - perm[j]) == abs(i - j):
                    return False
        return True


def measure_time(N_values):
    backtracking_times = []
    brute_force_times = []

    for n in N_values:
        # Backtracking timing
        start_time = time.time()
        NQueensVisualizerBacktracking(n)
        backtracking_times.append(time.time() - start_time)

        # Brute Force timing
        start_time = time.time()
        if n <= 10:  # Brute force becomes infeasible for large N
            NQueensVisualizerBruteForce(n)
            brute_force_times.append(time.time() - start_time)
        else:
            brute_force_times.append(None)  # Mark as None for large N

    return backtracking_times, brute_force_times


def plot_results(N_values, backtracking_times, brute_force_times):
    plt.figure(figsize=(10, 6))

    # Plot backtracking results
    plt.plot(N_values, backtracking_times, label="Backtracking", marker="o")

    # Plot brute force results (ignoring None values)
    bf_x = [N_values[i] for i in range(len(brute_force_times)) if brute_force_times[i] is not None]
    bf_y = [t for t in brute_force_times if t is not None]
    plt.plot(bf_x, bf_y, label="Brute Force", marker="o")

    plt.title("N-Queens Time Complexity Comparison")
    plt.xlabel("Board Size (N)")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Compare performance for various board sizes
    N_values = range(4, 12)  # Change range as needed
    bt_times, bf_times = measure_time(N_values)

    # Plot the results
    plot_results(N_values, bt_times, bf_times)

    # Optional: Visualize solutions for a given N
    N = int(input("Enter the board size for visualization: "))
    mode = input("Choose mode (backtracking/bruteforce): ").strip().lower()
    if mode == "backtracking":
        NQueensVisualizerBacktracking(N).display_solutions()
    elif mode == "bruteforce":
        NQueensVisualizerBruteForce(N).display_solutions()
    else:
        print("Invalid mode selected!")
