import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import itertools

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

    def display_solutions(self):
        """Display all solutions using matplotlib."""
        if not self.solutions:
            print("No solutions found.")
            return

        for idx, solution in enumerate(self.solutions):
            print(f"Displaying solution {idx + 1}/{len(self.solutions)}...")
            self._display_solution(solution)

    def _display_solution(self, solution):
        """Display a single solution using matplotlib."""
        fig, ax = plt.subplots()
        board_size = self.n
        ax.set_xlim(0, board_size)
        ax.set_ylim(0, board_size)
        ax.set_xticks(range(board_size + 1))
        ax.set_yticks(range(board_size + 1))
        ax.grid(True)

        for row, col in enumerate(solution):
            ax.add_patch(patches.Rectangle((col, board_size - row - 1), 1, 1, color='lightblue'))
            ax.text(col + 0.5, board_size - row - 0.5, 'Q', fontsize=20, ha='center', va='center')

        plt.title(f"N-Queens Solution (N={self.n})")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


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

    def display_solutions(self):
        """Display all solutions using matplotlib."""
        if not self.solutions:
            print("No solutions found.")
            return

        for idx, solution in enumerate(self.solutions):
            print(f"Displaying solution {idx + 1}/{len(self.solutions)}...")
            self._display_solution(solution)

    def _display_solution(self, solution):
        """Display a single solution using matplotlib."""
        fig, ax = plt.subplots()
        board_size = self.n
        ax.set_xlim(0, board_size)
        ax.set_ylim(0, board_size)
        ax.set_xticks(range(board_size + 1))
        ax.set_yticks(range(board_size + 1))
        ax.grid(True)

        for row, col in enumerate(solution):
            ax.add_patch(patches.Rectangle((col, board_size - row - 1), 1, 1, color='lightblue'))
            ax.text(col + 0.5, board_size - row - 0.5, 'Q', fontsize=20, ha='center', va='center')

        plt.title(f"N-Queens Solution (N={self.n})")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


def plot_results(N_values, backtracking_times, brute_force_times):
    plt.figure(figsize=(10, 6))

    # Plot backtracking results
    plt.plot(N_values, backtracking_times, label="Backtracking", marker="o", color='b')

    # Plot brute force results (ignoring None values)
    bf_x = [N_values[i] for i in range(len(brute_force_times)) if brute_force_times[i] is not None]
    bf_y = [t for t in brute_force_times if t is not None]
    plt.plot(bf_x, bf_y, label="Brute Force", marker="o", color='r')

    plt.title("N-Queens Time Complexity Comparison")
    plt.xlabel("Board Size (N)")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()


def measure_time(N_values):
    backtracking_times = []
    brute_force_times = []

    for n in N_values:
        print(f"Measuring performance for N={n}...")

        # Backtracking timing
        start_time = time.time()
        NQueensVisualizerBacktracking(n)
        backtracking_times.append(time.time() - start_time)

        # Brute Force timing
        if n <= 10:  # Brute force becomes infeasible for large N
            start_time = time.time()
            NQueensVisualizerBruteForce(n)
            brute_force_times.append(time.time() - start_time)
        else:
            brute_force_times.append(None)  # Mark as None for large N

    return backtracking_times, brute_force_times


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
        if N <= 10:  # Ensure brute force is feasible for the entered N
            NQueensVisualizerBruteForce(N).display_solutions()
        else:
            print("Brute force is not feasible for N > 10.")
    else:
        print("Invalid mode selected!")
