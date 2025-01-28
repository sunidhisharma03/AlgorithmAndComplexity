import random
import matplotlib.pyplot as plt

def evaluate_board(board):
    n = len(board)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                conflicts += 1
    return conflicts

def monte_carlo_n_queens(n, iterations=10000):
    best_board = None
    best_conflicts = float('inf')
    conflict_progress = []

    for _ in range(iterations):
        board = list(range(n))
        random.shuffle(board)
        conflicts = evaluate_board(board)

        if conflicts < best_conflicts:
            best_board = board
            best_conflicts = conflicts

        conflict_progress.append(best_conflicts)

        if best_conflicts == 0:
            break
    
    return best_board, best_conflicts, conflict_progress

# Example usage
n = 8  # Size of the board (N-Queens problem)
iterations = 10000  # Number of iterations to perform
solution, conflicts, conflict_progress = monte_carlo_n_queens(n, iterations)

# Plotting the conflict progression
plt.figure(figsize=(10, 6))
plt.plot(range(len(conflict_progress)), conflict_progress, label="Conflicts", color="blue")
plt.xlabel('Iterations')
plt.ylabel('Number of Conflicts')
plt.title('Monte Carlo N-Queens: Conflict Reduction Over Iterations')
plt.legend()
plt.grid(True)
plt.show()

# Output the result in the terminal
if conflicts == 0:
    print(f"Solution found: {solution}")
else:
    print(f"No perfect solution found. Best board: {solution} with {conflicts} conflicts.")
