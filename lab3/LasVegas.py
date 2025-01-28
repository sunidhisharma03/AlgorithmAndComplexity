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

def las_vegas_n_queens(n):
    iterations = 0
    while True:
        board = list(range(n))
        random.shuffle(board)  # Randomly shuffle the queens on the board
        conflicts = evaluate_board(board)
        iterations += 1
        if conflicts == 0:
            return board, iterations  # Return the solution and the number of iterations

# Example usage
n = 8  # Size of the board (N-Queens problem)
solution, iterations = las_vegas_n_queens(n)

# Plot the conflict progression (since it's Las Vegas, it will either be 0 or non-zero)
plt.plot(range(iterations), [0]*iterations, label="Conflicts (always 0 on success)", color="green")
plt.xlabel('Iterations')
plt.ylabel('Number of Conflicts')
plt.title('Las Vegas N-Queens: Conflict Reduction Over Iterations')
plt.legend()
plt.grid(True)
plt.show()

# Output the result in the terminal
print(f"Solution found: {solution}")
print(f"Iterations taken to find the solution: {iterations}")
