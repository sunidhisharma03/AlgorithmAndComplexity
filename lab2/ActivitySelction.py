import random
import time
import matplotlib.pyplot as plt

# Class to represent an activity
class Activity:
    def __init__(self, start, finish):
        self.start = start
        self.finish = finish

# Greedy function to calculate the maximum set of activities
def greedyMaxActivities(activities):
    # Sort activities based on their finish time
    activities.sort(key=lambda x: x.finish)

    n = len(activities)

    # The first activity is always selected
    i = 0
    selected_activities = 1

    # Iterate through the remaining activities
    for j in range(1, n):
        if activities[j].start >= activities[i].finish:
            selected_activities += 1
            i = j

    return selected_activities

# Brute-force function to calculate the maximum set of activities
def bruteForceMaxActivities(activities):
    n = len(activities)
    max_count = 0

    # Generate all possible subsets of activities
    for i in range(1, 1 << n):
        subset = [activities[j] for j in range(n) if (i & (1 << j)) > 0]
        
        # Check if the subset is valid (no overlapping activities)
        subset.sort(key=lambda x: x.finish)
        valid = True
        for k in range(len(subset) - 1):
            if subset[k].finish > subset[k + 1].start:
                valid = False
                break

        # Update max_count if this subset is valid and larger
        if valid:
            max_count = max(max_count, len(subset))

    return max_count

# Generate random start and finish times for activities
def generateActivities(num_activities, max_time):
    activities = []
    for _ in range(num_activities):
        start = random.randint(0, max_time // 2)
        finish = random.randint(start + 1, max_time)
        activities.append(Activity(start, finish))
    return activities

# Measure execution time for both approaches
def measureExecutionTimes(max_num_activities, step, max_time, runs=10):
    num_activities_list = list(range(10, max_num_activities + 1, step))
    greedy_times = []
    brute_force_times = []

    for num_activities in num_activities_list:
        greedy_total_time = 0
        brute_force_total_time = 0

        for _ in range(runs):
            activities = generateActivities(num_activities, max_time)

            # Measure execution time for greedy approach
            start_time = time.time()
            greedyMaxActivities(activities)
            end_time = time.time()
            greedy_total_time += (end_time - start_time)

            # Measure execution time for brute-force approach
            start_time = time.time()
            bruteForceMaxActivities(activities)
            end_time = time.time()
            brute_force_total_time += (end_time - start_time)

        # Average the execution time over all runs
        greedy_times.append(greedy_total_time / runs)
        brute_force_times.append(brute_force_total_time / runs)

    return num_activities_list, greedy_times, brute_force_times

# Plot the execution time graph
def plotExecutionTimes(num_activities_list, greedy_times, brute_force_times):
    plt.figure(figsize=(10, 6))
    plt.plot(num_activities_list, greedy_times, marker='o', label='Greedy Algorithm', color='blue')
    plt.plot(num_activities_list, brute_force_times, marker='o', label='Brute Force Algorithm', color='red')
    plt.title("Execution Time: Greedy vs Brute Force")
    plt.xlabel("Number of Activities")
    plt.ylabel("Average Execution Time (seconds)")
    plt.grid(True)
    plt.legend()
    plt.show()

# Driver code
if __name__ == "__main__":
    max_num_activities = 20  # Reduced maximum number of activities for brute force
    step = 2                  # Step size for increasing number of activities
    max_time = 100            # Maximum possible finish time
    runs = 10                 # Number of runs for averaging

    # Measure execution times for both approaches
    num_activities_list, greedy_times, brute_force_times = measureExecutionTimes(max_num_activities, step, max_time, runs)

    # Plot the results
    plotExecutionTimes(num_activities_list, greedy_times, brute_force_times)
