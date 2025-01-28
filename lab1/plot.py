from heapsort import heap_sort
from insertionsort import insertion_sort
from mergesort import merge_sort
from quicksort import quick_sort
from selectionsort import selection_sort
import time
import random

import matplotlib.pyplot as plt

def analyze_sorting_algorithms():
    algorithms = {
        'Heap Sort': heap_sort,
        'Insertion Sort': insertion_sort,
        'Merge Sort': merge_sort,
        'Quick Sort': quick_sort,
        'Selection Sort': selection_sort
    }

    # input_sizes = [100, 500, 1000, 5000, 10000]
    input_sizes = list(range(100,5000,400))
    results = {name: [] for name in algorithms}

    for size in input_sizes:
        for name, sort_function in algorithms.items():
            total_time = 0
            for _ in range(10):  # Run each algorithm 5 times for each input size
                data = [random.randint(0, 10000) for _ in range(size)]
                start_time = time.time()
                sort_function(data)
                end_time = time.time()
                total_time += (end_time - start_time)
            average_time = total_time / 5
            results[name].append(average_time)

    for name, times in results.items():
        plt.plot(input_sizes, times, label=name)

    plt.xlabel('Input Size')
    plt.ylabel('Average Time (seconds)')
    plt.title('Time Complexity of Sorting Algorithms')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    analyze_sorting_algorithms()