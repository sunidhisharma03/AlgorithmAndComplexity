import random

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot_index = random.randint(0, len(arr) - 1)
        pivot = arr[pivot_index]
        less_than_pivot = [x for x in arr if x < pivot]
        equal_to_pivot = [x for x in arr if x == pivot]
        greater_than_pivot = [x for x in arr if x > pivot]
        return quick_sort(less_than_pivot) + equal_to_pivot + quick_sort(greater_than_pivot)

# Example usage
if __name__ == "__main__":
    arr = [3, 6, 8, 10, 1, 2, 1]
    print("Original array:", arr)
    sorted_arr = quick_sort(arr)
    print("Sorted array:", sorted_arr)