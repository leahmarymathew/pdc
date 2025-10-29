#include <iostream>
#include <vector>
#include <omp.h>
#include <algorithm> // For std::swap and std::sort (for creating sorted array)
#include <cstdlib>   // For rand() and srand()
#include <ctime>     // For time()

using namespace std;

// --- Partition Function (Lomuto's Scheme) ---
// This function rearranges the array based on a pivot.
// strategy: 0=middle, 1=first, 2=last
int partition(vector<int>& arr, int low, int high, int strategy) {
    int pivot_index;
    if (strategy == 0) { // Middle element
        pivot_index = low + (high - low) / 2;
    } else if (strategy == 1) { // First element
        pivot_index = low;
    } else { // Last element
        pivot_index = high;
    }

    // Move the chosen pivot to the end (high) to use as the Lomuto pivot
    swap(arr[pivot_index], arr[high]);
    int pivot_value = arr[high];

    int i = (low - 1); // Index of the last element smaller than pivot

    for (int j = low; j < high; j++) {
        // If current element is smaller than or equal to pivot
        if (arr[j] <= pivot_value) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    
    // Place pivot in its correct final position
    swap(arr[i + 1], arr[high]);
    return (i + 1); // Return the partitioning index
}

// --- Serial Quick Sort (for small subarrays) ---
// This is the standard recursive quick sort.
// It's called when the array size is below the CUTOFF.
void quick_sort_serial(vector<int>& arr, int low, int high, int strategy) {
    if (low < high) {
        int pi = partition(arr, low, high, strategy);
        quick_sort_serial(arr, low, pi - 1, strategy);
        quick_sort_serial(arr, pi + 1, high, strategy);
    }
}

// --- Parallel Quick Sort (Task-based) ---
// This recursive function creates tasks for parallel execution.

// Cutoff: Sub-arrays smaller than this will be sorted serially.
// This avoids the overhead of creating tasks for tiny amounts of work.
const int CUTOFF = 1000; 

void quick_sort_task(vector<int>& arr, int low, int high, int strategy) {
    if (low < high) {
        // If the sub-array is small, sort it serially
        if ((high - low) < CUTOFF) {
            quick_sort_serial(arr, low, high, strategy);
        } else {
            // Partition the array
            int pi = partition(arr, low, high, strategy);

            // Create a task for the left sub-array
            #pragma omp task
            {
                quick_sort_task(arr, low, pi - 1, strategy);
            }

            // Create a task for the right sub-array
            #pragma omp task
            {
                quick_sort_task(arr, pi + 1, high, strategy);
            }
        }
    }
}

// --- Wrapper Function for Parallel Sort ---
// This sets up the parallel region.
void quick_sort_parallel(vector<int>& arr, int n, int strategy) {
    // Start a parallel region
    #pragma omp parallel
    {
        // One thread (the first one to reach) creates the initial tasks
        #pragma omp single nowait
        {
            quick_sort_task(arr, 0, n - 1, strategy);
        }
    } // The implicit barrier at the end of the parallel region
      // ensures all tasks are completed before the function returns.
}

// --- Utility to print the array ---
void print_array(const vector<int>& arr) {
    int n = arr.size();
    if (n <= 100) {
        // Print the whole array if it's small
        for (int i = 0; i < n; i++) {
            cout << arr[i] << " ";
        }
    } else {
        // Print first and last 10 elements if it's large
        cout << "First 10: ";
        for (int i = 0; i < 10; i++) cout << arr[i] << " ";
        cout << "... Last 10: ";
        for (int i = n - 10; i < n; i++) cout << arr[i] << " ";
    }
    cout << "\n";
}

int main() {
    int n;
    cout << "Enter the number of elements (N): ";
    cin >> n;

    if (n <= 0) {
        cout << "Invalid array size." << endl;
        return 1;
    }

    vector<int> arr_orig(n);
    vector<int> arr_test(n);

    // --- (a) Accept array from user (or generate for large N) ---
    if (n <= 100) {
        cout << "Enter " << n << " elements:\n";
        for (int i = 0; i < n; i++) {
            cin >> arr_orig[i];
        }
    } else {
        cout << "Generating " << n << " random elements..." << endl;
        srand(time(0));
        for (int i = 0; i < n; i++) {
            arr_orig[i] = rand() % (n * 10);
        }
    }

    cout << "\n--- Quick Sort Time Complexities ---" << endl;
    cout << "Best Case:   O(n log n) - Pivot divides array equally." << endl;
    cout << "Average Case: O(n log n) - Pivot divides array reasonably well." << endl;
    cout << "Worst Case:  O(n^2)     - Pivot is always min/max (e.g., sorted array with first element pivot)." << endl;
    

    cout << "\n--- Running Test Cases ---" << endl;
    double start_time, end_time;

    // --- Test Case 1: Unsorted array - Pivot=middle ---
    cout << "\nTest 1: Unsorted, Pivot = Middle" << endl;
    arr_test = arr_orig; // Reset to original unsorted data
    start_time = omp_get_wtime();
    quick_sort_parallel(arr_test, n, 0); // 0 = middle
    end_time = omp_get_wtime();
    cout << "Time: " << (end_time - start_time) << " seconds." << endl;
    cout << "Sorted: ";
    print_array(arr_test);

    // --- Test Case 2: Unsorted array - Pivot=first ---
    cout << "\nTest 2: Unsorted, Pivot = First" << endl;
    arr_test = arr_orig; // Reset
    start_time = omp_get_wtime();
    quick_sort_parallel(arr_test, n, 1); // 1 = first
    end_time = omp_get_wtime();
    cout << "Time: " << (end_time - start_time) << " seconds." << endl;

    // --- Test Case 3: Unsorted array - Pivot=last ---
    cout << "\nTest 3: Unsorted, Pivot = Last" << endl;
    arr_test = arr_orig; // Reset
    start_time = omp_get_wtime();
    quick_sort_parallel(arr_test, n, 2); // 2 = last
    end_time = omp_get_wtime();
    cout << "Time: " << (end_time - start_time) << " seconds." << endl;
    
    // --- Create a sorted array for the next tests ---
    cout << "\n--- Creating sorted array for worst-case tests ---" << endl;
    vector<int> arr_sorted = arr_orig;
    sort(arr_sorted.begin(), arr_sorted.end());
    cout << "Sorted array created." << endl;


    // --- Test Case 4: Sorted array - Pivot=first ---
    cout << "\nTest 4: Sorted, Pivot = First (Worst Case)" << endl;
    arr_test = arr_sorted; // Use the pre-sorted array
    start_time = omp_get_wtime();
    quick_sort_parallel(arr_test, n, 1); // 1 = first
    end_time = omp_get_wtime();
    cout << "Time: " << (end_time - start_time) << " seconds." << endl;

    // --- Test Case 5: Sorted array - Pivot=last ---
    cout << "\nTest 5: Sorted, Pivot = Last (Worst Case)" << endl;
    arr_test = arr_sorted; // Reset to sorted array
    start_time = omp_get_wtime();
    quick_sort_parallel(arr_test, n, 2); // 2 = last
    end_time = omp_get_wtime();
    cout << "Time: " << (end_time - start_time) << " seconds." << endl;
    
    // --- Test Case 6: Sorted array - Pivot=middle ---
    cout << "\nTest 6: Sorted, Pivot = Middle (Best Case for Sorted)" << endl;
    arr_test = arr_sorted; // Reset to sorted array
    start_time = omp_get_wtime();
    quick_sort_parallel(arr_test, n, 0); // 0 = middle
    end_time = omp_get_wtime();
    cout << "Time: " << (end_time - start_time) << " seconds." << endl;


    cout << "\n--- Justification of Results ---" << endl;
    cout << "Fill in a table with your observed times. You will likely see:\n" << endl;
    cout << "* **Test 1, 2, 3 (Unsorted):** These should all have fast, similar times. They represent the Average Case (O(n log n)) because the random data leads to good partitions.\n" << endl;
    cout << "* **Test 4 & 5 (Sorted, Pivot=First/Last):** These will be **significantly slower**. This is the Worst Case (O(n^2)). The pivot is always the smallest (Test 4) or largest (Test 5) element. The partition is extremely unbalanced (0 elements on one side, n-1 on the other). This prevents any meaningful parallelism and the recursion depth becomes 'n'.\n" << endl;
    cout << "* **Test 6 (Sorted, Pivot=Middle):** This will be fast again, even on sorted data. This is because picking the middle element of a sorted array is the *perfect* pivot, resulting in the Best Case (O(n log n)).\n" << endl;

    cout << "--- Example Table (Fill with your values) ---" << endl;
    cout << "------------------------------------------------------------------------------------------------------" << endl;
    cout << "| N         | Test 1 (Unsorted, Mid) | Test 2 (Unsorted, First) | Test 4 (Sorted, First) | Test 6 (Sorted, Mid) |" << endl;
    cout << "------------------------------------------------------------------------------------------------------" << endl;
    cout << "| 10,000    | (time 1)               | (time 2)                 | (time 3) >> (time 2)   | (time 4) ~ (time 1)  |" << endl;
    cout << "| 1,000,000 | (time 1)               | (time 2)                 | (time 3) >> (time 2)   | (time 4) ~ (time 1)  |" << endl;
    cout << "| 5,000,000 | (time 1)               | (time 2)                 | (time 3) >> (time 2)   | (time 4) ~ (time 1)  |" << endl;
    cout << "------------------------------------------------------------------------------------------------------" << endl;


    return 0;
}
