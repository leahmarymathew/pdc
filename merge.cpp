#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip> // For setprecision
#include <cstdlib> // For rand
#include <ctime>   // For time
#include <algorithm> // For copy

using namespace std;

// Initialize array with random values
void initArray(vector<int>& arr, int N) {
    for (int i = 0; i < N; ++i) {
        arr[i] = rand() % 10000; // Random 0 to 9999
    }
}

// Print array (for small N)
void printArray(const vector<int>& arr) {
    int N = arr.size();
    if (N > 20) {
        cout << "[Array too large to print]" << endl;
        return;
    }
    for (int i = 0; i < N; ++i) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

// Standard merge function
void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temp vectors
    vector<int> L(n1), R(n2);

    // Copy data to temp vectors
    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    // Merge the temp vectors back into arr[l..r]
    int i = 0; // Initial index of first subarray
    int j = 0; // Initial index of second subarray
    int k = l; // Initial index of merged subarray
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy remaining elements of L[]
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
    // Copy remaining elements of R[]
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// 4. a) Serial Merge Sort
void serialMergeSort(vector<int>& arr, int l, int r) {
    if (l >= r) {
        return; // Base case
    }
    int m = l + (r - l) / 2;
    serialMergeSort(arr, l, m);
    serialMergeSort(arr, m + 1, r);
    merge(arr, l, m, r);
}

// 4. b) Parallel Merge Sort (using OpenMP tasks)
// 
// 
void parallelMergeSort(vector<int>& arr, int l, int r) {
    if (l >= r) {
        return; // Base case
    }
    
    // Define a cutoff: for small subarrays, serial sort is faster
    int cutoff = 1000;
    
    if ((r - l) < cutoff) {
        // Sort serially if below cutoff
        serialMergeSort(arr, l, r);
    } else {
        int m = l + (r - l) / 2;
        
        // Create two parallel tasks for the recursive calls
        #pragma omp task
        parallelMergeSort(arr, l, m);
        
        #pragma omp task
        parallelMergeSort(arr, m + 1, r);
        
        // Wait for both tasks to complete before merging
        #pragma omp taskwait
        merge(arr, l, m, r);
    }
}

int main() {
    srand(time(NULL));
    cout << fixed << setprecision(8);

    vector<int> sizes = {10, 1000, 100000, 1000000};
    
    cout << "--- Merge Sort (Question 4) ---" << endl;
    cout << setw(12) << "Elements (N)"
         << setw(20) << "Serial Time (s)"
         << setw(20) << "Parallel Time (s)" << endl;
    cout << "---------------------------------------------------------" << endl;

    for (int N : sizes) {
        vector<int> arr_s(N);
        initArray(arr_s, N);
        // Make a copy for the parallel version
        vector<int> arr_p = arr_s;
        
        // Time Serial
        double start_s = omp_get_wtime();
        serialMergeSort(arr_s, 0, N - 1);
        double end_s = omp_get_wtime();
        
        // Time Parallel
        // We must wrap the parallelMergeSort call in a parallel region
        // and use 'single' to start the recursion
        double start_p = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            parallelMergeSort(arr_p, 0, N - 1);
        }
        double end_p = omp_get_wtime();

        cout << setw(12) << N
             << setw(20) << (end_s - start_s)
             << setw(20) << (end_p - start_p) << endl;
             
        if (N == 10) {
            cout << "\nN=10 Serial Sorted:" << endl;
            printArray(arr_s);
            cout << "N=10 Parallel Sorted:" << endl;
            printArray(arr_p);
            cout << "\n---------------------------------------------------------" << endl;
        }
    }

    return 0;
}
