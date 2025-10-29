#include <iostream>
#include <vector>
#include <omp.h>
#include <random>       // Still needed for the first serialPi
#include <iomanip>
#include <cstdlib>      // For rand, srand, RAND_MAX
#include <ctime>        // For time()

using namespace std;

void printArray(const vector<int>& arr) {
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;
}

void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

void serialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void serialOddEvenSort(vector<int>& arr) {
    int n = arr.size();
    bool sorted = false;
    while (!sorted) {
        sorted = true;
        for (int i = 1; i <= n - 2; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                sorted = false;
            }
        }
        for (int i = 0; i <= n - 2; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                sorted = false;
            }
        }
    }
}

void parallelOddEvenSort(vector<int>& arr) {
    int n = arr.size();
    for (int phase = 0; phase < n; ++phase) {
        if (phase % 2 == 0) {
            #pragma omp parallel for
            for (int i = 0; i < n - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    swap(arr[i], arr[i + 1]);
                }
            }
        } else {
            #pragma omp parallel for
            for (int i = 1; i < n - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    swap(arr[i], arr[i + 1]);
                }
            }
        }
    }
}

void runQ1() {
    vector<int> testCase = {19, 2, 72, 3, 18, 57, 603, 490, 45, 101};
    vector<int> arr1 = testCase;
    vector<int> arr2 = testCase;
    vector<int> arr3 = testCase;

    cout << "--- Question 1 ---" << endl;
    cout << "Original Array: ";
    printArray(testCase);

    serialBubbleSort(arr1);
    cout << "1a) Serial Bubble Sort: ";
    printArray(arr1);

    serialOddEvenSort(arr2);
    cout << "1b) Serial Odd-Even Sort: ";
    printArray(arr2);

    parallelOddEvenSort(arr3);
    cout << "1c) Parallel Odd-Even Sort: ";
    printArray(arr3);
    cout << "--------------------" << endl;
}

// Updated serialPi function
void serialPi(long n) {
    long circle_count = 0;
    double start = omp_get_wtime();
    for (long i = 0; i < n; ++i) {
        double x = rand() / (double)RAND_MAX;
        double y = rand() / (double)RAND_MAX;
        if (x * x + y * y <= 1.0) {
            circle_count++;
        }
    }
    double pi = 4.0 * circle_count / n;
    double end = omp_get_wtime();
    cout << "Pi: " << pi << ", Time: " << (end - start) << "s" << endl;
}

void parallelPi(long n) {
    long circle_count = 0;
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num();
        #pragma omp for
        for (long i = 0; i < n; ++i) {
            double x = rand_r(&seed) / (double)RAND_MAX;
            double y = rand_r(&seed) / (double)RAND_MAX;
            if (x * x + y * y <= 1.0) {
                #pragma omp atomic
                circle_count++;
            }
        }
    }
    double pi = 4.0 * circle_count / n;
    double end = omp_get_wtime();
    cout << "Pi: " << pi << ", Time: " << (end - start) << "s" << endl;
}

void parallelPiCritical(long n) {
    long circle_count = 0;
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num();
        #pragma omp for
        for (long i = 0; i < n; ++i) {
            double x = rand_r(&seed) / (double)RAND_MAX;
            double y = rand_r(&seed) / (double)RAND_MAX;
            if (x * x + y * y <= 1.0) {
                #pragma omp critical
                {
                    circle_count++;
                }
            }
        }
    }
    double pi = 4.0 * circle_count / n;
    double end = omp_get_wtime();
    cout << "Pi: " << pi << ", Time: " << (end - start) << "s" << endl;
}

void parallelPiAtomic(long n) {
    long circle_count = 0;
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num();
        #pragma omp for
        for (long i = 0; i < n; ++i) {
            double x = rand_r(&seed) / (double)RAND_MAX;
            double y = rand_r(&seed) / (double)RAND_MAX;
            if (x * x + y * y <= 1.0) {
                #pragma omp atomic
                circle_count++;
            }
        }
    }
    double pi = 4.0 * circle_count / n;
    double end = omp_get_wtime();
    cout << "Pi: " << pi << ", Time: " << (end - start) << "s" << endl;
}

void parallelPiReduction(long n) {
    long circle_count = 0;
    double start = omp_get_wtime();
    #pragma omp parallel reduction(+:circle_count)
    {
        unsigned int seed = omp_get_thread_num();
        #pragma omp for
        for (long i = 0; i < n; ++i) {
            double x = rand_r(&seed) / (double)RAND_MAX;
            double y = rand_r(&seed) / (double)RAND_MAX;
            if (x * x + y * y <= 1.0) {
                circle_count++;
            }
        }
    }
    double pi = 4.0 * circle_count / n;
    double end = omp_get_wtime();
    cout << "Pi: " << pi << ", Time: " << (end - start) << "s" << endl;
}

void runQ2() {
    long num_points = 10000000;
    cout << fixed << setprecision(8);
    cout << "--- Question 2 (N=" << num_points << ") ---" << endl;

    cout << "2a) Serial Version:           ";
    serialPi(num_points);

    cout << "2b) Parallel Version (Atomic):  ";
    parallelPi(num_points);

    cout << "--- 2c) Race Condition ---" << endl;
    cout << "    Using 'critical': ";
    parallelPiCritical(num_points);
    cout << "    Using 'atomic':   ";
    parallelPiAtomic(num_points);
    cout << "    Using 'reduction':";
    parallelPiReduction(num_points);
    cout << "------------------------" << endl;
}

int main() {
    // Seed the C-style random number generator once
    srand(time(NULL)); 
    
    runQ1();
    runQ2();
    return 0;
}
