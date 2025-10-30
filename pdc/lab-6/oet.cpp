#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;
void oddEvenSortSerial(vector<int> &arr, int n)
{
    bool sorted = false;
    while (!sorted)
    {
        sorted = true;
        // Odd phase
        for (int i = 1; i < n - 1; i += 2)
        {
            if (arr[i] > arr[i + 1])
            {
                swap(arr[i], arr[i + 1]);
                sorted = false;
            }
        }
        // Even phase
        for (int i = 0; i < n - 1; i += 2)
        {
            if (arr[i] > arr[i + 1])
            {
                swap(arr[i], arr[i + 1]);
                sorted = false;
            }
        }
    }
}
void oddEvenSortParallel(vector<int> &arr, int n)
{
    bool sorted = false;
    while (!sorted)
    {
        sorted = true;
#pragma omp parallel
        {
            bool localSorted = true;
#pragma omp for
            for (int i = 1; i < n - 1; i += 2)
            {
                if (arr[i] > arr[i + 1])
                {
                    swap(arr[i], arr[i + 1]);
                    localSorted = false;
                }
            }
#pragma omp barrier
#pragma omp for
            for (int i = 0; i < n - 1; i += 2)
            {
                if (arr[i] > arr[i + 1])
                {
                    swap(arr[i], arr[i + 1]);
                    localSorted = false;
                }
            }
#pragma omp critical
            {
                if (!localSorted)
                    sorted = false;
            }
        }
    }
}
void bubbleSort(vector<int> &arr, int n)
{
    bool swapped;
    for (int i = 0; i < n - 1; ++i)
    {
        swapped = false;
        for (int j = 1; j < n - i; ++j)
        {
            if (arr[j - 1] > arr[j])
            {
                swap(arr[j - 1], arr[j]);
                swapped = true;
            }
        }
        if (!swapped)
            break;
    }
}
int main()
{
    int n = 10;
    vector<int> arr = {19, 2, 72, 3, 18, 57, 603, 490, 45, 101};
    double start, end;
    cout << "Bubble Sort Output: \n";
    start = omp_get_wtime();
    bubbleSort(arr, n);
    end = omp_get_wtime();
    for (int a : arr)
        cout << a << " ";
    cout << endl;
    cout << "Time Taken: " << end - start << " seconds";
    cout << "\n\n";
    arr = {19, 2, 72, 3, 18, 57, 603, 490, 45, 101};
    cout << "Serial OET Sort Output: \n";
    start = omp_get_wtime();
    oddEvenSortSerial(arr, n);
    end = omp_get_wtime();
    for (int a : arr)
        cout << a << " ";
    cout << endl;
    cout << "Time Taken: " << end - start << " seconds";
    cout << "\n\n";
    arr = {19, 2, 72, 3, 18, 57, 603, 490, 45, 101};
    cout << "Parallel OET Sort Output: \n";
    start = omp_get_wtime();
    oddEvenSortParallel(arr, n);
    end = omp_get_wtime();
    for (int a : arr)
        cout << a << " ";
    cout << endl;
    cout << "Time Taken: " << end - start << " seconds";
    cout << "\n\n";
}