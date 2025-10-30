#include <iostream>
#include <omp.h>
#include <vector>
#include <random>
using namespace std;

// Generate a vector with random numbers from 1 to n
vector<int> generateRandomVector(int size, int maxValue)
{
    vector<int> result(size);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1, maxValue);

    for (int i = 0; i < size; i++)
    {
        result[i] = distrib(gen);
    }
    return result;
}

vector<int> oddEvenSortSerial(const vector<int> &input)
{
    vector<int> arr = input; // Create a copy
    int n = arr.size();
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
    return arr;
}

vector<int> oddEvenSortParallel(const vector<int> &input)
{
    vector<int> arr = input; // Create a copy
    int n = arr.size();
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
    return arr;
}

vector<int> bubbleSort(const vector<int> &input)
{
    vector<int> arr = input; // Create a copy
    int n = arr.size();
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
    return arr;
}

int main()
{
    int n = 100000000; // Using larger array for better parallel comparison
    vector<int> arr = generateRandomVector(n, n);
    vector<int> sorted;
    double start, end;

    cout << "Array size: " << n << "\n";
    cout << "Sample of original array (first 10 elements): \n";
    for (int i = 0; i < min(10, n); i++)
        cout << arr[i] << " ";
    cout << "\n\n";

    cout << "Bubble Sort: \n";
    start = omp_get_wtime();
    sorted = bubbleSort(arr);
    end = omp_get_wtime();
    cout << "First 10 elements after sorting: \n";
    for (int i = 0; i < min(10, n); i++)
        cout << sorted[i] << " ";
    cout << endl;
    cout << "Time Taken: " << end - start << " seconds";
    cout << "\n\n";

    cout << "Serial OET Sort: \n";
    start = omp_get_wtime();
    sorted = oddEvenSortSerial(arr);
    end = omp_get_wtime();
    cout << "First 10 elements after sorting: \n";
    for (int i = 0; i < min(10, n); i++)
        cout << sorted[i] << " ";
    cout << endl;
    cout << "Time Taken: " << end - start << " seconds";
    cout << "\n\n";

    cout << "Parallel OET Sort: \n";
    start = omp_get_wtime();
    sorted = oddEvenSortParallel(arr);
    end = omp_get_wtime();
    cout << "First 10 elements after sorting: \n";
    for (int i = 0; i < min(10, n); i++)
        cout << sorted[i] << " ";
    cout << endl;
    cout << "Time Taken: " << end - start << " seconds";
    cout << "\n\n";
}