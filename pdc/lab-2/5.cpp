#include <iostream>
#include <omp.h>
using namespace std;

void merge(int arr[], int l, int m, int r)
{
    int n1 = m - l + 1, n2 = r - m;
    int *L = new int[n1], *R = new int[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];
    delete[] L;
    delete[] R;
}

void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;

#pragma omp parallel sections
        {
#pragma omp section
            mergeSort(arr, l, m);

#pragma omp section
            mergeSort(arr, m + 1, r);
        }
        merge(arr, l, m, r);
    }
}

int main() {
    int N;
    cout << "Enter number of elements: ";
    cin >> N;
    int *arr = new int[N];

    cout << "Enter elements: ";
    for (int i = 0; i < N; i++)
        cin >> arr[i];

    double start = omp_get_wtime();
    mergeSort(arr, 0, N - 1);
    double end = omp_get_wtime();

    cout << "Sorted array: ";
    for (int i = 0; i < N; i++)
        cout << arr[i] << " ";
    cout << endl;

    cout << "Execution time: " << end - start << " seconds" << endl;
    return 0;
}
