#include <iostream>
#include <omp.h>
using namespace std;

int main() {
    int N;
    cout << "Enter matrix dimension N: ";
    cin >> N;

    int **A = new int *[N];
    int **B = new int *[N];
    int **C_serial = new int *[N];
    int **C_parallel = new int *[N];

    for (int i = 0; i < N; i++) {
        A[i] = new int[N];
        B[i] = new int[N];
        C_serial[i] = new int[N];
        C_parallel[i] = new int[N];
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }

    double start = omp_get_wtime();
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            C_serial[i][j] = 0;
            for (int k = 0; k < N; k++)
                C_serial[i][j] += A[i][k] * B[k][j];
        }
    double end = omp_get_wtime();
    cout << "Serial execution time = " << end - start << " seconds" << endl;

    start = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            C_parallel[i][j] = 0;
            for (int k = 0; k < N; k++)
                C_parallel[i][j] += A[i][k] * B[k][j];
        }
    end = omp_get_wtime();
    cout << "Parallel execution time = " << end - start << " seconds" << endl;

    return 0;
}
