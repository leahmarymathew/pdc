#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h> 
using namespace std;

// Gaussian elimination with partial pivoting
void gaussian_elimination(vector<vector<double>> &A, vector<double> &b)
{
    int n = A.size();

    for (int k = 0; k < n; k++)
    {
        // Partial pivoting
        int maxRow = k;
        for (int i = k + 1; i < n; i++)
        {
            if (fabs(A[i][k]) > fabs(A[maxRow][k]))
                maxRow = i;
        }
        swap(A[k], A[maxRow]);
        swap(b[k], b[maxRow]);

        // Elimination
        for (int i = k + 1; i < n; i++)
        {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < n; j++)
            {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
}

// Backward substitution
vector<double> back_substitution(vector<vector<double>> &A, vector<double> &b)
{
    int n = A.size();
    vector<double> x(n);

    for (int i = n - 1; i >= 0; i--)
    {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++)
        {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
    return x;
}

int main()
{
    int N = 3;
    vector<vector<double>> A = {
        {1, -1, 1},
        {1, -4, 2},
        {1, 2, 8}};
    vector<double> b = {4, 8, 12};

    double start = omp_get_wtime();
    gaussian_elimination(A, b);
    vector<double> x = back_substitution(A, b);
    double end = omp_get_wtime();

    cout << "Serial Solution:\n";
    for (int i = 0; i < N; i++)
    {
        cout << "x[" << i << "] = " << x[i] << "\n";
    }
    cout << "Execution time (serial): " << (end - start) << " seconds\n";

    return 0;
}
