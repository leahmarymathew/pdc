#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip> // For setprecision
#include <cstdlib> // For rand
#include <ctime>   // For time

using namespace std;

// Use 'double' for better precision in multiplication
typedef vector<vector<double>> Matrix;

// Initialize matrix with random values
void initMatrix(Matrix& mat, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            mat[i][j] = (rand() % 100) / 10.0; // Random 0.0 to 9.9
        }
    }
}

// Print matrix (for small N)
void printMatrix(const Matrix& mat, int N) {
    if (N > 10) {
        cout << "[Matrix too large to print]" << endl;
        return;
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << setw(6) << mat[i][j] << " ";
        }
        cout << endl;
    }
}

// 4. Serial Matrix Multiplication
double serialMatMul(const Matrix& A, const Matrix& B, Matrix& C, int N) {
    double start_time = omp_get_wtime();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0.0; // Initialize
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// 4. Parallel Matrix Multiplication
double parallelMatMul(const Matrix& A, const Matrix& B, Matrix& C, int N) {
    double start_time = omp_get_wtime();
    
    // Parallelize the outer two loops
    // 'collapse(2)' tells OpenMP to parallelize the i and j loops as one
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0.0; // Initialize
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

int main() {
    srand(time(NULL)); // Seed random number generator
    cout << fixed << setprecision(8);

    vector<int> dimensions = {3, 10, 100, 1000};
    
    cout << "--- Matrix Multiplication (Question 4) ---" << endl;
    cout << setw(12) << "Dimension (N)"
         << setw(20) << "Serial Time (s)"
         << setw(20) << "Parallel Time (s)" << endl;
    cout << "---------------------------------------------------------" << endl;

    for (int N : dimensions) {
        // Allocate matrices
        Matrix A(N, vector<double>(N));
        Matrix B(N, vector<double>(N));
        Matrix C_serial(N, vector<double>(N));
        Matrix C_parallel(N, vector<double>(N));

        // Initialize A and B
        initMatrix(A, N);
        initMatrix(B, N);

        double serial_time = serialMatMul(A, B, C_serial, N);
        double parallel_time = parallelMatMul(A, B, C_parallel, N);

        cout << setw(12) << N
             << setw(20) << serial_time
             << setw(20) << parallel_time << endl;
             
        if (N == 3) {
            cout << "\nN=3 Serial Result:" << endl;
            printMatrix(C_serial, N);
            cout << "\nN=3 Parallel Result:" << endl;
            printMatrix(C_parallel, N);
            cout << "\n---------------------------------------------------------" << endl;
        }
    }

    return 0;
}
