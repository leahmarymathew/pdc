#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip> // For setprecision
#include <cmath>   // For fabs
#include <algorithm> // For swap

using namespace std;

// Helper function to print the solution vector
void printSolution(const vector<double>& x) {
    cout << "Solution: ";
    for (size_t i = 0; i < x.size(); ++i) {
        cout << "x" << i << " = " << x[i] << (i == x.size() - 1 ? "" : ", ");
    }
    cout << endl;
}

// (a) Serial Gaussian Elimination with Pivoting and Backward Substitution
vector<double> serialSolve(int N, vector<vector<double>>& Ab) {
    
    // --- Forward Elimination (with Pivoting) ---
    for (int k = 0; k < N; ++k) {
        // 1. Find pivot row (max absolute value in column k)
        int max_row = k;
        for (int i = k + 1; i < N; ++i) {
            if (fabs(Ab[i][k]) > fabs(Ab[max_row][k])) {
                max_row = i;
            }
        }

        // 2. Swap current row (k) with pivot row (max_row)
        swap(Ab[k], Ab[max_row]);

        // 3. Elimination
        // For all rows below the pivot
        for (int i = k + 1; i < N; ++i) {
            double factor = Ab[i][k] / Ab[k][k];
            // For all columns to the right of k (including b)
            for (int j = k; j <= N; ++j) {
                Ab[i][j] -= factor * Ab[k][j];
            }
        }
    }

    // --- Backward Substitution ---
    vector<double> x(N);
    for (int i = N - 1; i >= 0; --i) {
        x[i] = Ab[i][N]; // Start with the b-value
        // Subtract all known x[j] values
        for (int j = i + 1; j < N; ++j) {
            x[i] -= Ab[i][j] * x[j];
        }
        // Divide by the diagonal
        x[i] /= Ab[i][i];
    }

    return x;
}


// (b) Parallel Gaussian Elimination
vector<double> parallelSolve(int N, vector<vector<double>>& Ab) {
    
    // --- Forward Elimination (with Pivoting) ---
    for (int k = 0; k < N; ++k) {
        // 1. Find pivot row (serial)
        // This is a "reduction" and could be parallelized,
        // but for N=3, the overhead would be much larger than the work.
        int max_row = k;
        for (int i = k + 1; i < N; ++i) {
            if (fabs(Ab[i][k]) > fabs(Ab[max_row][k])) {
                max_row = i;
            }
        }

        // 2. Swap current row (k) with pivot row (serial)
        swap(Ab[k], Ab[max_row]);

        // 3. Elimination (Parallel)
        // The outer loop (k) MUST be sequential.
        // But for a given k, all rows (i) below it can be updated independently.
        // This is the main source of parallelism.
        #pragma omp parallel
        {
            // Print thread info once per parallel region
            #pragma omp single
            {
                cout << "  [Elimination phase for k=" << k << " using " << omp_get_num_threads() << " threads]" << endl;
            }

            // Distribute the 'i' rows among the threads
            #pragma omp for
            for (int i = k + 1; i < N; ++i) {
                double factor = Ab[i][k] / Ab[k][k];
                // Each thread handles its 'i' row's columns
                for (int j = k; j <= N; ++j) {
                    Ab[i][j] -= factor * Ab[k][j];
                }
            }
            // Implicit barrier here: all threads wait until all 'i' rows are done
        }
    }

    // --- Backward Substitution (Serial) ---
    // This part is inherently sequential. To find x[i],
    // you MUST already have the values for x[i+1], x[i+2], etc.
    // This is a classic "loop-carried dependency".
    vector<double> x(N);
    for (int i = N - 1; i >= 0; --i) {
        x[i] = Ab[i][N];
        for (int j = i + 1; j < N; ++j) {
            x[i] -= Ab[i][j] * x[j];
        }
        x[i] /= Ab[i][i];
    }

    return x;
}


int main() {
    cout << fixed << setprecision(8);

    // --- Test Case 1 (From prompt section (a)) ---
    cout << "--- Test Case 1: (x,y,z) = (1.666..., -0.833..., 1.5) ---" << endl;
    int N1 = 3;
    vector<vector<double>> Ab1_orig = {{1, -1, 1, 4}, {1, -4, 2, 8}, {1, 2, 8, 12}};
    
    // We must pass copies, as the functions modify the matrix in-place
    vector<vector<double>> Ab_s1 = Ab1_orig; 
    vector<vector<double>> Ab_p1 = Ab1_orig; 

    // (a) Serial
    cout << "(a) Serial Version:" << endl;
    double start_s1 = omp_get_wtime();
    vector<double> x_s1 = serialSolve(N1, Ab_s1);
    double end_s1 = omp_get_wtime();
    printSolution(x_s1);
    cout << "Serial Time (Set 1): " << (end_s1 - start_s1) << " s" << endl << endl;

    // (b) Parallel
    cout << "(b) Parallel Version:" << endl;
    double start_p1 = omp_get_wtime();
    vector<double> x_p1 = parallelSolve(N1, Ab_p1);
    double end_p1 = omp_get_wtime();
    printSolution(x_p1);
    cout << "Parallel Time (Set 1): " << (end_p1 - start_p1) << " s" << endl << endl;


    // --- Test Case 2 (Set 2) ---
    cout << "--- Test Case 2: (x,y,z) = (4, -3, 1) ---" << endl;
    int N2 = 3;
    vector<vector<double>> Ab2_orig = {{1, -1, 1, 8}, {2, 3, -1, -2}, {3, -2, -9, 9}};
    
    vector<vector<double>> Ab_s2 = Ab2_orig; 
    vector<vector<double>> Ab_p2 = Ab2_orig; 

    // (a) Serial
    cout << "(a) Serial Version:" << endl;
    double start_s2 = omp_get_wtime();
    vector<double> x_s2 = serialSolve(N2, Ab_s2);
    double end_s2 = omp_get_wtime();
    printSolution(x_s2);
    cout << "Serial Time (Set 2): " << (end_s2 - start_s2) << " s" << endl << endl;

    // (b) Parallel
    cout << "(b) Parallel Version:" << endl;
    double start_p2 = omp_get_wtime();
    vector<double> x_p2 = parallelSolve(N2, Ab_p2);
    double end_p2 = omp_get_wtime();
    printSolution(x_p2);
    cout << "Parallel Time (Set 2): " << (end_p2 - start_p2) << " s" << endl;

    cout << "\n(c) See text explanation for Race Condition analysis." << endl;

    return 0;
}
