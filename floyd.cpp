#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip> // For setprecision and setw
#include <climits> // For INT_MAX

using namespace std;

// Define "infinity" as a large number
// Use INT_MAX / 2 to avoid integer overflow when adding
const int INF = INT_MAX / 2;

// --- Helper Function to Print Matrix ---
void printMatrix(const vector<vector<int>>& dist, int N) {
    cout << "Shortest Path Matrix:" << endl;
    cout << "      ";
    for (int i = 0; i < N; ++i) {
        cout << setw(5) << "v" << i;
    }
    cout << endl;
    cout << "-----------------------------------" << endl;
    for (int i = 0; i < N; ++i) {
        cout << "v" << i << " | ";
        for (int j = 0; j < N; ++j) {
            if (dist[i][j] == INF) {
                cout << setw(5) << "INF";
            } else {
                cout << setw(5) << dist[i][j];
            }
        }
        cout << endl;
    }
}

// --- Serial Floyd-Warshall ---
double serialFloydWarshall(vector<vector<int>>& adj, vector<vector<int>>& dist) {
    int N = adj.size();
    // Initialize dist matrix
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            dist[i][j] = adj[i][j];
        }
    }

    double start_time = omp_get_wtime();

    // The core algorithm
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                // If path via k is shorter, update it
                if (dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// --- Parallel Floyd-Warshall ---
double parallelFloydWarshall(vector<vector<int>>& adj, vector<vector<int>>& dist) {
    int N = adj.size();
    // Initialize dist matrix
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            dist[i][j] = adj[i][j];
        }
    }

    double start_time = omp_get_wtime();

    // The k-loop MUST be sequential (it's a data dependency)
    for (int k = 0; k < N; ++k) {
        // The i and j loops can be parallelized
        // We can process all pairs (i, j) independently for a given k
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                // If path via k is shorter, update it
                // This is a "read-read-write" pattern, but each thread
                // writes to a unique dist[i][j], so no race condition.
                // All threads only READ from the k-th row/column.
                // (Note: This is safe *within* one k-iteration)
                if (dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}


int main() {
    cout << fixed << setprecision(8);

    // --- Test Case 1: Positive Weights ---
    // A 4-node graph
    int N1 = 4;
    vector<vector<int>> adj1 = {
        {0, 5, INF, 10},
        {INF, 0, 3, INF},
        {INF, INF, 0, 1},
        {INF, INF, INF, 0}
    };
    vector<vector<int>> dist_s1(N1, vector<int>(N1));
    vector<vector<int>> dist_p1(N1, vector<int>(N1));

    cout << "--- Test Case 1: Positive Weights (N=4) ---" << endl;
    double serial_time1 = serialFloydWarshall(adj1, dist_s1);
    cout << "(1) Result Table - Serial:" << endl;
    printMatrix(dist_s1, N1);
    cout << "\nSerial Execution Time: " << serial_time1 << " s\n" << endl;

    double parallel_time1 = parallelFloydWarshall(adj1, dist_p1);
    cout << "(1) Result Table - Parallel:" << endl;
    printMatrix(dist_p1, N1);
    cout << "\nParallel Execution Time: " << parallel_time1 << " s\n" << endl;


    // --- Test Case 2: Negative Weights ---
    // A 4-node graph with some negative edges (no negative cycles)
    int N2 = 4;
    vector<vector<int>> adj2 = {
        {0, 1, INF, INF},
        {INF, 0, -1, INF},
        {INF, INF, 0, -1},
        {-1, INF, INF, 0}
    };
    vector<vector<int>> dist_s2(N2, vector<int>(N2));
    vector<vector<int>> dist_p2(N2, vector<int>(N2));
    
    cout << "--- Test Case 2: Negative Weights (N=4) ---" << endl;
    double serial_time2 = serialFloydWarshall(adj2, dist_s2);
    cout << "(1) Result Table - Serial:" << endl;
    printMatrix(dist_s2, N2);
    cout << "\nSerial Execution Time: " << serial_time2 << " s\n" << endl;
    
    double parallel_time2 = parallelFloydWarshall(adj2, dist_p2);
    cout << "(1) Result Table - Parallel:" << endl;
    printMatrix(dist_p2, N2);
    cout << "\nParallel Execution Time: " << parallel_time2 << " s\n" << endl;

    // --- (2) Comparison Table ---
    cout << "--- (2) Comparison Table ---" << endl;
    cout << "---------------------------------------------------------" << endl;
    cout << setw(30) << "Test Case" 
         << setw(20) << "Serial Time (s)"
         << setw(20) << "Parallel Time (s)" << endl;
    cout << "---------------------------------------------------------" << endl;
    cout << setw(30) << "Test Case 1 (N=4, +ve)" 
         << setw(20) << serial_time1
         << setw(20) << parallel_time1 << endl;
    cout << setw(30) << "Test Case 2 (N=4, -ve)" 
         << setw(20) << serial_time2
         << setw(20) << parallel_time2 << endl;
    cout << "---------------------------------------------------------" << endl;

    cout << "\nNote: For small N (like N=4), parallel overhead"
         << " may be larger than the serial time." << endl;
         
    return 0;
}
