#include <iostream>
#include <vector>
#include <climits> // For INT_MAX
#include <omp.h>   // OpenMP header
#include <chrono>  // For timing
#include <iomanip> // For setw

using namespace std;

#define V_TC1 6 // Vertices for Test Case 1
#define V_TC2 4 // Vertices for Test Case 2

// --- Helper Functions ---

int minDistance(const vector<int>& dist, const vector<bool>& visited, int V) {
    int minVal = INT_MAX;
    int minIdx = -1;
    for (int v = 0; v < V; v++) {
        if (!visited[v] && dist[v] <= minVal) {
            minVal = dist[v];
            minIdx = v;
        }
    }
    return minIdx;
}

void printSolution(const vector<int>& dist, int src, int V) {
    vector<char> labels;
    if (V == V_TC1) { // TC 1 Labels
        labels = {'A', 'B', 'C', 'D', 'E', 'F'};
    } else { // TC 2 Labels
        labels = {'S', 'A', 'B', 'C'};
    }

    cout << "Vertex \t Distance from Source " << labels[src] << endl;
    for (int i = 0; i < V; i++) {
        if (dist[i] == INT_MAX) {
            cout << labels[i] << " \t\t" << "INF" << endl;
        } else {
            cout << labels[i] << " \t\t" << dist[i] << endl;
        }
    }
    cout << "----------------------------------------" << endl; // Separator
}

// --- Dijkstra's Algorithm (Works for Positive Weights Only) ---

// Overload for 6x6 matrix (TC1)
void serialDijkstra(int graph[][V_TC1], int src) {
    int V = V_TC1;
    vector<int> dist(V, INT_MAX);
    vector<bool> visited(V, false);
    dist[src] = 0;
    for (int count = 0; count < V - 1; count++) {
        int u = minDistance(dist, visited, V);
        if (u == -1) break;
        visited[u] = true;
        for (int v = 0; v < V; v++) {
            if (!visited[v] && graph[u][v] && dist[u] != INT_MAX &&
                dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }
    cout << "--- Serial Dijkstra Result (TC1) ---" << endl;
    printSolution(dist, src, V);
}

// Overload for 4x4 matrix (TC2)
void serialDijkstra(int graph[][V_TC2], int src) {
    int V = V_TC2;
    vector<int> dist(V, INT_MAX);
    vector<bool> visited(V, false);
    dist[src] = 0;
    for (int count = 0; count < V - 1; count++) {
        int u = minDistance(dist, visited, V);
        if (u == -1) break;
        visited[u] = true;
        for (int v = 0; v < V; v++) {
            if (!visited[v] && graph[u][v] && dist[u] != INT_MAX &&
                dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }
    cout << "--- (INCORRECT) Serial Dijkstra Result (TC2) ---" << endl;
    printSolution(dist, src, V);
}

// Overload for 6x6 matrix (TC1)
void parallelDijkstra(int graph[][V_TC1], int src) {
    int V = V_TC1;
    vector<int> dist(V, INT_MAX);
    vector<bool> visited(V, false);
    dist[src] = 0;
    for (int count = 0; count < V - 1; count++) {
        int u = minDistance(dist, visited, V);
        if (u == -1) break;
        visited[u] = true;
        #pragma omp parallel for
        for (int v = 0; v < V; v++) {
            if (!visited[v] && graph[u][v] && dist[u] != INT_MAX &&
                dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }
    cout << "--- Parallel Dijkstra Result (TC1) ---" << endl;
    printSolution(dist, src, V);
}

// Overload for 4x4 matrix (TC2)
void parallelDijkstra(int graph[][V_TC2], int src) {
    int V = V_TC2;
    vector<int> dist(V, INT_MAX);
    vector<bool> visited(V, false);
    dist[src] = 0;
    for (int count = 0; count < V - 1; count++) {
        int u = minDistance(dist, visited, V);
        if (u == -1) break;
        visited[u] = true;
        #pragma omp parallel for
        for (int v = 0; v < V; v++) {
            if (!visited[v] && graph[u][v] && dist[u] != INT_MAX &&
                dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }
    cout << "--- (INCORRECT) Parallel Dijkstra Result (TC2) ---" << endl;
    printSolution(dist, src, V);
}


// --- Bellman-Ford Algorithm (Correct for Negative Weights) ---

void serialBellmanFord(int graph[][V_TC2], int src) {
    vector<int> dist(V_TC2, INT_MAX);
    dist[src] = 0;

    for (int i = 1; i <= V_TC2 - 1; i++) {
        for (int u = 0; u < V_TC2; u++) {
            for (int v = 0; v < V_TC2; v++) {
                int weight = graph[u][v];
                if (weight != 0 && dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                }
            }
        }
    }
    cout << "--- (CORRECT) Serial Bellman-Ford Result (TC2) ---" << endl;
    printSolution(dist, src, V_TC2);
}

void parallelBellmanFord(int graph[][V_TC2], int src) {
    vector<int> dist(V_TC2, INT_MAX);
    dist[src] = 0;

    for (int i = 1; i <= V_TC2 - 1; i++) {
        #pragma omp parallel for
        for (int u = 0; u < V_TC2; u++) {
            for (int v = 0; v < V_TC2; v++) {
                int weight = graph[u][v];
                if (weight != 0 && dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                    #pragma omp critical
                    {
                        if (dist[u] + weight < dist[v]) {
                           dist[v] = dist[u] + weight;
                        }
                    }
                }
            }
        }
    }

    for (int u = 0; u < V_TC2; u++) {
        for (int v = 0; v < V_TC2; v++) {
            int weight = graph[u][v];
            if (weight != 0 && dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                cout << "Graph contains a negative-weight cycle!" << endl;
                return;
            }
        }
    }
    
    cout << "--- (CORRECT) Parallel Bellman-Ford Result (TC2) ---" << endl;
    printSolution(dist, src, V_TC2);
}


int main() {
    // --- Test Case 1: Positive Weights ---
    int graph1[V_TC1][V_TC1] = {
        {0, 2, 4, 0, 0, 0}, {0, 0, 1, 7, 0, 0}, {0, 0, 0, 0, 3, 0},
        {0, 0, 0, 0, 0, 1}, {0, 0, 0, 2, 0, 5}, {0, 0, 0, 0, 0, 0}
    };
    int startNode1 = 0; // 'A'

    cout << "====== Q1: Test Case 1 (Positive Weights) ======" << endl;
    auto startSerial1 = chrono::high_resolution_clock::now();
    serialDijkstra(graph1, startNode1);
    auto endSerial1 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> serialTime1 = endSerial1 - startSerial1;

    auto startParallel1 = chrono::high_resolution_clock::now();
    parallelDijkstra(graph1, startNode1);
    auto endParallel1 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> parallelTime1 = endParallel1 - startParallel1;


    // --- Test Case 2: Negative Weights ---
    int graph2[V_TC2][V_TC2] = {
        {0, 5, 2, 0}, {0, 0, -4, 0}, {0, 0, 0, 1}, {0, 0, 0, 0}
    };
    int startNode2 = 0; // 'S'

    cout << "\n====== Q1: Test Case 2 (Negative Weights) ======" << endl;

    auto startSerialD = chrono::high_resolution_clock::now();
    serialDijkstra(graph2, startNode2);
    auto endSerialD = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> serialTimeD = endSerialD - startSerialD;

    auto startParallelD = chrono::high_resolution_clock::now();
    parallelDijkstra(graph2, startNode2);
    auto endParallelD = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> parallelTimeD = endParallelD - startParallelD;

    auto startSerialB = chrono::high_resolution_clock::now();
    serialBellmanFord(graph2, startNode2);
    auto endSerialB = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> serialTimeB = endSerialB - startSerialB;

    auto startParallelB = chrono::high_resolution_clock::now();
    parallelBellmanFord(graph2, startNode2);
    auto endParallelB = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> parallelTimeB = endParallelB - startParallelB;

    
    // --- Output Tables ---
    cout << "\n--- Comparison Table (Q1) ---" << endl;
    cout << setfill('-') << setw(65) << "" << setfill(' ') << endl;
    cout << left << setw(35) << "Algorithm" 
         << setw(20) << "Serial Code (ms)" 
         << setw(20) << "Parallel Code (ms)" << endl;
    cout << setfill('-') << setw(65) << "" << setfill(' ') << endl;
    cout << left << setw(35) << "TC 1: (6, 8) [Dijkstra]" 
         << setw(20) << serialTime1.count() 
         << setw(20) << parallelTime1.count() << endl;
    cout << left << setw(35) << "TC 2: (4, 4) [Bellman-Ford]" 
         << setw(20) << serialTimeB.count() 
         << setw(20) << parallelTimeB.count() << endl;
    cout << left << setw(35) << "TC 2: (4, 4) [Dijkstra-INCORRECT]"
         << setw(20) << serialTimeD.count()
         << setw(20) << parallelTimeD.count() << endl;
    cout << setfill('-') << setw(65) << "" << setfill(' ') << endl;

    return 0;
}
