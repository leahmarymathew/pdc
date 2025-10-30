#include <stdio.h>
#include <omp.h>
#include <mpi.h>

#define INF 9999
#define V 5 // Changed to 5 vertices (1, 2, 3, 4, 5)

int min(int a, int b)
{
    return (a < b) ? a : b;
}

void floydSerial(int graph[V][V])
{
    printf("Starting Serial Floyd-Warshall...\n");
    double start_time = omp_get_wtime();

    int dist[V][V];

    // Initialize distance matrix
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            dist[i][j] = graph[i][j];

    // Floyd-Warshall algorithm
    for (int k = 0; k < V; k++)
        for (int i = 0; i < V; i++)
            for (int j = 0; j < V; j++)
                if (dist[i][k] != INF && dist[k][j] != INF &&
                    dist[i][j] > dist[i][k] + dist[k][j])
                    dist[i][j] = dist[i][k] + dist[k][j];

    double end_time = omp_get_wtime();

    printf("\nSerial Floyd-Warshall Output:\n");
    printf("   ");
    for (int j = 0; j < V; j++)
        printf("    %d", j + 1);
    printf("\n");

    for (int i = 0; i < V; i++)
    {
        printf("%d: ", i + 1);
        for (int j = 0; j < V; j++)
        {
            if (dist[i][j] == INF)
                printf("  INF");
            else
                printf("%5d", dist[i][j]);
        }
        printf("\n");
    }
    printf("Serial Execution Time: %f seconds\n\n", end_time - start_time);
}

void floydParallelMPI(int graph[V][V])
{
    int rank, size;
    double start_time, end_time;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        printf("Starting Parallel Floyd-Warshall with MPI (%d processes)...\n", size);
        start_time = MPI_Wtime();
    }

    int dist[V][V];

    // Initialize distance matrix
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            dist[i][j] = graph[i][j];

    int rows_per_proc = V / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1) ? V : (rank + 1) * rows_per_proc;

    // Floyd-Warshall algorithm with MPI
    for (int k = 0; k < V; k++)
    {
        // Broadcast k-th row to all processes
        int owner = k / rows_per_proc;
        if (owner >= size)
            owner = size - 1; // Handle edge case

        MPI_Bcast(dist[k], V, MPI_INT, owner, MPI_COMM_WORLD);

        // Each process updates its assigned rows
        for (int i = start_row; i < end_row; i++)
        {
            for (int j = 0; j < V; j++)
            {
                if (dist[i][k] != INF && dist[k][j] != INF)
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
            }
        }
    }

    // Gather results at root process
    if (rank == 0)
    {
        for (int p = 1; p < size; p++)
        {
            int p_start = p * rows_per_proc;
            int p_end = (p == size - 1) ? V : (p + 1) * rows_per_proc;
            int p_rows = p_end - p_start;

            MPI_Recv(&dist[p_start][0], p_rows * V, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        end_time = MPI_Wtime();

        printf("\nParallel Floyd-Warshall Output (MPI):\n");
        printf("   ");
        for (int j = 0; j < V; j++)
            printf("    %d", j + 1);
        printf("\n");

        for (int i = 0; i < V; i++)
        {
            printf("%d: ", i + 1);
            for (int j = 0; j < V; j++)
            {
                if (dist[i][j] == INF)
                    printf("  INF");
                else
                    printf("%5d", dist[i][j]);
            }
            printf("\n");
        }
        printf("Parallel Execution Time: %f seconds\n", end_time - start_time);
    }
    else
    {
        int my_rows = end_row - start_row;
        MPI_Send(&dist[start_row][0], my_rows * V, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char *argv[])
{
    // Adjacency matrix based on Test Case 2 with negative weights
    // Vertices: 1=0, 2=1, 3=2, 4=3, 5=4
    int graph[V][V] = {
        {0, 3, 8, INF, -4},    // 1 -> 2(3), 3(8), 5(-4)
        {INF, 0, INF, 1, 7},   // 2 -> 4(1), 5(7)
        {INF, 4, 0, INF, INF}, // 3 -> 2(4)
        {2, INF, -5, 0, INF},  // 4 -> 1(2), 3(-5)
        {INF, INF, INF, 6, 0}  // 5 -> 4(6)
    };

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        printf("Graph representation with NEGATIVE WEIGHTS (Vertices 1-5):\n");
        printf("Edges: 1->2(3), 1->3(8), 1->5(-4), 2->4(1), 2->5(7), 3->2(4), 4->1(2), 4->3(-5), 5->4(6)\n");
        printf("NOTE: Floyd-Warshall can handle negative weights (unlike Dijkstra)\n\n");

        // Run serial version only on process 0
        floydSerial(graph);
    }

    // Run MPI parallel version
    floydParallelMPI(graph);

    MPI_Finalize();
    return 0;
}