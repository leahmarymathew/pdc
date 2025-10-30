#include <stdio.h>
#include <limits.h>
#include <omp.h>

#define V 5
#define INF 9999

int minDistance(int dist[], int visited[])
{
    int min = INF, min_index;
    for (int v = 0; v < V; v++)
        if (!visited[v] && dist[v] <= min)
            min = dist[v], min_index = v;
    return min_index;
}

void dijkstraSerial(int graph[V][V], int src)
{
    double start_time = omp_get_wtime();

    int dist[V], visited[V];
    for (int i = 0; i < V; i++)
        dist[i] = INF, visited[i] = 0;

    dist[src] = 0;

    for (int count = 0; count < V - 1; count++)
    {
        int u = minDistance(dist, visited);
        visited[u] = 1;

        for (int v = 0; v < V; v++)
            if (!visited[v] && graph[u][v] != 0 && dist[u] + graph[u][v] < dist[v])
                dist[v] = dist[u] + graph[u][v];
    }

    double end_time = omp_get_wtime();

    printf("\nSerial Shortest Distances:\n");
    char nodes[] = {'S', 'A', 'B', 'C', 'D'};
    for (int i = 0; i < V; i++)
        printf("%c -> %c = %d\n", nodes[src], nodes[i], dist[i]);
    printf("Serial Execution Time: %f seconds\n", end_time - start_time);
}

void dijkstraParallel(int graph[V][V], int src)
{
    double start_time = omp_get_wtime();

    int dist[V], visited[V];
    for (int i = 0; i < V; i++)
        dist[i] = INF, visited[i] = 0;

    dist[src] = 0;

    for (int count = 0; count < V - 1; count++)
    {
        int u = minDistance(dist, visited);
        visited[u] = 1;

#pragma omp parallel for shared(dist, visited)
        for (int v = 0; v < V; v++)
        {
            if (!visited[v] && graph[u][v] != 0 && dist[u] + graph[u][v] < dist[v])
            {
#pragma omp critical
                {
                    if (dist[u] + graph[u][v] < dist[v])
                        dist[v] = dist[u] + graph[u][v];
                }
            }
        }
    }

    double end_time = omp_get_wtime();

    printf("\nParallel Shortest Distances:\n");
    char nodes[] = {'S', 'A', 'B', 'C', 'D'};
    for (int i = 0; i < V; i++)
        printf("%c -> %c = %d\n", nodes[src], nodes[i], dist[i]);
    printf("Parallel Execution Time: %f seconds\n", end_time - start_time);
}

int main()
{
    int graph[V][V] = {
        {0, 5, 0, -2, 0},
        {0, 0, 1, 2, 4}, 
        {0, 0, 0, 0, 3},
        {0, 0, 0, 0, 4}, 
        {0, 0, 0, 0, 0}  
    };

    printf("Graph representation with NEGATIVE WEIGHTS:\n");
    printf("S=0, A=1, B=2, C=3, D=4\n");
    printf("Starting from node S (index 0)\n");

    dijkstraSerial(graph, 0);
    dijkstraParallel(graph, 0);

    return 0;
}
