#include <iostream>
#include <vector>
#include <omp.h>
#include <limits>
using namespace std;

#define INF 99999

int main()
{
    int n = 4;
    vector<vector<int>> dist = {
        {0, 3, INF, 5},
        {2, 0, INF, 4},
        {INF, 1, 0, INF},
        {INF, INF, 2, 0}};

    double start = omp_get_wtime();

    for (int k = 0; k < n; k++)
    {
#pragma omp parallel for collapse(2) shared(dist, k)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (dist[i][k] + dist[k][j] < dist[i][j])
                {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    double end = omp_get_wtime();

    cout << "All-pairs shortest paths:\n";
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (dist[i][j] == INF)
                cout << "INF ";
            else
                cout << dist[i][j] << " ";
        }
        cout << endl;
    }

    cout << "\nExecution Time: " << end - start << " seconds\n";
    return 0;
}
