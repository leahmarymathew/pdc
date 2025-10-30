#include <iostream>
#include <omp.h>
using namespace std;

int main()
{
    int arr[100][100][100][100] = {0};

#pragma omp parallel for collapse(2)
    for (int i = 0; i < 100; i++)
    {
        for (int j = 0; j < 100; j++)
        {
            for (int k = 10; k < 100; k++)
            {
                for (int l = 10; l < 100; l++)
                {
                    arr[i][j][k][l] = i + (j * 2) + arr[i][j][k - 10][l];
                }
            }
        }
    }

    cout << "arr[0][0][10][10] = %d\n", arr[0][0][10][10];
    cout << "arr[5][5][20][20] = %d\n", arr[5][5][20][20];

    return 0;
}
