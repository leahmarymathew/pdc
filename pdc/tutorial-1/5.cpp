#include <stdio.h>
#include <omp.h>

int main()
{
    omp_set_num_threads(4);
    static int arr[4] = {1, 2, 3, 4};
    int x = 0, y = 0, j;

#pragma omp parallel for schedule(static, 1)
    for (j = 0; j < 4; j++)
    {
#pragma omp critical
        x += arr[j] * omp_get_thread_num();
    }

#pragma omp parallel for schedule(static, 2)
    for (j = 0; j < 4; j++)
    {
#pragma omp critical
        y += arr[j] * omp_get_thread_num();
    }

    printf("%d, %d\n", x, y);
    return 0;
}
