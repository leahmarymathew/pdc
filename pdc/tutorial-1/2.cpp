#include <stdio.h>
#include <omp.h>

int main()
{
    int x[4];
    x[0] = x[1] = x[2] = x[3] = 0;

#pragma omp parallel
    {
        int i;
        int t = omp_get_thread_num();

#pragma omp for schedule(static, 10)
        for (i = 0; i < 100; i++)
        {
            x[t]++;
        }

        if (t == 3)
        {
            printf("x[%d] = %d\n", t, x[t]);
        }
    }

    return 0;
}
