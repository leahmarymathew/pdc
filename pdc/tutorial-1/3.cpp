#include <stdio.h>
#include <omp.h>

int main()
{
    int s = 0;
    omp_set_num_threads(5);

#pragma omp parallel shared(s)
    {
#pragma omp single nowait
        s++;

#pragma omp master
        s++;

#pragma omp single
        printf("s = %d\n", s);
    }

    return 0;
}
