#include <stdio.h>
#include <omp.h>
#include <iostream>
using namespace std;

int main()
{
    omp_set_num_threads(5);
    int t = omp_get_num_threads();

#pragma omp parallel
    {
        int x = omp_get_num_threads();
        int y = omp_get_thread_num();
        #pragma omp single
        t += x;
    }

    cout << t;
    return 0;
}
