#include <iostream>
#include <omp.h>
using namespace std;

int main()
{
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        cout << tid << ": Welcome to CSS 311 Course\n";
    }
    return 0;
}
