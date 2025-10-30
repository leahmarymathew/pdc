#include <iostream>
#include <omp.h>
using namespace std;

int main()
{
#pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            cout << "Number of threads = " << omp_get_num_threads() << endl;
        }
    }
    return 0;
}
