#include <iostream>
#include <omp.h>
using namespace std;

int main()
{
    double start, end;

    start = omp_get_wtime();
    for (int i = 0; i < 10; i++) {
        cout << "Sequential: Welcome to CSS 311 Course" << endl;
    }
    end = omp_get_wtime();
    cout << "Sequential execution time = " << end - start << " seconds\n" << endl;

    start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < 10; i++) {
        cout << "Parallel: Welcome to CSS 311 Course" << endl;
    }
    end = omp_get_wtime();
    cout << "Parallel execution time = " << end - start << " seconds" << endl;

    return 0;
}
