#include <iostream>
#include <omp.h>
#include <cmath>
using namespace std;

double compute_pi_series(int n) {
    double sum = 0.0;
    double dx = 1.0 / n;

    for (int i = 0; i < n; i++) {
        double x = i * dx;
        sum += 4.0 / (1.0 + x * x);
    }

    return sum * dx;
}

double compute_pi_parallel(int n) {
    double sum = 0.0;
    double dx = 1.0 / n;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double x = i * dx;
        sum += 4.0 / (1.0 + x * x);
    }

    return sum * dx;
}

double compute_pi_reduction(int n) {
    double sum = 0.0;
    double dx = 1.0 / n;

    int num_threads = 0;
    #pragma omp parallel
    {
    #pragma omp single
        {
            num_threads = omp_get_num_threads();
            cout << "Number of threads: " << num_threads << endl;
        }
    }

    #pragma omp parallel reduction(+:sum)
    {
        #pragma omp for
        for (int i = 0; i < n; i++) {
            double x = i * dx;
            sum += 4.0 / (1.0 + x * x);
        }
    }

    return sum * dx;
}

int main() {
    int n = 100000;
    double start, end;

    cout << "Series\n";
    start = omp_get_wtime();
    double pi_series = compute_pi_series(n); 
    end = omp_get_wtime();
    cout << "Pi estimate: " << pi_series << endl;
    cout << "Time taken: " << end - start << " seconds" << endl;

    cout << "Parallel (Race Condition)\n";
    start = omp_get_wtime();
    double pi_race = compute_pi_parallel(n);
    end = omp_get_wtime();
    cout << "Pi estimate " << pi_race << endl;
    cout << "Time taken: " << end - start << " seconds" << endl;

    cout << "Parallel (fixed)\n";
    start = omp_get_wtime();
    double pi_reduction = compute_pi_reduction(n);
    end = omp_get_wtime();
    cout << "Pi estimate " << pi_reduction << endl;
    cout << "Time taken: " << end - start << endl;
    cout << endl;

    cout << "Actual Pi: " << M_PI << endl;
    return 0;
}