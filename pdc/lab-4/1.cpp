#include <iostream>
#include <omp.h>
#include <math.h>
using namespace std;

double compute_pi_serial(int n)
{
    double area = 0;
    double dx = 1.0 / n;

    for (int i = 0; i < n; ++i)
    {
        double x = i * dx;
        area += 4.0 / (1.0 + (x * x));
    }

    return area * dx;
}

double compute_pi_parallel_reduction(int n, double c)
{
    double area = 0;
    double dx = 1.0 / n;

#pragma omp parallel for reduction(+ : area)
    for (int i = 0; i < n; ++i)
    {
        double x = (i + c) * dx;
        area += 4.0 / (1.0 + (x * x));
    }

    return area * dx;
}

double compute_pi_parallel_atomic(int n, double c)
{
    double area = 0;
    double dx = 1.0 / n;

#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        double x = (i + c) * dx;
        double temp = 4.0 / (1.0 + (x * x));
#pragma omp atomic
        area += temp;
    }

    return area * dx;
}

double compute_pi_parallel_critical(int n, double c)
{
    double area = 0;
    double dx = 1.0 / n;

#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        double x = (i + c) * dx;
        double temp = 4.0 / (1.0 + (x * x));
#pragma omp critical
        area += temp;
    }

    return area * dx;
}

int main()
{
    int n = 100000;
    double start, end;
    double start_all, end_all;
    start_all = omp_get_wtime();
    cout << "Serial Execution\n";
    start = omp_get_wtime();
    double serial_pi = compute_pi_serial(n);
    end = omp_get_wtime();

    cout << "- PI Estimate: " << serial_pi << endl;
    cout << "- Execution time: " << end - start << endl;
    cout << endl;

    for (double c : {0.0, 1.0, 0.5}) {
        cout << "Parallel Reduction (c = " << c << ")\n";
        start = omp_get_wtime();
        double pi = compute_pi_parallel_reduction(n, c);
        end = omp_get_wtime();
        cout << "- PI Estimate: " << pi << endl;
        cout << "- Error (A - E): " << M_PI - pi << endl;
        cout << "- Execution time: " << end - start << endl
             << endl;
    }

    for (double c : {0.0, 1.0, 0.5}) {
        cout << "Parallel Atomic (c = " << c << ")\n";
        start = omp_get_wtime();
        double pi = compute_pi_parallel_atomic(n, c);
        end = omp_get_wtime();
        cout << "- PI Estimate: " << pi << endl;
        cout << "- Error (A - E): " << M_PI - pi << endl;
        cout << "- Execution time: " << end - start << endl
             << endl;
    }

    for (double c : {0.0, 1.0, 0.5}) {
        cout << "Parallel Critical (c = " << c << ")\n";
        start = omp_get_wtime();
        double pi = compute_pi_parallel_critical(n, c);
        end = omp_get_wtime();
        cout << "- PI Estimate: " << pi << endl;
        cout << "- Error (A - E): " << M_PI - pi << endl;
        cout << "- Execution time: " << end - start << endl
             << endl;
    }

    cout << "Actual PI: " << M_PI << endl;
    end_all = omp_get_wtime();
    cout << "Total time: " << end_all - start_all;
}