#include <iostream>
#include <random>
#include <omp.h>
using namespace std;
long long NUM_POINTS = 1000000;
void serial_pi()
{
    long long count_inside = 0;
    mt19937 gen(42);
    uniform_real_distribution<double> dist(0.0, 1.0);
    double start = omp_get_wtime();
    for (long long i = 0; i < NUM_POINTS; i++)
    {
        double x = dist(gen);
        double y = dist(gen);
        if (x * x + y * y <= 1.0)
            count_inside++;
    }
    double pi = 4.0 * count_inside / NUM_POINTS;
    double end = omp_get_wtime();
    cout << "[Serial] Pi Estimate = " << pi
         << " | Time = " << (end - start) << " s\n";
}
void parallel_pi_race()
{
    long long count_inside = 0;
    double start = omp_get_wtime();
#pragma omp parallel for
    for (long long i = 0; i < NUM_POINTS; i++)
    {
        unsigned int seed = omp_get_thread_num() + i;
        double x = rand_r(&seed) / (double)RAND_MAX;
        double y = rand_r(&seed) / (double)RAND_MAX;
        if (x * x + y * y <= 1.0)
        {
            count_inside++;
        }
    }
    double pi = 4.0 * count_inside / NUM_POINTS;
    double end = omp_get_wtime();
    cout << "[Parallel - Race] Pi Estimate = " << pi
         << " | Time = " << (end - start) << " s\n";
}
void parallel_pi_critical()
{
    long long count_inside = 0;
    double start = omp_get_wtime();
#pragma omp parallel for
    for (long long i = 0; i < NUM_POINTS; i++)
    {
        unsigned int seed = omp_get_thread_num() + i;
        double x = rand_r(&seed) / (double)RAND_MAX;
        double y = rand_r(&seed) / (double)RAND_MAX;
        if (x * x + y * y <= 1.0)
        {
#pragma omp critical
            count_inside++;
        }
    }
    cout << count_inside;
    double pi = 4.0 * count_inside / NUM_POINTS;
    double end = omp_get_wtime();
    cout << "[Parallel - Critical] Pi Estimate = " << pi
         << " | Time = " << (end - start) << " s\n";
}
void parallel_pi_atomic()
{
    long long count_inside = 0;
    double start = omp_get_wtime();
#pragma omp parallel for
    for (long long i = 0; i < NUM_POINTS; i++)
    {
        unsigned int seed = omp_get_thread_num() + i;
        double x = rand_r(&seed) / (double)RAND_MAX;
        double y = rand_r(&seed) / (double)RAND_MAX;
        if (x * x + y * y <= 1.0)
        {
#pragma omp atomic
            count_inside++;
        }
    }
    double pi = 4.0 * count_inside / NUM_POINTS;
    double end = omp_get_wtime();
    cout << "[Parallel - Atomic] Pi Estimate = " << pi
         << " | Time = " << (end - start) << " s\n";
}
void parallel_pi_reduction()
{
    long long count_inside = 0;
    double start = omp_get_wtime();
#pragma omp parallel for reduction(+ : count_inside)
    for (long long i = 0; i < NUM_POINTS; i++)
    {
        unsigned int seed = omp_get_thread_num() + i;
        double x = rand_r(&seed) / (double)RAND_MAX;
        double y = rand_r(&seed) / (double)RAND_MAX;
        if (x * x + y * y <= 1.0)
        {
            count_inside++;
        }
    }
    double pi = 4.0 * count_inside / NUM_POINTS;
    double end = omp_get_wtime();
    cout << "[Parallel - Reduction] Pi Estimate = " << pi
         << " | Time = " << (end - start) << " s\n";
}
int main()
{
    cout << "Estimating Pi using Monte Carlo Simulation\n";
    cout << "Number of Points = " << NUM_POINTS << "\n\n";
    serial_pi();
    parallel_pi_race();
    parallel_pi_critical();
    parallel_pi_atomic();
    parallel_pi_reduction();
}