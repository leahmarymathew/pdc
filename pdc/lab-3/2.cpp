#include <iostream>
#include <omp.h>
using namespace std;

int main() {
    int N = 10; 
    long long serial_sum = 0;

    for (int i = 1; i <= N; i++) {
        serial_sum += i;
    }

    cout << "Serial sum: " << serial_sum << endl;

    long long parallel_sum = 0;
    int chunk = 2;

    parallel_sum = 0;
    #pragma omp parallel for reduction(+:parallel_sum) schedule(static)
    for (int i = 1; i <= N; i++) {
        parallel_sum += i;
        cout << "Thread " << omp_get_thread_num() << ": " << i << " (static)\n";
    }
    cout << "\nParallel sum with schedule(static) = " << parallel_sum << "\n\n";

    parallel_sum = 0;
    #pragma omp parallel for reduction(+:parallel_sum) schedule(static, chunk)
    for (int i = 1; i <= N; i++) {
        parallel_sum += i;
        cout << "Thread " << omp_get_thread_num() << ": " << i << " (static," << chunk << ")\n";
    }
    cout << "\nParallel sum with schedule(static," << chunk << ") = " << parallel_sum << "\n\n";

    parallel_sum = 0;
    #pragma omp parallel for reduction(+:parallel_sum) schedule(dynamic)
    for (int i = 1; i <= N; i++) {
        parallel_sum += i;
        cout << "Thread " << omp_get_thread_num() << ": " << i << " (dynamic)\n";
    }
    cout << "\nParallel sum with schedule(dynamic) = " << parallel_sum << "\n\n";

    parallel_sum = 0;
    #pragma omp parallel for reduction(+:parallel_sum) schedule(dynamic, chunk)
    for (int i = 1; i <= N; i++) {
        parallel_sum += i;
        cout << "Thread " << omp_get_thread_num() << ": " << i << " (dynamic," << chunk << ")\n";
    }
    cout << "\nParallel sum with schedule(dynamic," << chunk << ") = " << parallel_sum << "\n\n";

    parallel_sum = 0;
    #pragma omp parallel for reduction(+:parallel_sum) schedule(guided)
    for (int i = 1; i <= N; i++) {
        parallel_sum += i;
        cout << "Thread " << omp_get_thread_num() << ": " << i << " (guided)\n";
    }
    cout << "\nParallel sum with schedule(guided) = " << parallel_sum << "\n\n";

    parallel_sum = 0;
    #pragma omp parallel for reduction(+:parallel_sum) schedule(guided, chunk)
    for (int i = 1; i <= N; i++) {
        parallel_sum += i;
        cout << "Thread " << omp_get_thread_num() << ": " << i << " (guided," << chunk << ")\n";
    }
    cout << "\nParallel sum with schedule(guided," << chunk << ") = " << parallel_sum << "\n\n";

    return 0;
}
