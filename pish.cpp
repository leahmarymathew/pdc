#include <iostream>
#include <iomanip> // For setprecision
#include <omp.h>   // OpenMP library
#include <cmath>   // For math functions
#include <vector>  // For Q2
#include <string>  // For Q2

using namespace std;

// --- Question 1: Pi Estimation ---

// This is the function we are integrating: 4 / (1 + x*x)
double f(double x) {
    return 4.0 / (1.0 + x * x);
}

// 1. (a) Serial Version
void q1a_serial_pi(long num_steps) {
    cout << "--- Q1a: Serial Pi Calculation ---" << endl;
    double step = 1.0 / (double)num_steps;
    cout<<step<<endl;
    double sum = 0.0;

    double start_time = omp_get_wtime();
    
    // Rectangle rule: use the midpoint of each interval
    for (long i = 0; i < num_steps; ++i) {
        double x = (i + 0.5) * step;
        sum += f(x);
        cout<<"s"<<f(x)*step<<endl;
    }
    
    double pi = step * sum;
    double end_time = omp_get_wtime();

    cout << "Calculated Pi: " << setprecision(10) << pi << endl;
    cout << "Execution Time: " << (end_time - start_time) << " s" << endl;
}

// 1. (b) Parallel Version (with Race Condition)
void q1b_parallel_pi_race(long num_steps) {
    cout << "\n--- Q1b: Parallel Pi (with Race Condition) ---" << endl;
    double step = 1.0 / (double)num_steps;
    double sum = 0.0; // Shared variable, this will cause the race condition

    double start_time = omp_get_wtime();
    
    #pragma omp parallel
    {
        // Print thread info
        #pragma omp master
        {
            cout << "Parallel region running with " << omp_get_num_threads() << " threads." << endl;
        }

        // Each thread calculates a partial sum, but all write to the *same* 'sum'
        // This is the race condition.
        #pragma omp for
        for (long i = 0; i < num_steps; ++i) {
            double x = (i + 0.5) * step;
            // RACE CONDITION: Multiple threads read/write 'sum' simultaneously
            sum += f(x); 
        }
    }
    
    double pi = step * sum;
    double end_time = omp_get_wtime();

    cout << "Calculated Pi: " << setprecision(10) << pi << " (Note: Likely incorrect!)" << endl;
    cout << "Execution Time: " << (end_time - start_time) << " s" << endl;
}

// 1. (c) Parallel Version (Fixed with 'reduction' clause)
void q1c_parallel_pi_fixed(long num_steps) {
    cout << "\n--- Q1c: Parallel Pi (Fixed with 'reduction') ---" << endl;
    double step = 1.0 / (double)num_steps;
    double sum = 0.0;

    double start_time = omp_get_wtime();
    
    // Use the 'reduction' clause to fix the race condition
    // OpenMP creates a private 'sum' for each thread,
    // then safely adds them all together at the end.
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < num_steps; ++i) {
        double x = (i + 0.5) * step;
        sum += f(x); // Each thread adds to its private 'sum'
    }
    
    double pi = step * sum;
    double end_time = omp_get_wtime();

    cout << "Calculated Pi: " << setprecision(10) << pi << " (Correct)" << endl;
    cout << "Execution Time: " << (end_time - start_time) << " s" << endl;
}


// --- Question 2: Loop Scheduling ---

// 2. (a) Serial Sum
void q2a_serial_sum(long N) {
    cout << "\n--- Q2a: Serial Sum (N=" << N << ") ---" << endl;
    long long sum = 0;
    double start_time = omp_get_wtime();

    for (long i = 0; i < N; ++i) {
        sum += (i + 1); // Sum 1 to N
    }

    double end_time = omp_get_wtime();
    cout << "Total Sum: " << sum << endl;
    cout << "Execution Time: " << (end_time - start_time) << " s" << endl;
}

// Helper function to run and demonstrate schedules
// We use a vector of strings to record which thread did which work,
// as printing from inside the loop is messy and can be misleading.
void run_schedule_demo(long N, string type, int chunk) {
    long long sum = 0;
    // Get max threads and create a vector to hold work logs
    int max_threads = omp_get_max_threads();
    vector<string> thread_work(max_threads);

    if (type == "static") {
        cout << "\n--- Q2b: schedule(static, " << chunk << ") ---" << endl;
        #pragma omp parallel for reduction(+:sum) schedule(static, chunk)
        for (long i = 0; i < N; ++i) {
            sum += (i + 1);
            int tid = omp_get_thread_num();
            thread_work[tid] += to_string(i) + " ";
        }
    } else if (type == "dynamic") {
        cout << "\n--- Q2c: schedule(dynamic, " << chunk << ") ---" << endl;
        #pragma omp parallel for reduction(+:sum) schedule(dynamic, chunk)
        for (long i = 0; i < N; ++i) {
            sum += (i + 1);
            int tid = omp_get_thread_num();
            thread_work[tid] += to_string(i) + " ";
        }
    } else if (type == "guided") {
        cout << "\n--- Q2d: schedule(guided, " << chunk << ") ---" << endl;
        #pragma omp parallel for reduction(+:sum) schedule(guided, chunk)
        for (long i = 0; i < N; ++i) {
            sum += (i + 1);
            int tid = omp_get_thread_num();
            thread_work[tid] += to_string(i) + " ";
        }
    }

    // Print the work distribution
    cout << "Work Distribution (N=" << N << ", Chunk=" << chunk << "):" << endl;
    for (int i = 0; i < max_threads; ++i) {
        if (!thread_work[i].empty()) {
            cout << "Thread " << i << " did iterations: " << thread_work[i] << endl;
        }
    }
    cout << "Total Sum: " << sum << endl;
}


int main() {
    // --- Question 1 ---
    long num_steps = 16; // 10 million steps
    cout << "====== Question 1: Pi Estimation (Steps=" << num_steps << ") ======" << endl;
    q1a_serial_pi(num_steps);
    q1b_parallel_pi_race(num_steps);
    q1c_parallel_pi_fixed(num_steps);
    cout << "\n(1c) Race Condition Explanation:" << endl;
    cout << "The line 'sum += f(x);' in (1b) is the race condition." << endl;
    cout << "Multiple threads try to read 'sum', add their value, and write 'sum' at the same time, leading to lost updates." << endl;
    cout << "This was handled in (1c) using the 'reduction(+:sum)' clause, which is the most appropriate solution for parallel sums." << endl;

    // --- Question 2 ---
    long N = 40; // Small N to make schedule logs readable
    int chunk_size = 4;
    cout << "\n\n====== Question 2: Loop Scheduling (N=" << N << ", Chunk=" << chunk_size << ") ======" << endl;
    q2a_serial_sum(N);
    
    // 
    run_schedule_demo(N, "static", chunk_size);
    
    // 
    run_schedule_demo(N, "dynamic", chunk_size);
    
    // 
    run_schedule_demo(N, "guided", chunk_size);

    return 0;
}
