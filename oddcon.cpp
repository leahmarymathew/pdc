#include <iostream>
#include <vector>
#include <omp.h>       // OpenMP library
#include <algorithm>   // For std::sort
#include <cmath>       // For sqrt (or i*i)

using namespace std;

/**
 * @brief Checks if a number is prime.
 * This is the "work" that will be parallelized.
 * @param n The number to check.
 * @return true if n is prime, false otherwise.
 */
bool isPrime(int n) {
    // 0 and 1 are not prime
    if (n <= 1) return false;
    
    // 2 is the only even prime
    if (n == 2) return true;
    
    // Other even numbers are not prime
    if (n % 2 == 0) return false;

    // Check odd divisors from 3 up to sqrt(n)
    // We only need to check odd numbers
    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) {
            return false;
        }
    }
    
    // If no divisors found, it's prime
    return true;
}

int main() {
    // We start at 3, as we only want ODD primes (skips 1 and 2)
    const int START_NUM = 3;
    const int MAX_NUM = 200;
    
    vector<int> oddPrimes;
    
    double start_time = omp_get_wtime();

    // Parallelize the 'for' loop
    // Each thread will take a different 'i' and check if it's prime
    #pragma omp parallel for
    for (int i = START_NUM; i <= MAX_NUM; i += 2) { // Increment by 2 to only check odd numbers
        
        if (isPrime(i)) {
            // --- RACE CONDITION ---
            // If two threads find a prime at the same time,
            // they will try to call oddPrimes.push_back() simultaneously.
            // This is a race condition and will corrupt the vector.
            
            // --- SOLUTION ---
            // Use 'critical' to ensure only one thread
            // can add an element to the vector at a time.
            #pragma omp critical
            {
                oddPrimes.push_back(i);
            }
        }
    }
    
    double end_time = omp_get_wtime();

    // --- Serial Post-Processing ---
    
    // Because threads finish out of order, the `oddPrimes` vector
    // will not be sorted. We sort it here to print consecutively.
    sort(oddPrimes.begin(), oddPrimes.end());

    // Print the results
    cout << "Odd consecutive prime numbers (3-" << MAX_NUM << "):" << endl;
    for (int prime : oddPrimes) {
        cout << prime << " ";
    }
    cout << endl; // Newline after list

    cout << "\nTotal count of odd primes: " << oddPrimes.size() << endl;
    cout << "Parallel execution time: " << (end_time - start_time) << " s" << endl;

    return 0;
}
