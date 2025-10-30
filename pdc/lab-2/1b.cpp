#include <iostream>
#include <omp.h>
using namespace std;

int main() {
#pragma omp parallel
    {
        cout << "Welcome to CSS 311 Course" << endl;
    }
    return 0;
}
