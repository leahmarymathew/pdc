#include <stdio.h>
#include <time.h>

int main() {
    int n;
    printf("Enter array size: ");
    scanf("%d", &n);

    printf("Enter elements: \n");
    double nums[n];
    for (int i = 0; i < n; ++i) {
        scanf("%lf", &nums[i]);
    }

    clock_t start_time, end_time;

    start_time = clock();

    double sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += nums[i];
    }

    end_time = clock();
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Sum = %lf\n", sum);
    printf("Total time taken: %.6f seconds\n", time_taken);

    return 0;
}