#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>

int main() {
    int n;

    printf("Enter the number of elements in the array: ");
    scanf("%d", &n);

    int arr[n];

    printf("Enter %d integers:\n", n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    int fd[2];
    if (pipe(fd) == -1) {
        perror("Pipe failed");
        return 1;
    }

    clock_t start_time, end_time;

    start_time = clock();

    pid_t p = fork();

    if (p < 0) {
        perror("Fork failed");
        return 1;
    }

    if (p == 0) {
        close(fd[0]);

        int child_sum = 0;
        for (int i = n / 2; i < n; i++) {
            child_sum += arr[i];
        }

        write(fd[1], &child_sum, sizeof(child_sum));

        close(fd[1]);
        exit(0);
    } else {
        close(fd[1]);

        int parent_sum = 0;
        for (int i = 0; i < n / 2; i++) {
            parent_sum += arr[i];
        }

        int child_sum;
        read(fd[0], &child_sum, sizeof(child_sum));

        close(fd[0]);

        wait(NULL);

        int total_sum = parent_sum + child_sum;

        end_time = clock();
        double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        printf("Parent sum (first half): %d\n", parent_sum);
        printf("Child sum (second half): %d\n", child_sum);
        printf("Total sum: %d\n", total_sum);
        printf("Total time taken: %.6f seconds\n", time_taken);
    }

    return 0;
}
