#include <stdio.h>
#include <stdlib.h>     // For exit(), malloc(), free()
#include <unistd.h>     // For fork(), pipe(), close()
#include <sys/types.h>  // For pid_t
#include <sys/wait.h>   // For wait()
#include <time.h>       // For clock()
#include <pthread.h>    // For pthreads


/*
 * --- Question 1: fork() and square ---
 */
void q1() {
    int n = 7;
    pid_t pid = fork();


    if (pid < 0) {
        fprintf(stderr, "Fork failed\n");
        return;
    } else if (pid == 0) {
        // Child process
        printf("Child Process:\n");
        printf("  PID: %d\n", getpid());
        printf("  Parent PID: %d\n", getppid());
        printf("  Square of %d is %d\n", n, n * n);
    } else {
        // Parent process
        printf("Parent Process:\n");
        printf("  PID: %d\n", getpid());
        printf("  Child PID: %d\n", pid);
        printf("  Square of %d is %d\n", n, n * n);
        wait(NULL); // Wait for the child to finish
    }
}


/*
 * --- Question 2: fork(), pipe(), and array sum ---
 */
void q2() {
    int n, i, parent_sum = 0, child_sum = 0, total_sum = 0;
    int fd[2]; // Pipe file descriptors
    pid_t pid;
    clock_t start, end;
    double cpu_time_used;


    printf("Enter the number of elements: ");
    scanf("%d", &n);
    int* arr = (int*)malloc(n * sizeof(int));
    if (arr == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    printf("Enter %d elements:\n", n);
    for (i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }


    if (pipe(fd) == -1) {
        fprintf(stderr, "Pipe failed\n");
        free(arr);
        return;
    }


    start = clock();
    pid = fork();


    if (pid < 0) {
        fprintf(stderr, "Fork failed\n");
        free(arr);
        return;
    } else if (pid == 0) {
        // Child process: Sums the second half
        close(fd[0]); // Close unused read end
        int mid = n / 2;
        for (i = mid; i < n; i++) {
            child_sum += arr[i];
        }
        write(fd[1], &child_sum, sizeof(child_sum));
        close(fd[1]); // Close write end
        free(arr);
        exit(0);
    } else {
        // Parent process: Sums the first half
        close(fd[1]); // Close unused write end
        int mid = n / 2;
        for (i = 0; i < mid; i++) {
            parent_sum += arr[i];
        }
       
        read(fd[0], &child_sum, sizeof(child_sum)); // Read sum from child
        total_sum = parent_sum + child_sum;
       
        wait(NULL); // Wait for child to exit
        end = clock();
        close(fd[0]); // Close read end


        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;


        printf("\nParent Sum (First Half): %d\n", parent_sum);
        printf("Child Sum (Second Half): %d\n", child_sum);
        printf("Total Sum: %d\n", total_sum);
        printf("Execution Time: %f seconds\n", cpu_time_used);
        free(arr);
    }
}


/*
 * --- Question 3: Matrix multiplication with pthreads ---
 */
#define MAX_DIM 10 // Max matrix dimension


int A[MAX_DIM][MAX_DIM];
int B[MAX_DIM][MAX_DIM];
int C[MAX_DIM][MAX_DIM];
int r1, c1, r2, c2;


struct ThreadArgs {
    int row;
};


void* multiply_row(void* arg) {
    struct ThreadArgs* args = (struct ThreadArgs*)arg;
    int i = args->row;


    for (int j = 0; j < c2; j++) {
        C[i][j] = 0;
        for (int k = 0; k < c1; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
    free(arg);
    pthread_exit(NULL);
}


void q3() {
    printf("Enter dimensions of Matrix A (rows cols): ");
    scanf("%d %d", &r1, &c1);
    printf("Enter dimensions of Matrix B (rows cols): ");
    scanf("%d %d", &r2, &c2);


    if (c1 != r2) {
        printf("Matrix multiplication not possible (c1 != r2).\n");
        return;
    }
    if (r1 > MAX_DIM || c1 > MAX_DIM || r2 > MAX_DIM || c2 > MAX_DIM) {
        printf("Dimensions exceed MAX_DIM (%d)\n", MAX_DIM);
        return;
    }


    printf("Enter elements of Matrix A (%d x %d):\n", r1, c1);
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c1; j++) {
            scanf("%d", &A[i][j]);
        }
    }


    printf("Enter elements of Matrix B (%d x %d):\n", r2, c2);
    for (int i = 0; i < r2; i++) {
        for (int j = 0; j < c2; j++) {
            scanf("%d", &B[i][j]);
        }
    }


    pthread_t threads[r1]; // One thread per result row
    for (int i = 0; i < r1; i++) {
        struct ThreadArgs* args = (struct ThreadArgs*)malloc(sizeof(struct ThreadArgs));
        args->row = i;
        if (pthread_create(&threads[i], NULL, multiply_row, (void*)args) != 0) {
            perror("pthread_create");
            free(args);
        }
    }


    for (int i = 0; i < r1; i++) {
        pthread_join(threads[i], NULL);
    }


    printf("\nResult Matrix C (%d x %d):\n", r1, c2);
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            printf("%d\t", C[i][j]);
        }
        printf("\n");
    }
}


/*
 * --- Question 4: Serial array sum with timing ---
 */
#define ARRAY_SIZE 100000000 // Large size for measurable time


void q4() {
    int* arr = (int*)malloc(ARRAY_SIZE * sizeof(int));
    if (arr == NULL) {
        printf("Memory allocation failed\n");
        return;
    }


    // Initialize the array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = i % 10 + 1; // Fill with numbers 1-10
    }


    long long sum = 0;
    clock_t start, end;
    double cpu_time_used;


    start = clock();
    for (int i = 0; i < ARRAY_SIZE; i++) {
        sum += arr[i];
    }
    end = clock();


    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;


    printf("\nSerial Sum: %lld\n", sum);
    printf("Total Elements: %d\n", ARRAY_SIZE);
    printf("Execution Time: %f seconds\n", cpu_time_used);


    free(arr);
}




/*
 * --- Main Function to Select Question ---
 */
int main() {
    int choice;
    printf("Choose a question to run (1-4): ");
    scanf("%d", &choice);


    switch (choice) {
        case 1:
            printf("\n--- Running Q1 ---\n");
            q1();
            break;
        case 2:
            printf("\n--- Running Q2 ---\n");
            q2();
            break;
        case 3:
            printf("\n--- Running Q3 ---\n");
            q3();
            break;
        case 4:
            printf("\n--- Running Q4 ---\n");
            q4();
            break;
        default:
            printf("Invalid choice.\n");
    }
    return 0;
}
