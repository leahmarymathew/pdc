#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define MAX 100 

int A[MAX][MAX], B[MAX][MAX], C[MAX][MAX];
int r1, c1, r2, c2;

typedef struct {
    int row;
} ThreadData;

void* multiply_row(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int row = data->row;

    for (int j = 0; j < c2; j++) {
        C[row][j] = 0;
        for (int k = 0; k < c1; k++) {
            C[row][j] += A[row][k] * B[k][j];
        }
    }

    pthread_exit(NULL);
}

int main() {
    pthread_t threads[MAX];
    ThreadData threadData[MAX];

    printf("Enter rows and columns of Matrix A: ");
    scanf("%d %d", &r1, &c1);

    printf("Enter rows and columns of Matrix B: ");
    scanf("%d %d", &r2, &c2);

    if (c1 != r2) {
        printf("Error: Incompatible matrices for multiplication.\n");
        return 1;
    }

    printf("Enter elements of Matrix A (%d x %d):\n", r1, c1);
    for (int i = 0; i < r1; i++)
        for (int j = 0; j < c1; j++)
            scanf("%d", &A[i][j]);

    printf("Enter elements of Matrix B (%d x %d):\n", r2, c2);
    for (int i = 0; i < r2; i++)
        for (int j = 0; j < c2; j++)
            scanf("%d", &B[i][j]);

    clock_t start_time, end_time;

    start_time = clock();
    for (int i = 0; i < r1; i++) {
        threadData[i].row = i;
        pthread_create(&threads[i], NULL, multiply_row, &threadData[i]);
    }

    for (int i = 0; i < r1; i++) {
        pthread_join(threads[i], NULL);
    }

    end_time = clock();
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Resultant Matrix C (%d x %d):\n", r1, c2);
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    printf("Total time taken: %.6f seconds\n", time_taken);

    return 0;
}
