#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

void new_fork(double x) {
    pid_t p;
    p = fork();

    if (p == -1) {
        printf("fork failed\n");
        return;
    } else if (p == 0) {
        printf("Child process: PID = %d\n", getpid());
        printf("Square of the input: %lf\n", x * x);
        printf("\n");
    } else {
        printf("Parent process: PID = %d\n", getpid());
        printf("Square of the input: %lf\n", x * x);
        printf("\n");
    }
}

int main() {
    double x;
    printf("Give a number: ");
    scanf("%lf", &x);
    new_fork(x);
}