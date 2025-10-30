#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, size;
    int a = 10, b = 5, c;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        c = a + b;
        printf("Process %d: c = a + b = %d + %d = %d\n", rank, a, b, c);
    }
    else if (rank == 1)
    {
        c = a - b;
        printf("Process %d: c = a - b = %d - %d = %d\n", rank, a, b, c);
    }
    else
    {
        c = a * b;
        printf("Process %d: c = a * b = %d * %d = %d\n", rank, a, b, c);
    }

    MPI_Finalize();

    return 0;
}
