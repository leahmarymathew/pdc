#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, size;
    int A[4] = {10, 85, 150, 42};
    int local_element;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Scatter(A, 1, MPI_INT, &local_element, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process %d received element: %d\n", rank, local_element);

    MPI_Finalize();

    return 0;
}
