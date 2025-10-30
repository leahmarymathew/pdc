#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, size;
    int token;
    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {

        token = 50;
        printf("Process 0: Sending token value %d to Process 1\n", token);
        MPI_Send(&token, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1)
    {

        MPI_Recv(&token, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Process 1: Received token value %d from Process 0\n", token);
    }

    MPI_Finalize();

    return 0;
}
