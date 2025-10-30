#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, size;
    int B[4][4] = {
        {10, 20, 30, 40},
        {50, 60, 70, 80},
        {90, 100, 110, 120},
        {130, 140, 150, 160}};
    int local_row[4];

    
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Scatter(B, 4, MPI_INT, local_row, 4, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process %d received row: [%d, %d, %d, %d]\n",
           rank, local_row[0], local_row[1], local_row[2], local_row[3]);

    MPI_Finalize();

    return 0;
}
