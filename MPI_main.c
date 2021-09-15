int main (int nargs, char **args)
{
int M=0, N=0, K=0, my_rank;
float **input=NULL, **output=NULL, **kernel=NULL
MPI_Init (&nargs, &args);
MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
if (my_rank==0) {
// read from command line the values of M, N, and K
M = 5;
N = 4;
// allocate 2D array ’input’ with M rows and N columns

*input = malloc(M * sizeof *input);
*input[0] = malloc(M * N * sizeof (*input)[0]);
for(int i = 0; i < M; i++){
  for(int j = 0; j < N; j++){
    (*SNN_table)[i][j] = 0;
  }
}
for(int i = 0; i < M; i++){
  for(int j = 0; j < N; j++){
    printf("%d", (*SNN_table)[i][j]);
  }
}
// allocate 2D array ’output’ with M-K+1 rows and N-K+1 columns
// allocate the convolutional kernel with K rows and K columns
// fill 2D array ’input’ with some values
// fill kernel with some values
// ....
}
// process 0 broadcasts values of M, N, K to all the other processes
// ...
if (my_rank>0) {
// allocated the convolutional kernel with K rows and K columns
// ...
}
// process 0 broadcasts the content of kernel to all the other processes
// ...
// parallel computation of a single-layer convolution
MPI_double_layer_convolution(M, N, input, K, kernel, output);
if (my_rank==0) {
// For example, compare the content of array ’output’ with that is
// produced by the sequential function single_layer_convolution
// ...
}
MPI_Finalize();
return 0;
}
