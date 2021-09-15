#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>


void MPI_double_layer_convolution(int M, int N, float **input, int K1, float **kermel1, int K2, float **kernel2, float **output){
  int i,j,ii,jj;
  double temp;
  for (i=0; i<=M-K1; i++)
    for (j=0; j<=N-K1; j++) {
      temp = 0.0;
      for (ii=0; ii<K1; ii++)
        for (jj=0; jj<K1; jj++)
          temp += input[i+ii][j+jj]*kernel1[ii][jj];
      midpoint[i][j] = temp;
    }

    for (i=0; i<=M-K1-K2; i++)
      for (j=0; j<=N-K1-K2; j++) {
        temp = 0.0;
        for (ii=0; ii<K1-K2; ii++)
          for (jj=0; jj<K1-K2; jj++)
            temp += midpoint[i+ii][j+jj]*kernel2[ii][jj];
        output[i][j] = temp;
      }

}
