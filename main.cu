/*
name: Xihuai Wang
student_id: 16337236
file description: 
  sort the integer array
*/

#include "hw3_cuda.h"
#include "common.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include <thrust/sort.h>

int main(int argc, char *argv[]) {
  cudaSetDevice(3);

  unsigned int *h_in;
  unsigned int *h_out;
  unsigned int *d_out;
  unsigned int *d_in;
  unsigned int *d_out_scan;
  int num_elements;

  num_elements = atoi(argv[1]);
  h_in = (unsigned int *)malloc(num_elements * sizeof(unsigned int));
  h_out = (unsigned int *)malloc(num_elements * sizeof(unsigned int));
  unsigned int *h_out_scan =
      (unsigned int *)malloc(num_elements * sizeof(unsigned int));

  CHECK(cudaMalloc((void **)&d_in, num_elements * sizeof(unsigned int)));
  CHECK(cudaMalloc((void **)&d_out, num_elements * sizeof(unsigned int)));
  CHECK(cudaMalloc((void **)&d_out_scan, num_elements * sizeof(unsigned int)));

  // init array
  for (unsigned int i = 0; i < num_elements; i++) {
    if(rand() % 2) {
      #ifdef TEST_MODE
      h_in[i] = -rand() % 100;
      #else
      h_in[i] = -rand();
      #endif
    }
    else {
      #ifdef TEST_MODE
      h_in[i] = rand() % 100;
      #else
      h_in[i] = rand();
      #endif
    }
    #ifdef TEST_MODE
    printf("%d,", h_in[i]);
    #endif
  }
  #ifdef TEST_MODE
  printf("\n");
  #endif
  // Copy host variables to device ------------------------------------------

  CHECK(cudaMemcpy(d_in, h_in, num_elements * sizeof(unsigned int),
  cudaMemcpyHostToDevice));
  // Launch kernel ----------------------------------------------------------
  #ifdef CHECK_MODE
  clock_t start = clock();
  #endif

  radix_sort(&d_in, &d_out, d_out_scan, h_in, h_out_scan, num_elements);


  #ifdef CHECK_MODE
  printf("My GPU sort: %fs\n", double(clock()-start)/CLOCKS_PER_SEC);
  #endif

  // Copy device variables from host ----------------------------------------

  CHECK(cudaMemcpy(h_out, d_out, num_elements * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  #ifdef TEST_MODE
  printf("result\n");
  for (unsigned int i = 0; i < num_elements; i++) {
    printf("%d,", h_out[i]);
  }
  printf("\n");
  #endif

  #ifdef CHECK_MODE
  unsigned int * h_in_cpy = (unsigned int*)malloc(num_elements*sizeof(unsigned int));
  memcpy(h_in_cpy, h_in, num_elements*sizeof(unsigned int));
  start = clock();
  thrust::sort(h_in_cpy, h_in_cpy+num_elements);
  printf("Thrust GPU sort: %fs\n", double(clock()-start)/CLOCKS_PER_SEC);

  memcpy(h_in_cpy, h_in, num_elements*sizeof(unsigned int));
  start = clock();
  thrust::stable_sort(h_in_cpy, h_in_cpy+num_elements);
  printf("Thrust GPU stable sort: %fs\n", double(clock()-start)/CLOCKS_PER_SEC);
  memcpy(h_in_cpy, h_in, num_elements*sizeof(unsigned int));
  
  start = clock();
  std::sort(h_in_cpy, h_in_cpy+num_elements);
  printf("STL GPU sort: %fs\n", double(clock()-start)/CLOCKS_PER_SEC);
  
  start = clock();
  std::stable_sort(h_in, h_in+num_elements);
  printf("STL GPU stable sort: %fs\n", double(clock()-start)/CLOCKS_PER_SEC);

  bool flag = true;
  for(unsigned int i = 0; i < num_elements; ++i) {
    if(h_in[i] != h_out[i]) {
      flag = false;
    }
  }
  if (flag) {
    printf("check passed\n");
  } else {
    printf("check failed\n");
  }
  #endif

  cudaFree(d_in);
  cudaFree(d_out_scan);
  cudaFree(d_out);
  free(h_in);
  free(h_out);

  return 0;
}
