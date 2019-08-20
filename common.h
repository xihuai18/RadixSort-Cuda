/*
name: Xihuai Wang
student_id: 16337236
file description:
    some frequently used macros or functions
*/

#ifndef COMMON_H
#define COMMON_H

#include <stdlib.h>
#include <stdio.h>
/*
name: CHECK
function description:
    check whether the function is successfully called and print out the
error message if not.
parameters:
    *call* : the returned value after call a built-in CUDA function
*/
#define CHECK(call)                                          \
  {                                                          \
    const cudaError_t error = call;                          \
    if (error != cudaSuccess) {                              \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
      fprintf(stderr, "code: %d, reason: %s\n", error,       \
              cudaGetErrorString(error));                    \
      exit(1);                                               \
    }                                                        \
  }

inline unsigned int upDiv(unsigned int dividend, unsigned int divisor) {
  return (dividend + divisor - 1) / divisor;
}

#endif  // COMMON_H