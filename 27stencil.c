/* Copyright (c) 2013 The University of Edinburgh. */

/* Licensed under the Apache License, Version 2.0 (the "License"); */
/* you may not use this file except in compliance with the License. */
/* You may obtain a copy of the License at */

/*     http://www.apache.org/licenses/LICENSE-2.0 */

/* Unless required by applicable law or agreed to in writing, software */
/* distributed under the License is distributed on an "AS IS" BASIS, */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/* See the License for the specific language governing permissions and */
/* limitations under the License. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "common.h"
#include "main.h"
#include <string.h>
#ifdef _OPENACC
#include <openacc.h>
#endif

#define ITERATIONS 10
#define FAC (1./26)
#define TOLERANCE 1.0e-15


/* Forward Declarations of utility functions*/
double max_diff(double *, double *, int);


double stencil()
{
  extern unsigned int datasize;
  int sz = cbrt((datasize/sizeof(double))/2);
  int i, j, k, iter;
  int n = sz-2;
  double fac = FAC;
  double t1, t2;
  double md;

  /* Work buffers, with halos */
  double *a0 = (double*)malloc(sizeof(double)*sz*sz*sz);
  double *device_result = (double*)malloc(sizeof(double)*sz*sz*sz);
  double *a1 = (double*)malloc(sizeof(double)*sz*sz*sz);
  double *host_result = (double*)malloc(sizeof(double)*sz*sz*sz);
  double *a0_init = (double*)malloc(sizeof(double)*sz*sz*sz);

  if(a0==NULL||device_result==NULL||a1==NULL||host_result==NULL||a0_init==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }


  /* initialize input array a0 */

  /* zero all of array (including halos) */
  for (i = 0; i < sz; i++) {
    for (j = 0; j < sz; j++) {
      for (k = 0; k < sz; k++) {
        a0[i*sz*sz+j*sz+k] = 0.0;
      }
    }
  }


  /* use random numbers to fill interior */
  for (i = 1; i < n+1; i++) {
    for (j = 1; j < n+1; j++) {
      for (k = 1; k < n+1; k++) {
        a0[i*sz*sz+j*sz+k] = (double) rand()/ (double)(1.0 + RAND_MAX);
      }
    }
  }

  /* memcpy(&a0_init[0], &a0[0], sizeof(double)*sz*sz*sz); */

  /* save initial input array for later GPU run */
  for (i = 0; i < sz; i++) {
    for (j = 0; j < sz; j++) {
      for (k = 0; k < sz; k++) {
        a0_init[i*sz*sz+j*sz+k] = a0[i*sz*sz+j*sz+k];
      }
    }
  }


  /* run main computation on host */

  for (iter = 0; iter < ITERATIONS; iter++) {

    for (i = 1; i < n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a1[i*sz*sz+j*sz+k] = (
                         a0[i*sz*sz+(j-1)*sz+k] + a0[i*sz*sz+(j+1)*sz+k] +
                         a0[(i-1)*sz*sz+j*sz+k] + a0[(i+1)*sz*sz+j*sz+k] +
                         a0[(i-1)*sz*sz+(j-1)*sz+k] + a0[(i-1)*sz*sz+(j+1)*sz+k] +
                         a0[(i+1)*sz*sz+(j-1)*sz+k] + a0[(i+1)*sz*sz+(j+1)*sz+k] +
                         a0[i*sz*sz+(j-1)*sz+(k-1)] + a0[i*sz*sz+(j+1)*sz+(k-1)] +
                         a0[(i-1)*sz*sz+j*sz+(k-1)] + a0[(i+1)*sz*sz+j*sz+(k-1)] +
                         a0[(i-1)*sz*sz+(j-1)*sz+(k-1)] + a0[(i-1)*sz*sz+(j+1)*sz+(k-1)] +
                         a0[(i+1)*sz*sz+(j-1)*sz+(k-1)] + a0[(i+1)*sz*sz+(j+1)*sz+(k-1)] +
                         a0[i*sz*sz+(j-1)*sz+(k+1)] + a0[i*sz*sz+(j+1)*sz+(k+1)] +
                         a0[(i-1)*sz*sz+j*sz+(k+1)] + a0[(i+1)*sz*sz+j*sz+(k+1)] +
                         a0[(i-1)*sz*sz+(j-1)*sz+(k+1)] + a0[(i-1)*sz*sz+(j+1)*sz+(k+1)] +
                         a0[(i+1)*sz*sz+(j-1)*sz+(k+1)] + a0[(i+1)*sz*sz+(j+1)*sz+(k+1)] +
                         a0[i*sz*sz+j*sz+(k-1)] + a0[i*sz*sz+j*sz+(k+1)]
                         ) * fac;
        }
      }
    }


    for (i = 1; i < n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a0[i*sz*sz+j*sz+k] = a1[i*sz*sz+j*sz+k];
        }
      }
    }





  } /* end iteration loop */

  /* save result */
  /* memcpy(&host_result[0], &a0[0], sizeof(double)*sz*sz*sz); */
  for (i = 0; i < sz; i++) {
    for (j = 0; j < sz; j++) {
      for (k = 0; k < sz; k++) {
        host_result[i*sz*sz+j*sz+k] = a0[i*sz*sz+j*sz+k];
      }
    }
  }


  /* copy initial array back to a0 */
  /* memcpy(&a0[0], &a0_init[0], sizeof(double)*sz*sz*sz); */
  for (i = 0; i < sz; i++) {
    for (j = 0; j < sz; j++) {
      for (k = 0; k < sz; k++) {
        a0[i*sz*sz+j*sz+k] = a0_init[i*sz*sz+j*sz+k];
      }
    }
  }


  t1 = omp_get_wtime();
#pragma acc data copy(a0[0:sz*sz*sz]), create(a1[0:sz*sz*sz], i,j,k,iter), copyin(sz,fac,n)
  {

    for (iter = 0; iter < ITERATIONS; iter++) {
#pragma acc parallel loop
      for (i = 1; i < n+1; i++) {
#pragma acc loop
        for (j = 1; j < n+1; j++) {
#pragma acc loop
          for (k = 1; k < n+1; k++) {
            a1[i*sz*sz+j*sz+k] = (
                         a0[i*sz*sz+(j-1)*sz+k] + a0[i*sz*sz+(j+1)*sz+k] +
                         a0[(i-1)*sz*sz+j*sz+k] + a0[(i+1)*sz*sz+j*sz+k] +
                         a0[(i-1)*sz*sz+(j-1)*sz+k] + a0[(i-1)*sz*sz+(j+1)*sz+k] +
                         a0[(i+1)*sz*sz+(j-1)*sz+k] + a0[(i+1)*sz*sz+(j+1)*sz+k] +
                         a0[i*sz*sz+(j-1)*sz+(k-1)] + a0[i*sz*sz+(j+1)*sz+(k-1)] +
                         a0[(i-1)*sz*sz+j*sz+(k-1)] + a0[(i+1)*sz*sz+j*sz+(k-1)] +
                         a0[(i-1)*sz*sz+(j-1)*sz+(k-1)] + a0[(i-1)*sz*sz+(j+1)*sz+(k-1)] +
                         a0[(i+1)*sz*sz+(j-1)*sz+(k-1)] + a0[(i+1)*sz*sz+(j+1)*sz+(k-1)] +
                         a0[i*sz*sz+(j-1)*sz+(k+1)] + a0[i*sz*sz+(j+1)*sz+(k+1)] +
                         a0[(i-1)*sz*sz+j*sz+(k+1)] + a0[(i+1)*sz*sz+j*sz+(k+1)] +
                         a0[(i-1)*sz*sz+(j-1)*sz+(k+1)] + a0[(i-1)*sz*sz+(j+1)*sz+(k+1)] +
                         a0[(i+1)*sz*sz+(j-1)*sz+(k+1)] + a0[(i+1)*sz*sz+(j+1)*sz+(k+1)] +
                         a0[i*sz*sz+j*sz+(k-1)] + a0[i*sz*sz+j*sz+(k+1)]
                           ) * fac;
          }
        }
      }

#pragma acc parallel loop
      for (i = 1; i < n+1; i++) {
#pragma acc loop
        for (j = 1; j < n+1; j++) {
#pragma acc loop
          for (k = 1; k < n+1; k++) {
            a0[i*sz*sz+j*sz+k] = a1[i*sz*sz+j*sz+k];
          }
        }
      }


    } /* end iteration loop */

  } /* end data region */
#pragma acc wait
  t2 = omp_get_wtime();

  memcpy(&device_result[0], &a0[0], sizeof(double)*sz*sz*sz);
  md = max_diff(&host_result[0],&device_result[0], sz);

    /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a0_init);
  free(a1);
  free(host_result);
  free(device_result);

  if (md < TOLERANCE ){
    /* printf ("GPU matches host to within tolerance of %1.1e\n\n", TOLERANCE); */
    return(t2 - t1);
  }
  else{
    /* printf ("WARNING: GPU does not match to within tolerance of %1.1e\nIt is %lf\n", TOLERANCE, md); */
    return(-11000);
  }


}


/* Utility Functions */

double max_diff(double *array1,double *array2, int sz)
{
  double tmpdiff, diff;
  int i,j,k;
  int n = sz-2;
  diff=0.0;

  for (i = 1; i < n+1; i++) {
    for (j = 1; j < n+1; j++) {
      for (k = 1; k < n+1; k++) {
        tmpdiff = fabs(array1[i*sz*sz+j*sz+k] - array2[i*sz*sz+j*sz+k]);
        if (tmpdiff > diff) diff = tmpdiff;

      }
    }
  }
  return diff;
}


