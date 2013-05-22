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
#include <stdint.h>
#include <string.h>

#include "common.h"
#include "main.h"
#include "level0.h"
#include "level1.h"
#include "27stencil.h"
#include "le_core.h"
#include "himeno.h"

#ifdef _OPENACC
#include <openacc.h>
#endif

void wul();

int main(int argc, char **argv) {

  char testName[32];

  /* Initialise storage for test results & parse input arguements. */
  init(argc, argv);

  /* Ensure device is awake. */
  wul();

  /* Level 0 Tests - Speeds and Feeds */
  sprintf(testName, "ContigH2D");
  benchmark(testName, &contig_htod);

  sprintf(testName, "ContigD2H");
  benchmark(testName, &contig_dtoh);

  sprintf(testName, "SlicedD2H");
  benchmark(testName, &sliced_dtoh);

  sprintf(testName, "SlicedH2D");
  benchmark(testName, &sliced_htod);

  sprintf(testName, "Kernels_If");
  benchmark(testName, &kernels_if);

  sprintf(testName, "Parallel_If");
  benchmark(testName, &parallel_if);

  sprintf(testName, "Parallel_private");
  benchmark(testName, &parallel_private);

  sprintf(testName, "Parallel_1stprivate");
  benchmark(testName, &parallel_firstprivate);

  sprintf(testName, "Kernels_combined");
  benchmark(testName, &kernels_combined);
 
  sprintf(testName, "Parallel_combined");
  benchmark(testName, &parallel_combined);

  sprintf(testName, "Update_Host");
  benchmark(testName, &update);

  sprintf(testName, "Kernels_Invocation");
  benchmark(testName, &kernels_invoc);

  sprintf(testName, "Parallel_Invocation");
  benchmark(testName, &parallel_invoc);

  sprintf(testName, "Parallel_Reduction");
  benchmark(testName, &parallel_reduction);

  sprintf(testName, "Kernels_Reduction");
  benchmark(testName, &kernels_reduction);

  /* Level 1 Tests - BLAS-esque kernels */

  sprintf(testName, "2MM");
  benchmark(testName, &twomm);

  sprintf(testName, "3MM");
  benchmark(testName, &threemm);

  sprintf(testName, "ATAX");
  benchmark(testName, &atax);

  sprintf(testName, "BICG");
  benchmark(testName, &bicg);

  sprintf(testName, "MVT");
  benchmark(testName, &mvt);

  sprintf(testName, "SYRK");
  benchmark(testName, &syrk);

  sprintf(testName, "COV");
  benchmark(testName, &covariance);

  sprintf(testName, "COR");
  benchmark(testName, &correlation);

  sprintf(testName, "SYR2K");
  benchmark(testName, &syr2k);

  sprintf(testName, "GESUMMV");
  benchmark(testName, &gesummv);

  sprintf(testName, "GEMM");
  benchmark(testName, &gemm);

  sprintf(testName, "2DCONV");
  benchmark(testName, &twodconv);

  sprintf(testName, "3DCONV");
  benchmark(testName, &threedconv);

  /* Level 2 Tests - small applications */

  sprintf(testName, "27S");
  benchmark(testName, &stencil);

  sprintf(testName, "LE2D");
  benchmark(testName, &le_main);

  sprintf(testName, "HIMENO");
  benchmark(testName, &himeno_main);

  /* Print results & free results storage */
  finalise();

  return EXIT_SUCCESS;

}

/*
 * This function ensures the device is awake.
 * It is more portable than acc_init().
 */
void wul(){

  int data = 8192;
  double *arr_a = (double *)malloc(sizeof(double) * data);
  double *arr_b = (double *)malloc(sizeof(double) * data);
  int i = 0;

  if (arr_a==NULL||arr_b==NULL) {
      printf("Unable to allocate memory in wul.\n");
  }

  for (i=0;i<data;i++){
    arr_a[i] = (double) (rand()/(1.0+RAND_MAX));
  }

#pragma acc data copy(arr_b[0:data]), copyin(arr_a[0:data])
  {
#pragma acc parallel loop
    for (i=0;i<data;i++){
      arr_b[i] = arr_a[i] * 2;
    }
  }

  if (arr_a[0] < 0){
    printf("Error in WUL\n");
    /*
     * This should never be called as rands should be in the range (0,1].
     * This stops clever optimizers.
     */
  }

  free(arr_a);
  free(arr_b);

}
