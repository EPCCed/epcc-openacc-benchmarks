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


#include "common.h"
#include "main.h"
#include "level0.h"

#ifdef __OPENACC
#include <openacc.h>
#endif

double contig_htod(){

  extern unsigned int datasize;
  unsigned int i = 0;
  double t1_start = 0;
  double t1_end = 0;
  unsigned int n = (unsigned int)(datasize/sizeof(uint8_t));
  uint8_t *a = (uint8_t *)malloc(n*sizeof(uint8_t));

  if (a==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }

  for (i=0;i<n;i++){
    a[i] = (uint8_t)rand();
  }

  t1_start = omp_get_wtime();
#pragma acc data copyin(a[0:n])
  {
    if (n==0) { a[0] = (uint8_t)1;}
  }
  t1_end = omp_get_wtime();

  free(a);
  return( t1_end - t1_start );
}

double contig_dtoh(){

  extern unsigned int datasize;
  unsigned int i = 0;
  double t1_start = 0;
  double t1_end = 0;
  unsigned int n = (int)(datasize/sizeof(uint8_t));
  uint8_t *a = (uint8_t *)malloc(n*sizeof(uint8_t));

  if (a==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }

  for (i=0;i<n;i++){
    a[i] = (uint8_t)rand();
  }


  t1_start = omp_get_wtime();
#pragma acc data copyout(a[0:n])
  {
    if (n==0) { a[0] = (uint8_t)1;}
  }
  t1_end = omp_get_wtime();


  free(a);
  return( t1_end - t1_start );
}


double sliced_dtoh(){

  extern unsigned int datasize;
  double t1_start = 0;
  double t1_end = 0;
  int i = 0;
  int sqrtn = 0;
  uint8_t **a = NULL;
  uint8_t *aptr = NULL;
  unsigned int n = (unsigned int)(datasize/sizeof(uint8_t));
  sqrtn = sqrt((int)n);  /* sqrtn is the number of rows (or cols) you have in the array */
  a = (uint8_t**)malloc(sqrtn*sizeof(uint8_t*));


  if (a==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }

  for (i=0;i<sqrtn;i++){
    a[i] = (uint8_t*)malloc(sqrtn*sizeof(uint8_t));
    if (a[i]==NULL){
      /* Something went wrong in the memory allocation here, fail gracefully */
      return(-10000);
    }
  }

  aptr = a[1];



  



  t1_start = omp_get_wtime();
#pragma acc data copyout(aptr[0:sqrtn])
  {
    if (n==0) { a[0]++;}
  }
  t1_end = omp_get_wtime();

  for (i=0;i<sqrtn;i++){
    free(a[i]);
  }

  free(a);
  return( t1_end - t1_start );
}



double sliced_htod(){

  extern unsigned int datasize;
  unsigned int n = (int)(datasize/sizeof(uint8_t));
  uint8_t **a = NULL;
  int i = 0;
  int j = 0;
  double t1_start = 0;
  double t1_end = 0;
  int sqrtn = sqrt(n);
  uint8_t *aptr = NULL;
  a = (uint8_t**)malloc(sqrtn*sizeof(uint8_t*));
  if (a==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }
  for (i=0;i<sqrtn;i++){
    a[i] = (uint8_t*)malloc(sqrtn*sizeof(uint8_t));
    if (a[i]==NULL){
      /* Something went wrong in the memory allocation here, fail gracefully */
      return(-10000);
    }
  }
  aptr = a[1];


  for (j=0;j<sqrtn;j++){
    for (i=0;i<sqrtn;i++){
      a[j][i] = (uint8_t)rand();
    }
  }


  t1_start = omp_get_wtime();


#pragma acc data copyin(aptr[0:sqrtn])
  {
    if (n==0) { a[0]++;}
  }

  t1_end = omp_get_wtime();

  for (i=0;i<sqrtn;i++){
    free(a[i]);
  }

  free(a);

  return( t1_end - t1_start );
}



double kernels_if(){


  extern unsigned int datasize;

  unsigned int n = (int)(datasize/sizeof(uint8_t));
  uint8_t *a = NULL;
  unsigned int i = 0;
  double t1_start = 0;
  double t1_end = 0;
  double t2_start = 0;
  double t2_end = 0;
  a = (uint8_t *)malloc(n*sizeof(uint8_t));
  if (a==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }

  for (i=0;i<n;i++){
    a[i] = (uint8_t)rand();
  }


#pragma acc data copyin(a[0:n])
  {
    t1_start = omp_get_wtime();
#pragma acc kernels if(0)
    for (i=0;i<n;i++){
      a[i] = i*31;
    }
    t1_end = omp_get_wtime();
  }


  t2_start = omp_get_wtime();
  for (i=0;i<n;i++){
    a[i] = i*31;
  }
  t2_end = omp_get_wtime();


  free(a);
  return( (t1_end-t1_start) - (t2_end-t2_start) );

}

double parallel_if(){

  extern unsigned int datasize;

  unsigned int n = (int)(datasize/sizeof(uint8_t));
  uint8_t *a = NULL;
  unsigned int i = 0;
  double t1_start = 0;
  double t1_end = 0;
  double t2_start = 0;
  double t2_end = 0;


  a = (uint8_t *)malloc(n*sizeof(uint8_t));
  if (a==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }
  for (i=0;i<n;i++){
    a[i] = (uint8_t)rand();
  }




#pragma acc data copyin(a[0:n])
  {
    t1_start = omp_get_wtime();
#pragma acc parallel if(0)
    for (i=0;i<n;i++){
      a[i] = i*31;
    }
    t1_end = omp_get_wtime();
  }


  t2_start = omp_get_wtime();

  for (i=0;i<n;i++){
    a[i] = i*31;
  }
  t2_end = omp_get_wtime();

  free(a);
  return( (t1_end-t1_start) - (t2_end-t2_start) );

}



double parallel_private(){


  extern unsigned int datasize;
  unsigned int n = (unsigned int)(datasize/sizeof(uint8_t));
  unsigned int ppn = (unsigned int)(n/256);
  unsigned int i = 0;
  double t1_start = 0;
  double t1_end = 0;
  double t2_start = 0;
  double t2_end = 0;
  uint8_t *a = NULL;
  uint8_t *z = NULL;

  a = (uint8_t *)malloc(ppn*sizeof(uint8_t));
  z = (uint8_t*)malloc(ppn*sizeof(uint8_t));
  if (a==NULL||z==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }



  t1_start = omp_get_wtime();
#pragma acc parallel loop create(a[0:ppn],z[0:ppn]), num_gangs(256)
  for (i=0;i<ppn;i++){
    a[i] = i;
    z[i] = i;
  }
  t1_end = omp_get_wtime();


  t2_start = omp_get_wtime();
#pragma acc parallel loop create(a[0:ppn]), private(z[0:ppn]), num_gangs(256)
  for (i=0;i<ppn;i++){
    a[i] = i;
    z[i] = i;
  }
  t2_end = omp_get_wtime();

  free(a);
  free(z);
  return( (t2_end-t2_start) - (t1_end-t1_start) );

}



double parallel_firstprivate(){

  extern unsigned int datasize;
  unsigned int i = 0;
  unsigned int n = (int)(datasize/sizeof(uint8_t));
  uint8_t *a = NULL;
  uint8_t *z = NULL;
  double t1_start = 0;
  double t1_end = 0;
  double t2_start = 0;
  double t2_end = 0;

  a = (uint8_t *)malloc(n*sizeof(uint8_t));
  z = (uint8_t*)malloc(n*sizeof(uint8_t));

  if (a==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }

  for (i=0;i<n;i++){
    z[i] = (uint8_t)rand();
  }



  t1_start = omp_get_wtime();
#pragma acc parallel loop create(a[0:n]) copyin(z[0:n])
  for (i=0;i<n;i++){
    a[i] = z[i];
  }
  t1_end = omp_get_wtime();



  t2_start = omp_get_wtime();
#pragma acc parallel loop create(a[0:n]), firstprivate(z[0:n])
  for (i=0;i<n;i++){
    a[i] = z[i];
  }
  t2_end = omp_get_wtime();

  free(a);
  free(z);
  return( (t2_end-t2_start) - (t1_end-t1_start) );
}



double parallel_reduction(){


  extern unsigned int datasize;

  unsigned int n = (int)(datasize/sizeof(uint8_t));
  uint8_t *a = NULL;
  int z = 0;
  unsigned int i = 0;
  double t1_start = 0;
  double t1_end = 0;
  double t2_start = 0;
  double t2_end = 0;


  a = (uint8_t *)malloc(n*sizeof(uint8_t));
  if (a==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }

  for (i=0;i<n;i++){
    a[i] = (uint8_t)rand();
  }


#pragma acc data copyin(a[0:n])
  {
    t1_start = omp_get_wtime();
#pragma acc parallel loop reduction(+:z)
    for (i=0;i<n;i++){
      z += a[i];
    }
    t1_end = omp_get_wtime();



    t2_start = omp_get_wtime();
#pragma acc parallel loop
    for (i=0;i<n;i++){
      z += a[i];
    }
    t2_end = omp_get_wtime();

  }

  if (n==0){
    printf("Z=%d\n", z);
  }

  free(a);
  return( (t2_end-t2_start) - (t1_end-t1_start) );
}


double kernels_reduction(){


  extern unsigned int datasize;

  unsigned int n = (int)(datasize/sizeof(uint8_t));
  uint8_t *a = NULL;
  int z = 0;
  unsigned int i = 0;
  double t1_start = 0;
  double t1_end = 0;
  double t2_start = 0;
  double t2_end = 0;
  a = (uint8_t *)malloc(n*sizeof(uint8_t));
  if (a==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }
  for (i=0;i<n;i++){
    a[i] = (uint8_t)rand();
  }


#pragma acc data copyin(a[0:n])
  {
    t1_start = omp_get_wtime();
#pragma acc kernels
#pragma acc loop reduction(+:z)
    for (i=0;i<n;i++){
      z += a[i];
    }
    t1_end = omp_get_wtime();

    t2_start = omp_get_wtime();
#pragma acc kernels
#pragma acc loop
    for (i=0;i<n;i++){
      z += a[i];
    }
    t2_end = omp_get_wtime();
  }

  if (n==0){
    printf("Z=%d\n", z);
  }
  free(a);
  return( (t2_end-t2_start) - (t1_end-t1_start) );
}



double kernels_combined(){

  extern unsigned int datasize;

  unsigned int n = (int)(datasize/sizeof(uint8_t));
  uint8_t *a = NULL;
  unsigned int i = 0;
  double t1_start = 0;
  double t1_end = 0;
  double t2_start = 0;
  double t2_end = 0;

  a = (uint8_t *)malloc(n*sizeof(uint8_t));
  if (a==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }


  t1_start = omp_get_wtime();
#pragma acc kernels loop create(a[0:n])
  for (i=0;i<n;i++){
    a[i] = i;
  }
  t1_end = omp_get_wtime();


  t2_start = omp_get_wtime();
#pragma acc kernels create(a[0:n])
#pragma acc loop
  for (i=0;i<n;i++){
    a[i] = i;
  }
  t2_end = omp_get_wtime();

  free(a);
  return( (t2_end-t2_start) - (t1_end-t1_start) );
}


double parallel_combined(){

  extern unsigned int datasize;

  unsigned int n = (int)(datasize/sizeof(uint8_t));
  uint8_t *a = NULL;
  unsigned int i = 0;
  double t1_start = 0;
  double t1_end = 0;
  double t2_start = 0;
  double t2_end = 0;

  a = (uint8_t *)malloc(n*sizeof(uint8_t));
  if (a==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }

  t1_start = omp_get_wtime();
#pragma acc parallel loop create(a[0:n])
  for (i=0;i<n;i++){
    a[i] = i;
  }
  t1_end = omp_get_wtime();


  t2_start = omp_get_wtime();
#pragma acc parallel create(a[0:n])
#pragma acc loop
  for (i=0;i<n;i++){
    a[i] = i;
  }
  t2_end = omp_get_wtime();

  free(a);
  return( (t2_end-t2_start) - (t1_end-t1_start) );
}



double update(){

  extern unsigned int datasize;

  unsigned int n = (int)(datasize/sizeof(uint8_t));
  uint8_t *a = NULL;
  unsigned int i = 0;
  double t1_start = 0;
  double t1_end = 0;
  double t2_start = 0;
  double t2_end = 0;

  a = (uint8_t *)malloc(n*sizeof(uint8_t));
  if (a==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }

  for (i=0;i<n;i++){
    a[i] = (uint8_t)rand();
  }

  t1_start = omp_get_wtime();
#pragma acc data copyin(a[0:n])
  {

#pragma acc kernels
	  {
		#pragma acc loop
			for (i=0;i<n;i++){
			  a[i] = i;
			}
	  }
#pragma acc kernels
	  {
		#pragma acc loop
		for (i=0;i<n;i++){
		  a[i] = i;
		}
	  }
  }
  t1_end = omp_get_wtime();


  t2_start = omp_get_wtime();
#pragma acc data copyin(a[0:n])
  {

#pragma acc kernels
{
	#pragma acc loop
    for (i=0;i<n;i++){
      a[i] = i;
    }
}
#pragma acc update host(a[0:n])

#pragma acc kernels
	{
		#pragma acc loop
		for (i=0;i<n;i++){
		  a[i] = i;
		}
	}
  }
  t2_end = omp_get_wtime();

  free(a);
  return( (t2_end-t2_start) - (t1_end-t1_start) );

}


/* Kernels Invocation */
double kernels_invoc(){

  extern unsigned int datasize;
  unsigned int n = (int)(datasize/sizeof(uint8_t));
  uint8_t *a = NULL;
  unsigned int i = 0;
  double t1_start = 0;
  double t1_end = 0;
  double t2_start = 0;
  double t2_end = 0;
  double t3_start = 0;
  double t3_end = 0;
  double tmp = 0;

  a = (uint8_t *)malloc(n*sizeof(uint8_t));
  if (a==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }

  for (i=0;i<n;i++){
    a[i] = (uint8_t)rand();
  }

#pragma acc data copyin(a[0:n],i,n)
  {


    t1_start = omp_get_wtime();
#pragma acc kernels loop
    for (i=0;i<n;i++){
      a[i] = 10*i;
    }
    t1_end = omp_get_wtime();


    t2_start = omp_get_wtime();
#pragma acc kernels loop
    for (i=0;i<n;i++){
      a[i] = 10*a[i];
    }
    t2_end = omp_get_wtime();


    t3_start = omp_get_wtime();
#pragma acc kernels loop
    for (i=0;i<n;i++){
      a[i] = 10*a[i];
      a[i] = 10*a[i];
    }
    t3_end = omp_get_wtime();

  }


  tmp = ( (t1_end-t1_start) + (t2_end-t2_start) ) - (t3_end-t3_start);
  free(a);
  return(tmp);

}

/* Parallel Invocation */
double parallel_invoc(){

  extern unsigned int datasize;
  unsigned int n = (int)(datasize/sizeof(uint8_t));
  uint8_t *a = NULL;
  unsigned int i = 0;
  double t1_start = 0;
  double t1_end = 0;
  double t2_start = 0;
  double t2_end = 0;
  double t3_start = 0;
  double t3_end = 0;
  double tmp = 0;

  a = (uint8_t *)malloc(n*sizeof(uint8_t));
  if (a==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }

  for (i=0;i<n;i++){
    a[i] = (uint8_t)rand();
  }


#pragma acc data copyin(a[0:n])
  {


    t1_start = omp_get_wtime();
#pragma acc parallel loop
    for (i=0;i<n;i++){
      a[i] = a[i]*10;
    }
    t1_end = omp_get_wtime();


    t2_start = omp_get_wtime();
#pragma acc parallel loop
    for (i=0;i<n;i++){
      a[i] = a[i]*10;
    }
    t2_end = omp_get_wtime();


    t3_start = omp_get_wtime();
#pragma acc parallel loop
    for (i=0;i<n;i++){
      a[i] = a[i]*10;
      a[i] = a[i]*10;
    }
    t3_end = omp_get_wtime();

  }

  tmp = ( (t1_end-t1_start) + (t2_end-t2_start) ) - (t3_end-t3_start);
  free(a);
  return(tmp);

}
