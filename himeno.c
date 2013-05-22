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

/*********************************************************************
      This benchmark test program is measuring a cpu performance 
      of floating point operation and memory access speed.  

      Modification needed for testing turget computer!!
      Please adjust parameter : nn to take one minute to execute
      all calculation.  Original parameter set is for PC with 
      200 MHz MMX PENTIUM, whose score using this benchmark test
      is about 32.3 MFLOPS.

      If you have any question, please ask me via email.
      written by Ryutaro HIMENO, October 3, 1998.
      Version 2.0  
      ----------------------------------------------
         Ryutaro Himeno, Dr. of Eng.
         Head of Computer Information Center, 
         The Institute of Pysical and Chemical Research (RIKEN)
         Email : himeno@postman.riken.go.jp
      ---------------------------------------------------------------
      You can adjust the size of this benchmark code to fit your target
      computer.  In that case, please chose following sets of 
      (mimax,mjmax,mkmax):
       small : 129,65,65
       midium: 257,129,129
       large : 513,257,257
       ext.large: 1025,513,513
      This program is to measure a computer performance in MFLOPS
      by using a kernel which appears in a linear solver of pressure 
      Poisson included in an incompressible Navier-Stokes solver. 
      A point-Jacobi method is employed in this solver.
      ------------------
      Finite-difference method, curvilinear coodinate system
      Vectorizable and parallelizable on each grid point
      No. of grid points : imax x jmax x kmax including boundaries
      ------------------
      A,B,C:coefficient matrix, wrk1: source term of Poisson equation
      wrk2 : working area, OMEGA : relaxation parameter
      BND:control variable for boundaries and objects ( = 0 or 1)
      P: pressure 
       -----------------
      -------------------     
      "use portlib" statement on the next line is for Visual fortran 
      to use UNIX libraries.  Please remove it if your system is UNIX.
      -------------------
     use portlib

     Version 0.2 
*********************************************************************/

#include <stdio.h>
#include <omp.h>
#include "himeno.h"

#ifdef SMALL
#define MIMAX            129
#define MJMAX            65
#define MKMAX            65
#endif

#ifdef MIDDLE
#define MIMAX            257
#define MJMAX            129
#define MKMAX            129
#endif

#ifdef LARGE
#define MIMAX            513
#define MJMAX            257
#define MKMAX            257
#endif

static double  p[MIMAX][MJMAX][MKMAX];
static double  a[MIMAX][MJMAX][MKMAX][4],
              b[MIMAX][MJMAX][MKMAX][3],
              c[MIMAX][MJMAX][MKMAX][3];
static double  bnd[MIMAX][MJMAX][MKMAX];
static double  wrk1[MIMAX][MJMAX][MKMAX],
              wrk2[MIMAX][MJMAX][MKMAX];

#define NN               3


static int imax, jmax, kmax;
static double omega;

double himeno_main()
{
  int i, j, k;
  double gosa;
  double cpu0, cpu1, nflop, xmflops2, score;

  omega = 0.8;
  imax = MIMAX-1;
  jmax = MJMAX-1;
  kmax = MKMAX-1;

  /*
   *    Initializing matrixes
   */
  initmt();

  /* printf("mimax = %d mjmax = %d mkmax = %d\n",MIMAX, MJMAX, MKMAX); */
  /* printf("imax = %d jmax = %d kmax =%d\n",imax,jmax,kmax); */

  /*
   *    Start measuring
   */
  /* cpu0 = second(); */

  /*
   *    Jacobi iteration
   */

  gosa = jacobi(NN);
  
  /* cpu1 = second(); */
  /* cpu1 = cpu1 - cpu0; */

  /* nflop = (kmax-2)*(jmax-2)*(imax-2)*34; */

  /* if(cpu1 != 0.0) */
  /*   xmflops2 = nflop/cpu1*1.0e-6*(double)NN; */

  /* score = xmflops2/32.27; */
  
  /* printf("\ncpu : %f sec.\n", cpu1); */
  /* printf("Loop executed for %d times\n",NN); */
  /* printf("Gosa : %e \n",gosa); */
  /* printf("MFLOPS measured : %f\n",xmflops2); */
  /* printf("Score based on MMX Pentium 200MHz : %f\n",score); */
  
  /* // Now estimate how many iterations could be done in 20s */
  /* int nn2 = 20.0/cpu1*NN; */
  /* cpu0 = second(); */
  /* gosa = jacobi(nn2); */
  /* cpu1 = second(); */
  /* cpu1 = cpu1 - cpu0; */

  /* nflop = (kmax-2)*(jmax-2)*(imax-2)*34; */

  /* if(cpu1 != 0.0) */
  /*   xmflops2 = nflop/cpu1*1.0e-6*(double)nn2; */

  /* score = xmflops2/32.27; */
  
  /* printf("\ncpu : %f sec.\n", cpu1); */
  /* printf("Loop executed for %d times\n",nn2); */
  /* printf("Gosa : %e \n",gosa); */
  /* printf("MFLOPS measured : %f\n",xmflops2); */
  /* printf("Score based on MMX Pentium 200MHz : %f\n",score); */
  
  return (gosa);
}

void initmt()
{
	int i,j,k;

  for(i=0 ; i<imax ; ++i)
    for(j=0 ; j<jmax ; ++j)
      for(k=0 ; k<kmax ; ++k){
        a[i][j][k][0]=0.0;
        a[i][j][k][1]=0.0;
        a[i][j][k][2]=0.0;
        a[i][j][k][3]=0.0;
        b[i][j][k][0]=0.0;
        b[i][j][k][1]=0.0;
        b[i][j][k][2]=0.0;
        c[i][j][k][0]=0.0;
        c[i][j][k][1]=0.0;
        c[i][j][k][2]=0.0;
        p[i][j][k]=0.0;
        wrk1[i][j][k]=0.0;
        bnd[i][j][k]=0.0;
      }

  for(i=0 ; i<imax ; ++i)
    for(j=0 ; j<jmax ; ++j)
      for(k=0 ; k<kmax ; ++k){
        a[i][j][k][0]=1.0;
        a[i][j][k][1]=1.0;
        a[i][j][k][2]=1.0;
        a[i][j][k][3]=1.0/6.0;
        b[i][j][k][0]=0.0;
        b[i][j][k][1]=0.0;
        b[i][j][k][2]=0.0;
        c[i][j][k][0]=1.0;
        c[i][j][k][1]=1.0;
        c[i][j][k][2]=1.0;
        p[i][j][k]=(double)(k*k)/(double)((kmax-1)*(kmax-1));
        wrk1[i][j][k]=0.0;
        bnd[i][j][k]=1.0;
      }
}

double jacobi(int nn)
{

  int i,j,k,n;
  double gosa, s0, ss, t1, t2;
  t1 = omp_get_wtime();
#pragma acc data copyin(a,b,p), create(i,j,k,s0,ss)
  {
    for(n=0;n<nn;++n){
#pragma acc parallel loop private(i,j,k,s0,ss), reduction(+:gosa)
      for(i=1 ; i<imax-1 ; ++i){
	for(j=1 ; j<jmax-1 ; ++j){
	  for(k=1 ; k<kmax-1 ; ++k){
	    s0 = a[i][j][k][0] * p[i+1][j  ][k  ]
	      + a[i][j][k][1] * p[i  ][j+1][k  ]
	      + a[i][j][k][2] * p[i  ][j  ][k+1]
	      + b[i][j][k][0] * ( p[i+1][j+1][k  ] - p[i+1][j-1][k  ]
				  - p[i-1][j+1][k  ] + p[i-1][j-1][k  ] )
	      + b[i][j][k][1] * ( p[i  ][j+1][k+1] - p[i  ][j-1][k+1]
				  - p[i  ][j+1][k-1] + p[i  ][j-1][k-1] )
	      + b[i][j][k][2] * ( p[i+1][j  ][k+1] - p[i-1][j  ][k+1]
				  - p[i+1][j  ][k-1] + p[i-1][j  ][k-1] )
	      + c[i][j][k][0] * p[i-1][j  ][k  ]
	      + c[i][j][k][1] * p[i  ][j-1][k  ]
	      + c[i][j][k][2] * p[i  ][j  ][k-1]
	      + wrk1[i][j][k];
	    
	    ss = ( s0 * a[i][j][k][3] - p[i][j][k] ) * bnd[i][j][k];
	    
	    gosa = gosa + ss*ss;
	  
	    wrk2[i][j][k] = p[i][j][k] + omega * ss;
	  }
	}
      }
#pragma acc wait      
#pragma acc parallel loop
      for(i=1 ; i<imax-1 ; ++i)
	for(j=1 ; j<jmax-1 ; ++j)
	  for(k=1 ; k<kmax-1 ; ++k)
	    p[i][j][k] = wrk2[i][j][k];
    } /* end n loop */
  } /* end data loop */ 
  t2 = omp_get_wtime();
  return(t2-t1);
}

double second()
{
#include <sys/time.h>

  struct timeval tm;
  double t ;

  static int base_sec = 0,base_usec = 0;

  gettimeofday(&tm, NULL);
  
  if(base_sec == 0 && base_usec == 0)
    {
      base_sec = tm.tv_sec;
      base_usec = tm.tv_usec;
      t = 0.0;
  } else {
    t = (double) (tm.tv_sec-base_sec) + 
      ((double) (tm.tv_usec-base_usec))/1.0e6 ;
  }

  return t ;
}
