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

#ifndef LE_CORE_H
#define LE_CORE_H

/*
 * Base include file.
 * All data structures and functions.
 * Prefix le_* used (Linear Elastic) for all structures and function.
 */

#include <sys/time.h>

/* static __inline__ unsigned long getCC(void) */
/* { */
/* 	unsigned a, d; */
/* 	__asm volatile("rdtsc" : "=a" (a), "=d" (d)); */
/* 	return ((unsigned long)a) | (((unsigned long)d) << 32); */
/* } */


static __inline__ double timer()
{
	struct timeval theStartTime;
	gettimeofday(&theStartTime, NULL);
	return theStartTime.tv_sec + 1e-6 * theStartTime.tv_usec;
}



/* Call string for le_main */
double le_main();

/*
 * Storage types:
 * ST_SOA - Structure of srrays.
 * ST_AOS - Array of structures.
 */
#define ST_SOA 0
#define ST_AOS 1

/* Real type. */
typedef double real;
/* Integer type. */
typedef int int_t;

/* 2d vector struct. */
typedef struct {
	real x, y;
} le_vec2;

/* 
 * Symmetrical tensor 2d.
 * sxx sxy
 * sxy sxx
 */
typedef struct {
	real xx, xy, yy;
} le_smatrix2;

/* Integer 2d point. */
typedef struct {
	int_t x, y;
} le_point2;

/*
 * Elastic node structure.
 * Store velocity and stress tensor (http://en.wikipedia.org/wiki/Stress_%28mechanics%29).
 */
typedef struct {
  le_vec2 v;
  le_smatrix2 s;
} le_node;

/*
 * Riemann invariant structure (http://en.wikipedia.org/wiki/Riemann_invariant).
 */
typedef	struct {
  real w1, w2, w3, w4, w5;
} le_w;

/*
 * Elastic material structure.
 * We use simple approach when all parameters are constant on whole region.
 * c1 - speed of P-wave http://en.wikipedia.org/wiki/P-wave.
 * c2 - speed of S-wave http://en.wikipedia.org/wiki/S-wave.
 * rho - density.
 */
typedef struct {
  real c1, c2, rho;
	
  /*
   * Some cached values to speedup calculations.
   */
  real irhoc1; /* 1.0 / (c1 * rho) */
  real irhoc2; /* 1.0 / (c2 * rho) */
  real rhoc1;  /* c1 * rho */
  real rhoc2;  /* c2 * rho */
  real rhoc3;  /* c3 * rho */
} le_material;

/*
 * Structure for storing all parameters of task.
 */
typedef struct {
	/* Time step.*/
	real dt;
	
	/* Grid spacing. */
	le_vec2 h;
	
	/* Number of nodes ing grid on each axis. */
	le_point2 n;
	
	/* Material. */
	le_material mat;
	
	/*
	 * Storage type.
	 * Array of structure or structure of arrays.
	 */
	int stype;
	
	/* Grid data (nodes). */
	le_node *grid;

	/* Temp store for grid data (nodes). */
	le_node *tmpGrid;
} le_task;

/*
 * Create material and init all fields of structure.
 */
void le_init_material(const real c1, const real c2, const real rho, le_material *m);

/* Create task with given parameters. Allocate memory for nodes. */
void le_init_task(le_task *task, const real dt, const le_vec2 h, const le_material mat, const le_point2 n, const int stype);

/* Free memory. */
void le_free_task(le_task *task);

/*
 * Set initial disturbance on the grid.
 */
void le_set_ball(le_task *t, const le_vec2 c, const real r, const real s);

/*
 * Save grid to legasy VTK format (http://www.vtk.org/VTK/img/file-formats.pdf).
 * You can use ParaView (http://www.paraview.org/),
 * MayaVi (http://mayavi.sourceforge.net/) or VisIt (https://wci.llnl.gov/codes/visit/)
 * to visualize results.
 * Return: 0 - all ok, 1 - error.
 */
int le_save_task(le_task *task, const char *file);

/*
 * One time step of difference scheme.
 */
void le_step(int noSteps, le_task *task);


void le_step_x_mil(const real timeStep, const le_material mat, \
	       const le_vec2 spacing, const int_t nx, \
	       const int_t ny, const le_node *inGrid, \
	       le_node *outGrid);

void le_step_y_mil(const real timeStep, const le_material mat, \
	       const le_vec2 spacing, const int_t nx, \
	       const int_t ny, const le_node *ingrid, \
	       le_node *outGrid);

/*
 * One time step of difference scheme.
 * Cache friendly version.
 */
/* void le_step_cf(le_task *task, const int cfs); */

#endif /* LE_CORE_H */
