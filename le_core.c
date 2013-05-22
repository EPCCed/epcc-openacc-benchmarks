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

#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#ifdef _OPENACC
#include <openacc.h>
#endif
#include "le_core.h"

#ifdef SAVE_EVERY_STEPS
const int save_every_step = 1;
#else
const int save_every_step = 0;
#endif


/*
 * Mapping 2d array to 1d array.
 */
#define ind(i, j) ((i) + (j) * t->n.x)
#define gind(i, j) (((le_node *) t->grid)[ind((i), (j))])
#define hind(i, j) ((i) + (j) * nx)
#define hgind(i, j) (((le_node *) grid)[hind((i), (j))])
/*
 * Vector norm.
 */
#define vnorm(v) (sqrt(v.x * v.x + v.y * v.y))

#define TVD2_EPS 1e-6

inline real le_min(real a, real b) { return a > b ? b : a; }
inline real le_max(real a, real b) { return a > b ? a : b; }
inline real le_max3(real a, real b, real c) { return le_max(a, le_max(b, c)); }

#define limiter_minmod(r) (le_max(0.0, le_min(1.0, (r))))
#define limiter_cir(r) (0.0)
#define limiter_superbee(r) (le_max3(0.0, le_min(1.0, 2.0 * r), le_min(2.0, r)))

/*
 * Set TVD limiter (http://en.wikipedia.org/wiki/Flux_limiters).
 */
#define limiter limiter_superbee

/*
 * Second order TVD scheme
 * (http://en.wikipedia.org/wiki/Total_variation_diminishing).
 */
inline real tvd2(const real c, const real u_2, const real u_1,  \
                 const real u, const real u1){
  real r1 = (u  - u_1);
  real r2 = (u1 - u);
  if (r2 == 0.0) {
    r1 += TVD2_EPS;
    r2 += TVD2_EPS;
  }
  const real r = r1 / r2;
  r1 = (u_1 - u_2);
  r2 = (u   - u_1);
  if (r2 == 0.0) {
    r1 += TVD2_EPS;
    r2 += TVD2_EPS;
  }
  const real r_1 = r1 / r2;

  const real f12  = u   + limiter(r)   / 2.0 * (1.0 - c) * (u1 - u);
  const real f_12 = u_1 + limiter(r_1) / 2.0 * (1.0 - c) * (u  - u_1);

  return u - c * (f12 - f_12);
}

void le_set_ball(le_task *t, const le_vec2 c, const real r, const real s)
{
  assert(t->stype == ST_AOS);
  int i, j;
  for (i = 0; i < t->n.x; i++) {
    for (j = 0; j < t->n.y; j++) {
      le_vec2 x = {t->h.x * i, t->h.y * j};
      le_vec2 d = {x.x - c.x, x.y - c.y};

      if (vnorm(d) < r) {
        /*
         * Set pressure disturbance, in both the main grid and the
         * temporary copy
         */
        t->grid[i + j*t->n.x].s.xx = s;
        t->grid[i + j*t->n.x].s.yy = s;
        t->tmpGrid[i + j*t->n.x].s.xx = s;
        t->tmpGrid[i + j*t->n.x].s.yy = s;
      }
    }
  }
}

/*
 * Write float to file and reverse byte order.
 */
void write_float(FILE* f, const float v)
{
  union {
    float f;
    unsigned char b[4];
  } dat1, dat2;
  dat1.f = v;
  dat2.b[0] = dat1.b[3];
  dat2.b[1] = dat1.b[2];
  dat2.b[2] = dat1.b[1];
  dat2.b[3] = dat1.b[0];
  fwrite(dat2.b, sizeof(unsigned char), 4, f);
}


void le_init_task(le_task *task, const real dt, const le_vec2 h, \
                  const le_material mat, const le_point2 n, \
                  const int stype){
  task->dt  = dt;
  task->h   = h;
  task->mat = mat;
  task->n   = n;
  task->grid = (le_node *)malloc(sizeof(le_node) * n.x * n.y);
  task->tmpGrid = (le_node *)malloc(sizeof(le_node) * n.x * n.y);
  task->stype = stype;
  memset((real *) task->grid, 0, sizeof(le_node) * n.x * n.y);
  memset((real *) task->tmpGrid, 0, sizeof(le_node) * n.x * n.y);
}

void le_free_task(le_task* task){
  free(task->grid);
  free(task->tmpGrid);
}

int le_save_task(le_task *t, const char *file){
  assert(t->stype == ST_AOS);
  int i, j;
  FILE *fp = fopen(file, "w");
  if (fp == NULL) {
    perror("Failed to open file");
    return 1;
  }
  fprintf(fp, "# vtk DataFile Version 3.0\n");
  fprintf(fp, "Created by le_save_task\n");
  fprintf(fp, "BINARY\n");
  fprintf(fp, "DATASET STRUCTURED_POINTS\n");
  fprintf(fp, "DIMENSIONS %d %d 1\n", t->n.x, t->n.y);
  fprintf(fp, "SPACING %f %f 0.0\n", t->h.x, t->h.y);
  fprintf(fp, "ORIGIN 0.0 0.0 0.0\n");
  fprintf(fp, "POINT_DATA %d\n", t->n.x * t->n.y);

  /* velocity */
  fprintf(fp, "SCALARS v float 1\n");
  fprintf(fp, "LOOKUP_TABLE v_table\n");
  for (j = 0; j < t->n.y; j++) {
    for (i = 0; i < t->n.x; i++) {
      float v = vnorm(gind(i, j).v);
      write_float(fp, v);
    }
  }

  /*
   * You can use the same code for saving other variables.
   */
  fclose(fp);
  return 0;
}

void le_init_material(const real c1, const real c2, const real rho, \
                      le_material *m){
  m->c1 = c1;
  m->c2 = c2;
  m->rho = rho;

  /*
   * Cached values.
   */
  m->irhoc1 = 1.0 / (c1 * rho);
  m->irhoc2 = 1.0 / (c2 * rho);
  m->rhoc1 = c1 * rho;
  m->rhoc2 = c2 * rho;
  real mu = rho * c2 * c2;
  real la = rho * c1 * c1 - 2.0 * mu;
  m->rhoc3 = rho * c1 * la / (la + 2.0 * mu);
}



inline void inc_x(const le_material *m, le_node *n, const le_w *d){
  const real d1 = 0.5 * d->w1;
  const real d2 = 0.5 * d->w2;
  const real d3 = 0.5 * d->w3;
  const real d4 = 0.5 * d->w4;

  n->v.x += d1 + d2;
  n->v.y += d3 + d4;

  n->s.xx += (d2 - d1) * m->rhoc1;
  n->s.yy += (d2 - d1) * m->rhoc3;
  n->s.xy += m->rhoc2 * (d4 - d3);
}

inline void inc_y(const le_material *m, le_node *n, const le_w *d){
  const real d1 = 0.5 * d->w1;
  const real d2 = 0.5 * d->w2;
  const real d3 = 0.5 * d->w3;
  const real d4 = 0.5 * d->w4;

  n->v.y += d1 + d2;
  n->v.x += d3 + d4;

  n->s.yy += (d2 - d1) * m->rhoc1;
  n->s.xx += (d2 - d1) * m->rhoc3;
  n->s.xy += m->rhoc2 * (d4 - d3);
}

inline void reconstruct(const le_w ppu, const le_w pu, const le_w u, \
                        const le_w nu, const le_w nnu, const real k1, \
                        const real k2, le_w *d){
  d->w1 = tvd2(k1, ppu.w1, pu.w1, u.w1, nu.w1) - u.w1; // c1
  d->w2 = tvd2(k1, nnu.w2, nu.w2, u.w2, pu.w2) - u.w2; // -c1
  d->w3 = tvd2(k2, ppu.w3, pu.w3, u.w3, nu.w3) - u.w3; // c2
  d->w4 = tvd2(k2, nnu.w4, nu.w4, u.w4, pu.w4) - u.w4; // -c2
}

/*
 * Due to our system of pde is linear, we can use some simple way to
 * solve it.  du/dt + A * du/dx = 0.  Vector u = {vx, vy, sxx, sxy,
 * syy}.  Matrix A could be represent in form OmegaL * Lambda *
 * OmegaR, where Lambda - diagonal matrix of eigen values of matrix A.
 * In our case Lambda = diag{c1, -c1, c2, -c2, 0}.  OmegaR and OmegaL
 * - metrices from eigen vectors of matrix A, OmegaR * OmegaL = E,
 * where E = diag{1, 1, 1, 1, 1}.
 *
 * We can rewrite out system in form: du/dt + OmegaL * Lambda * OmegaR
 * du/dx = 0, multiply on matrix OmegaR:
*
 * OmegaR * du/dt + OmegaR * OmegaL * Lambda * OmegaR du/dx = 0.
 *
 * Introduce new variables
 * (http://en.wikipedia.org/wiki/Riemann_invariant):
 * w = {w1, w2, w3, w4, w5},
 * w = OmegaR * u, then we got:
 *
 * dw/dt + Lambda * dw/dx = 0.
 *
 * And we get system of independent advection equations, that we can
 * solve separatly.
 *
 * So we get next algorithm:
 * 1. Introduce new variables w = OmegaR * u;
 * 2. Solve 5 equations of linear advection (in real we solve only
 *    4, because of in fifth equation speed is 0);
 * 3. Make inverse transformation u = OmegaL * w.
 */


void le_step_x_mil(const real timeStep, const le_material mat, const le_vec2 spacing, const int_t nx, const int_t ny, const le_node *inGrid, le_node *outGrid){
  /* Courant number (http://en.wikipedia.org/wiki/ \
     Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition). */
  const real k1 = timeStep * mat.c1 / spacing.x;
  const real k2 = timeStep * mat.c2 / spacing.x;
  int i,j;

  //#pragma acc kernels loop independent, present(inGrid[0:nx*ny],outGrid[0:nx*ny])
#pragma acc parallel loop gang, present(inGrid[0:nx*ny],outGrid[0:nx*ny])
  for (j = 0; j < ny; j++) {
#pragma acc loop vector
    for (i = 0; i < nx; i++){
      /* Riemann invariants for 5-point sctencil difference scheme.
       */
      le_w w_2, w_1, w, w1, w2, d;
      real r1, r2, r, r_1, min1, min2, max1, max2, f12, f_12;
      real nv;
      real N00T;
      real n1v;
      real N01T;

      /* omega_x(&mat, &inGrid[((i>1)     ?(i-2):0)  + j*nx],   &w_2); */
      /* omega_x(&mat, &inGrid[((i>0)     ?(i-1):0)  + j*nx],   &w_1); */
      /* omega_x(&mat, &inGrid[ i                    + j*nx],   &w); */
      /* omega_x(&mat, &inGrid[((i<(nx-1))?i+1:nx-1) + j*nx],   &w1); */
      /* omega_x(&mat, &inGrid[((i<(nx-2))?i+2:nx-1) + j*nx],   &w2); */

      /* omega_x(&mat, &inGrid[((i>1)     ?(i-2):0)  + j*nx],   &w_2); */
      {
        nv = inGrid[((i>1)     ?(i-2):0)  + j*nx].v.x;
        N00T = inGrid[((i>1)     ?(i-2):0)  + j*nx].s.xx * mat.irhoc1;

        n1v = inGrid[((i>1)     ?(i-2):0)  + j*nx].v.y;
        N01T = inGrid[((i>1)     ?(i-2):0)  + j*nx].s.xy * mat.irhoc2;

        w_2.w1 = nv  - N00T;
        w_2.w2 = nv  + N00T;
        w_2.w3 = n1v - N01T;
        w_2.w4 = n1v + N01T;
      }


      /* omega_x(&mat, &inGrid[((i>0)     ?(i-1):0)  + j*nx],   &w_1); */
      {
        nv = inGrid[((i>1)     ?(i-1):0)  + j*nx].v.x;
        N00T = inGrid[((i>1)     ?(i-1):0)  + j*nx].s.xx * mat.irhoc1;

        n1v = inGrid[((i>1)     ?(i-1):0)  + j*nx].v.y;
        N01T = inGrid[((i>1)     ?(i-1):0)  + j*nx].s.xy * mat.irhoc2;

        w_1.w1 = nv  - N00T;
        w_1.w2 = nv  + N00T;
        w_1.w3 = n1v - N01T;
        w_1.w4 = n1v + N01T;
      }

      /* omega_x(&mat, &inGrid[ i                    + j*nx],   &w); */
      {
        nv = inGrid[ i                    + j*nx].v.x;
        N00T = inGrid[ i                    + j*nx].s.xx * mat.irhoc1;

        n1v = inGrid[ i                    + j*nx].v.y;
        N01T = inGrid[ i                    + j*nx].s.xy * mat.irhoc2;

        w.w1 = nv  - N00T;
        w.w2 = nv  + N00T;
        w.w3 = n1v - N01T;
        w.w4 = n1v + N01T;
      }


      /* omega_x(&mat, &inGrid[((i<(nx-1))?i+1:nx-1) + j*nx],   &w1); */
      {
        nv = inGrid[((i<(nx-1))?i+1:nx-1) + j*nx].v.x;
        N00T = inGrid[((i<(nx-1))?i+1:nx-1) + j*nx].s.xx * mat.irhoc1;

        n1v = inGrid[((i<(nx-1))?i+1:nx-1) + j*nx].v.y;
        N01T = inGrid[((i<(nx-1))?i+1:nx-1) + j*nx].s.xy * mat.irhoc2;

        w1.w1 = nv  - N00T;
        w1.w2 = nv  + N00T;
        w1.w3 = n1v - N01T;
        w1.w4 = n1v + N01T;
      }


      /* omega_x(&mat, &inGrid[((i<(nx-2))?i+2:nx-1) + j*nx],   &w2); */
      {
        nv = inGrid[((i<(nx-2))?i+2:nx-1) + j*nx].v.x;
        N00T = inGrid[((i<(nx-2))?i+2:nx-1) + j*nx].s.xx * mat.irhoc1;

        n1v = inGrid[((i<(nx-2))?i+2:nx-1) + j*nx].v.y;
        N01T = inGrid[((i<(nx-2))?i+2:nx-1) + j*nx].s.xy * mat.irhoc2;

        w2.w1 = nv  - N00T;
        w2.w2 = nv  + N00T;
        w2.w3 = n1v - N01T;
        w2.w4 = n1v + N01T;
      }




      /* reconstruct(w_2, w_1, w, w1, w2, k1, k2, &d); */
      {
        //d.w1 = tvd2(k1, w_2.w1, w_1.w1, w.w1, w1.w1) - w.w1; // c1
        {
          r1 = (w.w1  - w_1.w1);
          r2 = (w1.w1 - w.w1);
          if (r2 == 0.0) {
            r1 += TVD2_EPS;
            r2 += TVD2_EPS;
          }
          r = r1 / r2;
          r1 = (w_1.w1 - w_2.w1);
          r2 = (w.w1   - w_1.w1);
          if (r2 == 0.0) {
            r1 += TVD2_EPS;
            r2 += TVD2_EPS;
          }
          r_1 = r1 / r2;

          // Make some temp vars
          min1 = (1.0 > 2.0*r ? 2.0*r:1.0);
          min2 = (2.0 > r     ? r: 2.0  );
          max1 = (min1 > min2 ? min1 : min2);
          max2 = (0.0 > max1 ? 0.0 : max1);

          f12  = w.w1   + max2   / 2.0 * (1.0 - k1) * (w1.w1 - w.w1);

          min1 = (1.0 > 2.0*r_1 ? 2.0*r_1: 1.0);
          min2 = (2.0 > r_1     ? r_1 : 2.0  );


          max1 = (min1 > min2 ? min1 : min2);
          max2 = (0.0 > max1 ? 0.0 : max1);

          f_12 = w_1.w1 + max2 / 2.0 * (1.0 - k1) * (w.w1  - w_1.w1);

          d.w1 = (w.w1 - k1 * (f12 - f_12)) - w.w1;
        }


        //d.w2 = tvd2(k1, w2.w2, w1.w2, w.w2, w_1.w2) - w.w2; // -c1
        {
          r1 = (w.w2  - w1.w2);
          r2 = (w_1.w2 - w.w2);
          if (r2 == 0.0) {
            r1 += TVD2_EPS;
            r2 += TVD2_EPS;
          }
          r = r1 / r2;
          r1 = (w1.w2 - w2.w2);
          r2 = (w.w2   - w1.w2);
          if (r2 == 0.0) {
            r1 += TVD2_EPS;
            r2 += TVD2_EPS;
          }
          r_1 = r1 / r2;
          min1 = (1.0 > 2.0*r ? 2.0*r:1.0);
          min2 = (2.0 > r     ? r: 2.0  );

          max1 = (min1 > min2 ? min1 : min2);
          max2 = (0.0 > max1 ? 0.0 : max1);

          f12  = w.w2   + max2   / 2.0 * (1.0 - k1) * (w_1.w2 - w.w2);

          min1 = (1.0 > 2.0*r_1 ? 2.0*r_1: 1.0);
          min2 = (2.0 > r_1     ? r_1 : 2.0  );


          max1 = (min1 > min2 ? min1 : min2);
          max2 = (0.0 > max1 ? 0.0 : max1);

          f_12 = w1.w2 + max2 / 2.0 * (1.0 - k1) * (w.w2  - w1.w2);

          d.w2 =  (w.w2 - k1 * (f12 - f_12)) - w.w2;

        }

        //      d.w3 = tvd2(k2, w_2.w3, w_1.w3, w.w3, w1.w3) - w.w3; // c2
        {
          //inline real tvd2(const real c, const real u_2, const real u_1, const real u, const real u1)
          r1 = (w.w3  - w_1.w3);
          r2 = (w1.w3 - w.w3);
          if (r2 == 0.0) {
            r1 += TVD2_EPS;
            r2 += TVD2_EPS;
          }
          r = r1 / r2;
          r1 = (w_1.w3 - w_2.w3);
          r2 = (w.w3   - w_1.w3);
          if (r2 == 0.0) {
            r1 += TVD2_EPS;
            r2 += TVD2_EPS;
          }
          r_1 = r1 / r2;


          min1 = (1.0 > 2.0*r ? 2.0*r:1.0);
          min2 = (2.0 > r     ? r: 2.0  );
          max1 = (min1 > min2 ? min1 : min2);
          max2 = (0.0 > max1 ? 0.0 : max1);

          f12  = w.w3   + max2   / 2.0 * (1.0 - k2) * (w1.w3 - w.w3);

          min1 = (1.0 > 2.0*r_1 ? 2.0*r_1: 1.0);
          min2 = (2.0 > r_1     ? r_1 : 2.0  );


          max1 = (min1 > min2 ? min1 : min2);
          max2 = (0.0 > max1 ? 0.0 : max1);
          f_12 = w_1.w3 + max2 / 2.0 * (1.0 - k2) * (w.w3  - w_1.w3);

          d.w3 =   (w.w3 - k2 * (f12 - f_12)) - w.w3;
        }

        //      d.w4 = tvd2(k2, w2.w4, w1.w4, w.w4, w_1.w4) - w.w4; // -c2
        {

          //inline real tvd2(const real c, const real u_2, const real u_1, const real u, const real u1)
          r1 = (w.w4  - w1.w4);
          r2 = (w_1.w4 - w.w4);
          if (r2 == 0.0) {
            r1 += TVD2_EPS;
            r2 += TVD2_EPS;
          }
          r = r1 / r2;
          r1 = (w1.w4 - w2.w4);
          r2 = (w.w4   - w1.w4);
          if (r2 == 0.0) {
            r1 += TVD2_EPS;
            r2 += TVD2_EPS;
          }
          r_1 = r1 / r2;


          min1 = (1.0 > 2.0*r ? 2.0*r:1.0);
          min2 = (2.0 > r     ? r: 2.0  );

          max1 = (min1 > min2 ? min1 : min2);
          max2 = (0.0 > max1 ? 0.0 : max1);
          f12  = w.w4   + max2   / 2.0 * (1.0 - k2) * (w_1.w4 - w.w4);

          min1 = (1.0 > 2.0*r_1 ? 2.0*r_1: 1.0);
          min2 = (2.0 > r_1     ? r_1 : 2.0  );

          max1 = (min1 > min2 ? min1 : min2);
          max2 = (0.0 > max1 ? 0.0 : max1);
          f_12 = w1.w4 + max2 / 2.0 * (1.0 - k2) * (w.w4  - w1.w4);

          d.w4 =  (w.w4 - k2 * (f12 - f_12)) - w.w4;
        }


      }







      outGrid[i+j*nx] = inGrid[i+j*nx];
      /* inc_x(&mat, &outGrid[i+j*nx], &d); */

      const real d1 = 0.5 * d.w1;
      const real d2 = 0.5 * d.w2;
      const real d3 = 0.5 * d.w3;
      const real d4 = 0.5 * d.w4;

      outGrid[i+j*nx].v.x += d1 + d2;
      outGrid[i+j*nx].v.y += d3 + d4;

      outGrid[i+j*nx].s.xx += (d2 - d1) * mat.rhoc1;
      outGrid[i+j*nx].s.yy += (d2 - d1) * mat.rhoc3;
      outGrid[i+j*nx].s.xy += mat.rhoc2 * (d4 - d3);




    }
  }
}



void le_step_y_mil(const real timeStep, const le_material mat, const le_vec2 spacing, const int_t nx, const int_t ny, const le_node *inGrid, le_node *outGrid){
  /* Courant number (http://en.wikipedia.org/wiki/ \
     Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition). */
  const real k1 = timeStep * mat.c1 / spacing.y;
  const real k2 = timeStep * mat.c2 / spacing.y;
  int i,j;

#pragma acc parallel loop gang, present(inGrid[0:nx*ny],outGrid[0:nx*ny])
  for (j = 0; j < ny; j++) {
#pragma acc loop vector
    for (i = 0; i < nx; i++){
      /* Riemann invariants for 5-point sctencil difference scheme.
       */
      le_w w_2, w_1, w, w1, w2, d;
      real r1, r2, r, r_1, min1, min2, max1, max2, f12, f_12;
      real nv;
      real N00T;
      real n1v;
      real N01T;




      /*       omega_y(&mat, &inGrid[i + ((j>1)   ?(j-2):0)   *ny], &w_2); */
      /*       omega_y(&mat, &inGrid[i + ((j>0)   ?(j-1):0)   *ny], &w_1); */
      /*       omega_y(&mat, &inGrid[i +   j                  *ny], &w); */
      /*       omega_y(&mat, &inGrid[i + ((j<(ny-1))?j+1:ny-1)*ny], &w1); */
      /*       omega_y(&mat, &inGrid[i + ((j<(ny-2))?j+2:ny-1)*ny], &w2); */

      /* omega_y(&mat, &inGrid[i + ((j>1)   ?(j-2):0)   *ny], &w_2); */

      nv = inGrid[i + ((j>1)   ?(j-2):0)   *nx].v.y;
      N00T = inGrid[i + ((j>1)   ?(j-2):0)   *nx].s.yy * mat.irhoc1;

      n1v = inGrid[i + ((j>1)   ?(j-2):0)   *nx].v.x;
      N01T = inGrid[i + ((j>1)   ?(j-2):0)   *nx].s.xy * mat.irhoc2;

      w_2.w1 = nv  - N00T;
      w_2.w2 = nv  + N00T;
      w_2.w3 = n1v - N01T;
      w_2.w4 = n1v + N01T;



      /* omega_y(&mat, &inGrid[i + ((j>0)   ?(j-1):0)   *ny], &w_1); */

      nv = inGrid[i + ((j>0)   ?(j-1):0)   *nx].v.y;
      N00T = inGrid[i + ((j>0)   ?(j-1):0)   *nx].s.yy * mat.irhoc1;

      n1v = inGrid[i + ((j>0)   ?(j-1):0)   *nx].v.x;
      N01T = inGrid[i + ((j>0)   ?(j-1):0)   *nx].s.xy * mat.irhoc2;

      w_1.w1 = nv  - N00T;
      w_1.w2 = nv  + N00T;
      w_1.w3 = n1v - N01T;
      w_1.w4 = n1v + N01T;


      /* omega_y(&mat, &inGrid[i +   j                  *ny], &w); */

      nv = inGrid[ i                    + j*nx].v.y;
      N00T = inGrid[ i                    + j*nx].s.yy * mat.irhoc1;

      n1v = inGrid[ i                    + j*nx].v.x;
      N01T = inGrid[ i                    + j*nx].s.xy * mat.irhoc2;

      w.w1 = nv  - N00T;
      w.w2 = nv  + N00T;
      w.w3 = n1v - N01T;
      w.w4 = n1v + N01T;


      /* omega_y(&mat, &inGrid[i + ((j<(ny-1))?j+1:ny-1)*ny], &w1); */

      nv = inGrid[i + ((j<(ny-1))?j+1:ny-1)*nx].v.y;
      N00T = inGrid[i + ((j<(ny-1))?j+1:ny-1)*nx].s.yy * mat.irhoc1;

      n1v = inGrid[i + ((j<(ny-1))?j+1:ny-1)*nx].v.x;
      N01T = inGrid[i + ((j<(ny-1))?j+1:ny-1)*nx].s.xy * mat.irhoc2;

      w1.w1 = nv  - N00T;
      w1.w2 = nv  + N00T;
      w1.w3 = n1v - N01T;
      w1.w4 = n1v + N01T;


      /* omega_y(&mat, &inGrid[i + ((j<(ny-2))?j+2:ny-1)*ny], &w2); */

      nv = inGrid[i + ((j<(ny-2))?j+2:ny-1)*nx].v.y;
      N00T = inGrid[i + ((j<(ny-2))?j+2:ny-1)*nx].s.yy * mat.irhoc1;

      n1v = inGrid[i + ((j<(ny-2))?j+2:ny-1)*nx].v.x;
      N01T = inGrid[i + ((j<(ny-2))?j+2:ny-1)*nx].s.xy * mat.irhoc2;

      w2.w1 = nv  - N00T;
      w2.w2 = nv  + N00T;
      w2.w3 = n1v - N01T;
      w2.w4 = n1v + N01T;





      /* reconstruct(w_2, w_1, w, w1, w2, k1, k2, &d); */

      //d.w1 = tvd2(k1, w_2.w1, w_1.w1, w.w1, w1.w1) - w.w1; // c1

      r1 = (w.w1  - w_1.w1);
      r2 = (w1.w1 - w.w1);
      if (r2 == 0.0) {
        r1 += TVD2_EPS;
        r2 += TVD2_EPS;
      }
      r = r1 / r2;
      r1 = (w_1.w1 - w_2.w1);
      r2 = (w.w1   - w_1.w1);
      if (r2 == 0.0) {
        r1 += TVD2_EPS;
        r2 += TVD2_EPS;
      }
      r_1 = r1 / r2;

      // Make some temp vars
      min1 = (1.0 > 2.0*r ? 2.0*r:1.0);
      min2 = (2.0 > r     ? r: 2.0  );

      max1 = (min1 > min2 ? min1 : min2);
      max2 = (0.0 > max1 ? 0.0 : max1);

      f12  = w.w1   + max2   / 2.0 * (1.0 - k1) * (w1.w1 - w.w1);
      min1 = (1.0 > 2.0*r_1 ? 2.0*r_1: 1.0);
      min2 = (2.0 > r_1     ? r_1 : 2.0  );

      max1 = (min1 > min2 ? min1 : min2);
      max2 = (0.0 > max1 ? 0.0 : max1);

      f_12 = w_1.w1 + max2 / 2.0 * (1.0 - k1) * (w.w1  - w_1.w1);

      d.w1 = (w.w1 - k1 * (f12 - f_12)) - w.w1;



      //d.w2 = tvd2(k1, w2.w2, w1.w2, w.w2, w_1.w2) - w.w2; // -c1

      r1 = (w.w2  - w1.w2);
      r2 = (w_1.w2 - w.w2);
      if (r2 == 0.0) {
        r1 += TVD2_EPS;
        r2 += TVD2_EPS;
      }
      r = r1 / r2;
      r1 = (w1.w2 - w2.w2);
      r2 = (w.w2   - w1.w2);
      if (r2 == 0.0) {
        r1 += TVD2_EPS;
        r2 += TVD2_EPS;
      }
      r_1 = r1 / r2;


      min1 = (1.0 > 2.0*r ? 2.0*r:1.0);
      min2 = (2.0 > r     ? r: 2.0  );

      max1 = (min1 > min2 ? min1 : min2);
      max2 = (0.0 > max1 ? 0.0 : max1);

      f12  = w.w2   + max2   / 2.0 * (1.0 - k1) * (w_1.w2 - w.w2);
      min1 = (1.0 > 2.0*r_1 ? 2.0*r_1: 1.0);
      min2 = (2.0 > r_1     ? r_1 : 2.0  );

      max1 = (min1 > min2 ? min1 : min2);
      max2 = (0.0 > max1 ? 0.0 : max1);

      f_12 = w1.w2 + max2 / 2.0 * (1.0 - k1) * (w.w2  - w1.w2);

      d.w2 =  (w.w2 - k1 * (f12 - f_12)) - w.w2;



      //      d.w3 = tvd2(k2, w_2.w3, w_1.w3, w.w3, w1.w3) - w.w3; // c2

      //inline real tvd2(const real c, const real u_2, const real u_1, const real u, const real u1)
      r1 = (w.w3  - w_1.w3);
      r2 = (w1.w3 - w.w3);
      if (r2 == 0.0) {
        r1 += TVD2_EPS;
        r2 += TVD2_EPS;
      }
      r = r1 / r2;
      r1 = (w_1.w3 - w_2.w3);
      r2 = (w.w3   - w_1.w3);
      if (r2 == 0.0) {
        r1 += TVD2_EPS;
        r2 += TVD2_EPS;
      }
      r_1 = r1 / r2;


      min1 = (1.0 > 2.0*r ? 2.0*r:1.0);
      min2 = (2.0 > r     ? r: 2.0  );
      max1 = (min1 > min2 ? min1 : min2);
      max2 = (0.0 > max1 ? 0.0 : max1);

      f12  = w.w3   + max2   / 2.0 * (1.0 - k2) * (w1.w3 - w.w3);
      min1 = (1.0 > 2.0*r_1 ? 2.0*r_1: 1.0);
      min2 = (2.0 > r_1     ? r_1 : 2.0  );
      max1 = (min1 > min2 ? min1 : min2);
      max2 = (0.0 > max1 ? 0.0 : max1);
      f_12 = w_1.w3 + max2 / 2.0 * (1.0 - k2) * (w.w3  - w_1.w3);

      d.w3 =   (w.w3 - k2 * (f12 - f_12)) - w.w3;


      //      d.w4 = tvd2(k2, w2.w4, w1.w4, w.w4, w_1.w4) - w.w4; // -c2


      //inline real tvd2(const real c, const real u_2, const real u_1, const real u, const real u1)
      r1 = (w.w4  - w1.w4);
      r2 = (w_1.w4 - w.w4);
      if (r2 == 0.0) {
        r1 += TVD2_EPS;
        r2 += TVD2_EPS;
      }
      r = r1 / r2;
      r1 = (w1.w4 - w2.w4);
      r2 = (w.w4   - w1.w4);
      if (r2 == 0.0) {
        r1 += TVD2_EPS;
        r2 += TVD2_EPS;
      }
      r_1 = r1 / r2;

      min1 = (1.0 > 2.0*r ? 2.0*r:1.0);
      min2 = (2.0 > r     ? r: 2.0  );
      max1 = (min1 > min2 ? min1 : min2);
      max2 = (0.0 > max1 ? 0.0 : max1);
      f12  = w.w4   + max2   / 2.0 * (1.0 - k2) * (w_1.w4 - w.w4);

      min1 = (1.0 > 2.0*r_1 ? 2.0*r_1: 1.0);
      min2 = (2.0 > r_1     ? r_1 : 2.0  );
      max1 = (min1 > min2 ? min1 : min2);
      max2 = (0.0 > max1 ? 0.0 : max1);
      f_12 = w1.w4 + max2 / 2.0 * (1.0 - k2) * (w.w4  - w1.w4);

      d.w4 =  (w.w4 - k2 * (f12 - f_12)) - w.w4;


      outGrid[i+j*nx] = inGrid[i+j*nx];
      /* inc_y(&mat, &outGrid[i+j*nx], &d); */

      const real d1 = 0.5 * d.w1;
      const real d2 = 0.5 * d.w2;
      const real d3 = 0.5 * d.w3;
      const real d4 = 0.5 * d.w4;

      outGrid[i+j*nx].v.y += d1 + d2;
      outGrid[i+j*nx].v.x += d3 + d4;

      outGrid[i+j*nx].s.yy += (d2 - d1) * mat.rhoc1;
      outGrid[i+j*nx].s.xx += (d2 - d1) * mat.rhoc3;
      outGrid[i+j*nx].s.xy += mat.rhoc2 * (d4 - d3);

    }
  }
}




/*
 * We solve regular hyperbolic system of PDE
 * (http://en.wikipedia.org/wiki/Hyperbolic_partial_differential_equation)
 * in form: du/dt + Ax * du/dx + Ay * du/dy = 0.
 *
 * During time integration we use dimension split method:
 * 1. Step:
 * Integrate system dv/dt + Ax * dv/dx = 0, get u = v^(n + 1).
 *
 * 2. Step:
 * Integrate system du/dt + Ay * du/dy = 0, get on next time step
 * u^(n + 1).
 */
void le_step(int noSteps, le_task *task){

  int x,y, i;
  x = task->n.x;
  y = task->n.y;
  le_node *gr = task->grid;
  le_node *tgr = task->tmpGrid;


#pragma acc data copy(gr[0:x*y], tgr[0:x*y])
  {
    for(i=0; i<noSteps; i++){
      le_step_x_mil(task->dt, task->mat, task->h, task->n.x, task->n.y, task->grid, task->tmpGrid);
      le_step_y_mil(task->dt, task->mat, task->h, task->n.x, task->n.y, task->tmpGrid, task->grid);
    }
  }

}

double le_main()
{
  
  /* For the purposes of this benchmark, fix sizes at 2000 x 2000, 400 steps */
  int nx, ny, steps;
  nx = 2000;
  ny = 2000;
  steps = 400;
  le_point2 n = {nx, ny};
  le_task task;
  le_material mat;
  le_vec2 h = {1.0, 1.0};
  real dt = 0.3;
  le_vec2 center = {n.x / 2, n.y / 2};
  
  double t;
  
  /*
   * Init material.
   */
  le_init_material(2.0, 1.0, 1.5, &mat);
  
  /*
   * Init task.
   */
  le_init_task(&task, dt, h, mat, n, ST_AOS);
  
  /*
   * Initial condition.
   */
  le_set_ball(&task, center, 10.0, 1.0);
  
  t = timer();

  le_step(steps, &task);
  
  t = timer() - t;
  
  /*
   * Save last step.
   */
  /* le_save_task(&task, "result.vtk"); */
  
  /*
   * Free memory.
   */
  le_free_task(&task);
  return(t);
}

