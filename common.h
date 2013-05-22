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

#ifndef COMMON_H
#define COMMON_H

#define DEFAULT_DATASIZE 1048576  /* Default datasize. */
#define DEFAULT_REPS 10           /* Default repetitions. */
#define CONF95 1.96

extern int reps;              /* Repetitions. */
extern double *times;         /* Array to store results in. */
extern int flag;              /* Flag to set CPU or GPU invocation. */
extern unsigned int datasize; /* Datasize passed to benchmark functions. */


/*
 * Function prototypes for common functions.
 */
void init(int argc, char **argv);
void finalisetest(char *);
void finalise(void);
void benchmark(char *, double (*test)(void));
void print_results(char *, double, double);

#endif /* COMMON_H */
