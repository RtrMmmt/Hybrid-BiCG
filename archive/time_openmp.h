#ifndef TIME_OPENMP_H
#define TIME_OPENMP_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

void openmp_start_time(double *start_time_local);
void openmp_end_time(double start_time_local, double result_time_global);

#endif // TIME_OPENMP_H