//Zachary G. Nicolaou 2/4/2024
//Dormand Prince 4/5 stepper on the GPU
#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

int dp45_step (double *t, double *h, void* pars);
double* dp45_init(int n, double atl, double rtl, int fixedstep, double *yloc, cublasHandle_t h, void (*dydt)(double, double*, double*, void*));
double* dp45_run(double *t, double *h, double t1, void *pars, void (*step_eval)(double, double, double*, void*));
void dp45_destroy();
double *dp45_eval(const double t, const double t1);
