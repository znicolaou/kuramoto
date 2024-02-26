//Zachary G. Nicolaou 2/4/2024
//Dormand Prince 4/5 stepper on the GPU
#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

int dp45_step (float *t, float *h, void* pars);
float* dp45_init(int n, float atl, float rtl, int fixedstep, float *yloc, cublasHandle_t h, void (*dydt)(float, float*, float*, void*));
float* dp45_run(float *t, float *h, float t1, void *pars, void (*step_eval)(float, float, float*, void*));
void dp45_destroy();
float *dp45_eval(const float t, const float t1);
