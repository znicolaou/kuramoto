//Zachary G. Nicolaou 2/4/2024
//Dormand Prince 4/5 stepper on the GPU
#include "dp45_64.h"

//dp45 coefficients
const double a1loc[1] = {1.0/5};
const double a2loc[2] = {3.0/40, 9.0/40};
const double a3loc[3] = {44.0/45, -56.0/15, 32.0/9};
const double a4loc[4] = {19372.0/6561, -25360.0/2187, 64448.0/6561, -212.0/729};
const double a5loc[5] = {9017.0/3168, -355.0/33, 46732.0/5247, 49.0/176, -5103.0/18656};
const double b1loc[6] = {35.0/384, 0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84};
const double c[6] = {0.0, 1.0/5, 3.0/10, 4.0/5, 8.0/9, 1.0};
const double eloc[7] = {-71.0/57600, 0, 71.0/16695, -71.0/1920, 17253.0/339200, -22.0/525,1.0/40};

const double p1loc[4] = {1.0, -8048581381.0/2820520608, 8663915743.0/2820520608, -12715105075.0/11282082432};
const double p2loc[4] = {0, 0, 0, 0};
const double p3loc[4] = {0, 131558114200.0/32700410799, -68118460800.0/10900136933,87487479700.0/32700410799};
const double p4loc[4] = {0, -1754552775.0/470086768, 14199869525.0/1410260304, -10690763975.0/1880347072};
const double p5loc[4] = {0, 127303824393.0/49829197408, -318862633887.0/49829197408, 701980252875.0/199316789632};
const double p6loc[4] = {0, -282668133.0/205662961, 2019193451.0/616988883, -1453857185.0/822651844};
const double p7loc[4] = {0, 40617522.0/29380423, -110615467.0/29380423, 69997945.0/29380423};


static double *y, *ylast, *ytemp, *yerr, *y_eval, *k1, *k2, *k3, *k4, *k5, *k6, *k7;
static double *a1, *a2, *a3, *a4, *a5, *b1, *e, *p1, *p2, *p3, *p4, *p5, *p6, *p7;

static unsigned long int N;
static void (*dydt)(double, double*, double*, void*) = NULL;
static double atl, rtl, t_last;
static int fixed;
static cublasHandle_t handle;

//Steps for the DP stepper
__global__ void step2 (double* y, double* k1, double* ytemp, const double *a1, const double h, const unsigned long int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    ytemp[i]=y[i]+h*a1[0]*k1[i];
  }
}
__global__ void step3 (double* y, double* k1, double* k2, double* ytemp, const double *a2, const double h, const unsigned long int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    ytemp[i]=y[i]+h*(a2[0]*k1[i]+a2[1]*k2[i]);
  }
}
__global__ void step4 (double* y, double* k1, double* k2, double* k3, double* ytemp, const double *a3, const double h, const unsigned long int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    ytemp[i]=y[i]+h*(a3[0]*k1[i]+a3[1]*k2[i]+a3[2]*k3[i]);
  }
}
__global__ void step5 (double* y, double* k1, double* k2, double* k3, double* k4, double* ytemp, const double *a4, const double h, const unsigned long int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    ytemp[i]=y[i]+h*(a4[0]*k1[i]+a4[1]*k2[i]+a4[2]*k3[i]+a4[3]*k4[i]);
  }
}
__global__ void step6 (double* y, double* k1, double* k2, double* k3, double* k4, double* k5, double* ytemp, const double *a5, const double h, const unsigned long int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    ytemp[i]=y[i]+h*(a5[0]*k1[i]+a5[1]*k2[i]+a5[2]*k3[i]+a5[3]*k4[i]+a5[4]*k5[i]);
  }
}
__global__ void step7 (double* y, double* k1, double* k2, double* k3, double* k4, double* k5, double* k6, double *ytemp, const double *b1, const double h, const unsigned long int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    ytemp[i]=y[i]+h*(b1[0]*k1[i]+b1[1]*k2[i]+b1[2]*k3[i]+b1[3]*k4[i]+b1[4]*k5[i]+b1[5]*k6[i]);
  }
}
__global__ void interpolate (double* y, double* k1, double* k2, double* k3, double* k4, double* k5, double* k6, double* k7, double *ytemp, const double *p1, const double *p2,const double *p3,const double *p4,const double *p5,const double *p6,const double *p7, const double h, const double h2, const unsigned long int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    double x=h2/h;
    ytemp[i]=y[i]+h*(x*(p1[0]*k1[i]+p2[0]*k2[i]+p3[0]*k3[i]+p4[0]*k4[i]+p5[0]*k5[i]+p6[0]*k6[i]+p7[0]*k7[i])+x*x*(p1[1]*k1[i]+p2[1]*k2[i]+p3[1]*k3[i]+p4[1]*k4[i]+p5[1]*k5[i]+p6[1]*k6[i]+p7[1]*k7[i])+x*x*x*(p1[2]*k1[i]+p2[2]*k2[i]+p3[2]*k3[i]+p4[2]*k4[i]+p5[2]*k5[i]+p6[2]*k6[i]+p7[2]*k7[i])+x*x*x*x*(p1[3]*k1[i]+p2[3]*k2[i]+p3[3]*k3[i]+p4[3]*k4[i]+p5[3]*k5[i]+p6[3]*k6[i]+p7[3]*k7[i]));
  }
}

//Error estimate for the DP stepper
__global__ void error (double *y, double *ytemp, double* k1, double* k2, double* k3, double* k4, double* k5, double* k6, double *k7, double* yerr, const double *e, const double h, const double atl, const double rtl, const unsigned long int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    yerr[i]=h*(e[0]*k1[i]+e[1]*k2[i]+e[2]*k3[i]+e[3]*k4[i]+e[4]*k5[i]+e[5]*k6[i]+e[6]*k7[i])/(atl+rtl*fmax(fabs(y[i]),fabs(ytemp[i])));
  }
}

//Attempt a DP step
int dp45_step (double *t, double *h, void* pars){
  double norm=0;
  //Calculate the intermediate steps and error estimates using the CUDA kernels
  step2<<<(N+255)/256, 256>>>(y, k1, ytemp, a1, *h, N);
  (*dydt)((*t)+(*h)*c[1],ytemp,k2,pars);

  step3<<<(N+255)/256, 256>>>(y, k1, k2, ytemp, a2, *h, N);
  (*dydt)((*t)+(*h)*c[2],ytemp,k3,pars);

  step4<<<(N+255)/256, 256>>>(y, k1, k2, k3, ytemp, a3, *h, N);
  (*dydt)((*t)+(*h)*c[3],ytemp,k4,pars);

  step5<<<(N+255)/256, 256>>>(y, k1, k2, k3, k4, ytemp, a4, *h, N);
  (*dydt)((*t)+(*h)*c[4],ytemp,k5,pars);

  step6<<<(N+255)/256, 256>>>(y, k1, k2, k3, k4, k5, ytemp, a5, *h, N);
  (*dydt)((*t)+(*h)*c[5],ytemp,k6,pars);

  step7<<<(N+255)/256, 256>>>(y,  k1, k2, k3, k4, k5, k6, ytemp, b1, *h, N);
  (*dydt)((*t)+(*h),ytemp,k7,pars);


  if(fixed){
    t_last=*t;
    cublasDcopy(handle, N, y, 1, ylast, 1);
    cublasDcopy(handle, N, ytemp, 1, y, 1);
    (*t)=(*t)+(*h);
    return 1;
  }
  else{
    error<<<(N+255)/256, 256>>>(y, ytemp, k1, k2, k3, k4, k5, k6, k7, yerr, e, *h, atl, rtl, N);
    cublasDnrm2(handle, N, yerr, 1, &norm);
    norm/=pow(N,0.5);
    double factor=0.9*pow(norm,-0.2);
    //Accept or reject the step and update the step size
    if(norm<1){
      t_last=*t;
      cublasDcopy(handle, N, y, 1, ylast, 1);
      cublasDcopy(handle, N, ytemp, 1, y, 1);


      (*t)=(*t)+(*h);
      if (factor>10)
        factor=10;
      (*h)*=factor;
      return 1;
    }
    else if (factor<1){
      if (factor<0.2)
        factor=0.2;
      (*h)*=factor;
    }
  }
  return 0;
}

double *dp45_eval(const double t,const double t_eval){
  interpolate<<<(N+255)/256, 256>>>(ylast, k1, k2, k3, k4, k5, k6, k7, y_eval, p1, p2, p3, p4, p5, p6, p7, t-t_last, t_eval-t_last, N);
  return y_eval;
}

double* dp45_run(double *t, double *h, double t1, void *pars, void (*step_eval)(double, double, double*, void*)){

  cudaMalloc ((void**)&y_eval, N*sizeof(double));
  (*dydt)(*t,y,k1,pars);

  while(*t<t1){
    // if(*t+*h>t1)
    //   *h=t1-*t;

    int success=dp45_step (t, h, pars);
    if(success){
      (*step_eval)(*t,*h,y,pars);
      cublasDcopy(handle, N, k7, 1, k1, 1);
    }
  }
  return y;
}

double* dp45_init(int n, double atol, double rtol, int fixedstep, double *yloc, cublasHandle_t h, void (*func)(double, double*, double*, void*)){
  N=n;
  rtl=rtol;
  atl=atol;
  fixed=fixedstep;
  dydt=func;
  handle=h;

  cudaMalloc ((void**)&y, N*sizeof(double));
  cudaMalloc ((void**)&yerr, N*sizeof(double));
  cudaMalloc ((void**)&ytemp, N*sizeof(double));
  cudaMalloc ((void**)&ylast, N*sizeof(double));
  cudaMalloc ((void**)&k1, N*sizeof(double));
  cudaMalloc ((void**)&k2, N*sizeof(double));
  cudaMalloc ((void**)&k3, N*sizeof(double));
  cudaMalloc ((void**)&k4, N*sizeof(double));
  cudaMalloc ((void**)&k5, N*sizeof(double));
  cudaMalloc ((void**)&k6, N*sizeof(double));
  cudaMalloc ((void**)&k7, N*sizeof(double));

  cudaMalloc ((void**)&a1, 1*sizeof(double));
  cudaMalloc ((void**)&a2, 2*sizeof(double));
  cudaMalloc ((void**)&a3, 3*sizeof(double));
  cudaMalloc ((void**)&a4, 4*sizeof(double));
  cudaMalloc ((void**)&a5, 5*sizeof(double));
  cudaMalloc ((void**)&b1, 6*sizeof(double));
  cudaMalloc ((void**)&e, 7*sizeof(double));

  cudaMalloc ((void**)&p1, 4*sizeof(double));
  cudaMalloc ((void**)&p2, 4*sizeof(double));
  cudaMalloc ((void**)&p3, 4*sizeof(double));
  cudaMalloc ((void**)&p4, 4*sizeof(double));
  cudaMalloc ((void**)&p5, 4*sizeof(double));
  cudaMalloc ((void**)&p6, 4*sizeof(double));
  cudaMalloc ((void**)&p7, 4*sizeof(double));

  cudaMemcpy (y, yloc, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (a1, a1loc, 1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (a2, a2loc, 2*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (a3, a3loc, 3*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (a4, a4loc, 4*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (a5, a5loc, 5*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (b1, b1loc, 6*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (e, eloc, 7*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy (p1, p1loc, 4*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (p2, p2loc, 4*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (p3, p3loc, 4*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (p4, p4loc, 4*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (p5, p5loc, 4*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (p6, p6loc, 4*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (p7, p7loc, 4*sizeof(double), cudaMemcpyHostToDevice);

  return y;
}

void dp45_destroy(){
  cudaFree(y);
  cudaFree(yerr);
  cudaFree(ytemp);
  cudaFree(ylast);
  cudaFree(y_eval);
  cudaFree(k1);
  cudaFree(k2);
  cudaFree(k3);
  cudaFree(k4);
  cudaFree(k5);
  cudaFree(k6);
  cudaFree(k7);
  cudaFree(a1);
  cudaFree(a2);
  cudaFree(a3);
  cudaFree(a4);
  cudaFree(a5);
  cudaFree(b1);
  cudaFree(e);
  cudaFree(p1);
  cudaFree(p2);
  cudaFree(p3);
  cudaFree(p4);
  cudaFree(p5);
  cudaFree(p6);
  cudaFree(p7);


}
