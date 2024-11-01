//Zachary G. Nicolaou 5/13/2021
//Pseudospectral methods to integrate the complex ginzburg landau equation and stuart landau oscillators in two dimensions with adaptive Runke Kutta Fehlberg timestepping on a gpu
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cusparse.h>
#include "cublas_v2.h"
#define PI 3.141592653589793

//Rkf45 coefficients
const double a1[1] = {1.0/4};
const double a2[2] = {3.0/32, 9.0/32};
const double a3[3] = {1932.0/2197, -7200.0/2197, 7296.0/2197};
const double a4[4] = {439.0/216, -8.0, 3680.0/513, -845.0/4104};
const double a5[5] = {-8.0/27, 2.0, -3544.0/2565, 1859.0/4104, -11.0/40};
const double b1[6] = {16.0/135, 0.0, 6656.0/12825, 28561.0/56430, -9.0/50, 2.0/55};
const double b2[6] = {25.0/216, 0.0, 1408.0/2565, 2197.0/4104, -1.0/5, 0.0};
const double c[6] = {0.0, 1.0/4, 3.0/8, 12.0/13, 1.0, 1.0/2};
const cuDoubleComplex alpha={1,0},beta={0,0};

__global__ void tot_2dcgle (cuDoubleComplex* y, cuDoubleComplex* f, double *omegas, const double c1, const double c3) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  double amp=(y[i].x*y[i].x+y[i].y*y[i].y);
  double temp=omegas[i]*y[i].x+(f[i].x-c1*f[i].y)-amp*(y[i].x+c3*y[i].y);
  f[i].y=omegas[i]*y[i].y+(f[i].y+c1*f[i].x)-amp*(y[i].y-c3*y[i].x);
  f[i].x=temp;
}
//this is really kuramoto, since coupling is only Im(conjugate(z_i)*r_i)
__global__ void tot_stuartlandau (cuDoubleComplex* y, cuDoubleComplex* f, double *omegas, const double c1, const double c3) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  double amp=(y[i].x*y[i].x+y[i].y*y[i].y);
  double coupling=c1*(y[i].y*f[i].x-y[i].x*f[i].y);
  double temp=-omegas[i]*y[i].y+y[i].y*coupling+c3*y[i].x*(1-amp);
  f[i].y=omegas[i]*y[i].x-y[i].x*coupling+c3*y[i].y*(1-amp);
  f[i].x=temp;
}

__global__ void step3 (cuDoubleComplex* y, cuDoubleComplex* k1, cuDoubleComplex* k2, cuDoubleComplex* ytemp, const double a20, const double a21) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  ytemp[i].x=y[i].x+a20*k1[i].x+a21*k2[i].x;
  ytemp[i].y=y[i].y+a20*k1[i].y+a21*k2[i].y;
}

__global__ void step4 (cuDoubleComplex* y, cuDoubleComplex* k1, cuDoubleComplex* k2, cuDoubleComplex* k3, cuDoubleComplex* ytemp, const double a30, const double a31, const double a32) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  ytemp[i].x=y[i].x+a30*k1[i].x+a31*k2[i].x+a32*k3[i].x;
  ytemp[i].y=y[i].y+a30*k1[i].y+a31*k2[i].y+a32*k3[i].y;
}

__global__ void step5 (cuDoubleComplex* y, cuDoubleComplex* k1, cuDoubleComplex* k2, cuDoubleComplex* k3, cuDoubleComplex* k4, cuDoubleComplex* ytemp, const double a40, const double a41, const double a42, const double a43) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  ytemp[i].x=y[i].x+a40*k1[i].x+a41*k2[i].x+a42*k3[i].x+a43*k4[i].x;
  ytemp[i].y=y[i].y+a40*k1[i].y+a41*k2[i].y+a42*k3[i].y+a43*k4[i].y;
}

__global__ void step6 (cuDoubleComplex* y, cuDoubleComplex* k1, cuDoubleComplex* k2, cuDoubleComplex* k3, cuDoubleComplex* k4, cuDoubleComplex* k5, cuDoubleComplex* ytemp, const double a50, const double a51, const double a52, const double a53, const double a54) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  ytemp[i].x=y[i].x+a50*k1[i].x+a51*k2[i].x+a52*k3[i].x+a53*k4[i].x+a54*k5[i].x;
  ytemp[i].y=y[i].y+a50*k1[i].y+a51*k2[i].y+a52*k3[i].y+a53*k4[i].y+a54*k5[i].y;
}

__global__ void error (cuDoubleComplex* y, cuDoubleComplex* k1, cuDoubleComplex* k2, cuDoubleComplex* k3, cuDoubleComplex* k4, cuDoubleComplex* k5, cuDoubleComplex* k6, cuDoubleComplex* yerr, cuDoubleComplex* ytemp, const double atl, const double rtl, const double a50, const double a51, const double a52, const double a53, const double a54, const double a55) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  ytemp[i].x=atl+rtl*y[i].x;
  ytemp[i].y=rtl*y[i].y;
  double tempx=(a50*k1[i].x+a51*k2[i].x+a52*k3[i].x+a53*k4[i].x+a54*k5[i].x+a55*k6[i].x);
  double tempy=(a50*k1[i].y+a51*k2[i].y+a52*k3[i].y+a53*k4[i].y+a54*k5[i].y+a55*k6[i].y);
  yerr[i].x=(tempx*ytemp[i].x+tempy*ytemp[i].y)/(ytemp[i].x*ytemp[i].x+ytemp[i].y*ytemp[i].y);
  yerr[i].y=(-tempx*ytemp[i].y+tempy*ytemp[i].x)/(ytemp[i].x*ytemp[i].x+ytemp[i].y*ytemp[i].y);
}

__global__ void accept (cuDoubleComplex* y, cuDoubleComplex* k1, cuDoubleComplex* k2, cuDoubleComplex* k3, cuDoubleComplex* k4, cuDoubleComplex* k5, cuDoubleComplex* k6, const double a50, const double a51, const double a52, const double a53, const double a54, const double a55) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  y[i].x=(y[i].x+a50*k1[i].x+a51*k2[i].x+a52*k3[i].x+a53*k4[i].x+a54*k5[i].x+a55*k6[i].x);
  y[i].y=(y[i].y+a50*k1[i].y+a51*k2[i].y+a52*k3[i].y+a53*k4[i].y+a54*k5[i].y+a55*k6[i].y);
}

void coupling (cublasHandle_t handle, double t, double t2, int N, cuDoubleComplex* y, cuDoubleComplex* f, cuDoubleComplex* yfft, cuDoubleComplex* frequencies, double *omegas, double c1i, double c1f, double c3i, double c3f, cufftHandle plan, cusparseHandle_t sphandle, cusparseSpMatDescr_t adj, int diff, int eqn, void *externalBuffer, cusparseDnVecDescr_t invec, cusparseDnVecDescr_t outvec){
  if (diff==0){
    cusparseCreateDnVec(&invec, N*N, y, CUDA_C_64F);
    cusparseCreateDnVec(&outvec, N*N, f, CUDA_C_64F);
    cusparseSpMV(sphandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, adj, invec, &beta, outvec, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &externalBuffer);
    cusparseDestroyDnVec(invec);
    cusparseDestroyDnVec(outvec);
  }
  else {
    cufftExecZ2Z(plan, y, yfft, CUFFT_FORWARD);
    cublasZdgmm(handle, CUBLAS_SIDE_LEFT, N*N, 1, frequencies, N*N, yfft, 1, yfft, N*N);
    cufftExecZ2Z(plan, yfft, f, CUFFT_INVERSE);
  }
}


void dydt (cublasHandle_t handle, double t, double t2, int N, cuDoubleComplex* y, cuDoubleComplex* f, cuDoubleComplex* yfft, cuDoubleComplex* frequencies, double *omegas, double c1i, double c1f, double c3i, double c3f, cufftHandle plan, cusparseHandle_t sphandle, cusparseSpMatDescr_t adj, int diff, int eqn, void *externalBuffer, cusparseDnVecDescr_t invec, cusparseDnVecDescr_t outvec){
  double c1=c1f, c3=c3f;
  if (t<t2){
    c1=c1i+t/t2*(c1f-c1i);
    c3=c3i+t/t2*(c3f-c3i);
  }

  coupling(handle, t, t2, N, y, f, yfft, frequencies, omegas, c1i, c1f, c3i, c3f, plan, sphandle, adj, diff, eqn, externalBuffer, invec, outvec);
  if (eqn==0){
    tot_2dcgle<<<(N*N+255)/256, 256>>>(y, f, omegas, c1, c3);
  }
  else{
    tot_stuartlandau<<<(N*N+255)/256, 256>>>(y, f, omegas, c1, c3);
  }
}


void rkf45 (cublasHandle_t handle, double *t, double *h, double t2, int N, double c1i, double c1f, double c3i, double c3f, cuDoubleComplex *y, cuDoubleComplex *ytemp, cuDoubleComplex *frequencies, double *omegas, cuDoubleComplex *k1, cuDoubleComplex *k2, cuDoubleComplex *k3, cuDoubleComplex *k4, cuDoubleComplex *k5, cuDoubleComplex *k6, cuDoubleComplex *yfft, double atl, cuDoubleComplex *yerr, double rtl, cufftHandle plan, cusparseHandle_t sphandle, cusparseSpMatDescr_t adj, int diff, int eqn, void *externalBuffer, cusparseDnVecDescr_t invec, cusparseDnVecDescr_t outvec){
  double norm;
  cuDoubleComplex A10={*h*a1[0],0};

  cublasZcopy(handle, N*N, y, 1, ytemp, 1);
  dydt (handle, (*t)+(*h)*c[0], t2, N, ytemp, k1, yfft, frequencies, omegas, c1i, c1f, c3i, c3f, plan, sphandle, adj, diff, eqn, externalBuffer, invec, outvec);
  cublasZaxpy(handle, N*N, &A10, k1, 1, ytemp, 1);
  dydt (handle, (*t)+(*h)*c[1], t2, N, ytemp, k2, yfft, frequencies, omegas, c1i, c1f, c3i, c3f, plan, sphandle, adj, diff, eqn, externalBuffer, invec, outvec);
  step3<<<(N*N+255)/256, 256>>>(y, k1, k2, ytemp, *h*a2[0], *h*a2[1]);
  dydt (handle, (*t)+(*h)*c[2], t2, N, ytemp, k3, yfft, frequencies, omegas, c1i, c1f, c3i, c3f, plan, sphandle, adj, diff, eqn, externalBuffer, invec, outvec);
  step4<<<(N*N+255)/256, 256>>>(y, k1, k2, k3, ytemp, *h*a3[0], *h*a3[1], *h*a3[2]);
  dydt (handle, (*t)+(*h)*c[3], t2, N, ytemp, k4, yfft, frequencies, omegas, c1i, c1f, c3i, c3f, plan, sphandle, adj, diff, eqn, externalBuffer, invec, outvec);
  step5<<<(N*N+255)/256, 256>>>(y, k1, k2, k3, k4, ytemp, *h*a4[0], *h*a4[1], *h*a4[2], *h*a4[3]);
  dydt (handle, (*t)+(*h)*c[4], t2, N, ytemp, k5, yfft, frequencies, omegas, c1i, c1f, c3i, c3f, plan, sphandle, adj, diff, eqn, externalBuffer, invec, outvec);
  step6<<<(N*N+255)/256, 256>>>(y, k1, k2, k3, k4, k5, ytemp, *h*a5[0], *h*a5[1], *h*a5[2], *h*a5[3], *h*a5[4]);
  dydt (handle, (*t)+(*h)*c[5], t2, N, ytemp, k6, yfft, frequencies, omegas, c1i, c1f, c3i, c3f, plan, sphandle, adj, diff, eqn, externalBuffer, invec, outvec);

  error<<<(N*N+255)/256, 256>>>(y, k1, k2, k3, k4, k5, k6, yerr, ytemp, atl, rtl, *h*(b1[0]-b2[0]), *h*(b1[1]-b2[1]), *h*(b1[2]-b2[2]), *h*(b1[3]-b2[3]), *h*(b1[4]-b2[4]), *h*(b1[5]-b2[5]));

  cublasDznrm2(handle, N*N, yerr, 1, &norm);
  norm/=N;
  double factor=0.9*pow(norm,-0.2);
  if (factor<0.2)
    factor=0.2;
  if (factor>10)
    factor=10;

  if(norm<1){
    cuDoubleComplex B0={(*h)*(b1[0]),0}, B1={(*h)*(b1[1]),0}, B2={(*h)*(b1[2]),0}, B3={(*h)*(b1[3]),0}, B4={(*h)*(b1[4]),0}, B5={(*h)*(b1[5]),0};
    accept<<<(N*N+255)/256, 256>>>(y,  k1, k2, k3, k4, k5, k6, *h*b1[0], *h*b1[1], *h*b1[2], *h*b1[3], *h*b1[4], *h*b1[5]);
    (*t)=(*t)+(*h);
    (*h)*=factor;
  }
  else if (factor<1){
    (*h)*=factor;
  }
}

void getadj(int N, double L, int *rows, int *cols, cuDoubleComplex *vals, int eqn){
  int i1,i2,j1,j2,k1,k2, ind=0;
  for(k1=0; k1<N*N; k1++){
    i1=k1/N;
    j1=k1%N;
    rows[ind]=k1;
    cols[ind]=k1;
    if (eqn==0){
      vals[ind].x=-4.0/(2*PI*L/N*2*PI*L/N);
    }
    else{
      vals[ind].x=0;
    }
    vals[ind++].y=0.0;
    i2=(i1+1)%N;
    j2=j1;
    k2=i2*N+j2;
    rows[ind]=k1;
    cols[ind]=k2;
    vals[ind].x=1.0/(2*PI*L/N*2*PI*L/N);
    vals[ind++].y=0.0;
    i2=(i1-1+N)%N;
    j2=j1;
    k2=i2*N+j2;
    rows[ind]=k1;
    cols[ind]=k2;
    vals[ind].x=1.0/(2*PI*L/N*2*PI*L/N);
    vals[ind++].y=0.0;
    i2=i1;
    j2=(j1+1)%N;
    k2=i2*N+j2;
    rows[ind]=k1;
    cols[ind]=k2;
    vals[ind].x=1.0/(2*PI*L/N*2*PI*L/N);
    vals[ind++].y=0.0;
    i2=i1;
    j2=(j1-1+N)%N;
    k2=i2*N+j2;
    rows[ind]=k1;
    cols[ind]=k2;
    vals[ind].x=1.0/(2*PI*L/N*2*PI*L/N);
    vals[ind++].y=0.0;
  }
}

__device__ double  phase(double phi) {
    if((phi)>PI)
        return phi-2*PI;
    else if ((phi)<-PI)
        return phi+2*PI;
    else
        return phi;
}

__global__ void  findcharges (const int N, cuDoubleComplex *y, int *np, int *nn, int *rowp, int *colp, int *rown, int *coln){
    int i,j,k,l, ind;
    k = blockIdx.x*blockDim.x + threadIdx.x;
    i=k/N;
    j=k%N;

    double theta1=atan2(y[N*i+j].y, y[N*i+j].x);
    double theta2=atan2(y[N*((i+1)%N)+j].y, y[N*((i+1)%N)+j].x);
    double theta3=atan2(y[N*((i+1)%N)+((j+1)%N)].y, y[N*((i+1)%N)+((j+1)%N)].x);
    double theta4=atan2(y[N*i+((j+1)%N)].y, y[N*i+((j+1)%N)].x);
    double d1=phase(theta2-theta1);
    double d2=phase(theta3-theta2);
    double d3=phase(theta4-theta3);
    double d4=phase(theta1-theta4);

    int charge=round((d1+d2+d3+d4)/(1.9*PI));
    if (charge>0){
      for (l=0; l<charge; l++){
        ind=atomicAdd(np, 1);
        rowp[ind]=i;
        colp[ind]=j;
      }
    }
    if (charge<0){
      for (l=0; l<-charge; l++){
        ind=atomicAdd(nn, 1);
        rown[ind]=i;
        coln[ind]=j;
      }
    }
}

__global__ void  average (int countR, cuDoubleComplex *y, cuDoubleComplex *f, double *R){
  int k = blockIdx.x*blockDim.x + threadIdx.x;
  R[k]=R[k]+(sqrt(f[k].x*f[k].x+f[k].y*f[k].y)/sqrt(y[k].x*y[k].x+y[k].y*y[k].y)-R[k])/countR;
}


int main (int argc, char* argv[]) {
    struct timeval start,end;
    double L, c1i, c1f, c3i, c3f, t1, t2, t3, t4, dt;
    double atl, rtl;
    int gpu, N, seed, diff, eqn;
    char* filebase;

    N=1536;
    L=192;
    c1i=2.0;
    c1f=2.0;
    c3i=0.85;
    c3f=0.85;
    t1=1e2;
    t2=0;
    t3=0;
    t4=0;
    dt=1e0;
    gpu=0;
    seed=1;
    diff=1;
    eqn=0;
    int verbose=0;
    rtl=1e-6;
    atl=1e-6;
    char c;
    int help=1;
    int phase=0;

    while (optind < argc) {
      if ((c = getopt(argc, argv, "N:L:B:b:C:c:g:t:T:A:R:d:s:p:r:a:hvDoP")) != -1) {
        switch (c) {
          case 'N':
              N = (int)atoi(optarg);
              break;
          case 'L':
              L = (double)atof(optarg);
              break;
          case 'B':
              c1f = (double)atof(optarg);
              break;
          case 'b':
              c1i = (double)atof(optarg);
              break;
          case 'C':
              c3f = (double)atof(optarg);
              break;
          case 'c':
              c3i = (double)atof(optarg);
              break;
          case 'g':
              gpu = (double)atof(optarg);
              break;
          case 't':
              t1 = (double)atof(optarg);
              break;
          case 'T':
              t2 = (double)atof(optarg);
              break;
          case 'A':
              t3 = (double)atof(optarg);
              break;
          case 'R':
              t4 = (double)atof(optarg);
              break;
          case 'd':
              dt = (double)atof(optarg);
              break;
          case 's':
              seed = (int)atoi(optarg);
              break;
          case 'r':
              rtl = (double)atof(optarg);
              break;
          case 'a':
              atl = (double)atof(optarg);
              break;
          case 'o':
              eqn = 1;
              break;
          case 'D':
              diff = 0;
              break;
          case 'P':
              phase = 1;
              break;
          case 'h':
              help=1;
              break;
          case 'v':
              verbose=1;
              break;
        }
      }
      else {
        filebase=argv[optind];
        optind++;
        help=0;
      }
    }
    if (help) {
      printf("usage:\t2dcgle [-h] [-v] [-N N] [-L L] [-b c1i] [-B c1f]\n");
      printf("\t[-c c3i] [-C c3f] [-t t1] [-T t2] [-A t3] [-R t4] [-d dt] [-s seed] \n");
      printf("\t[-r rtol] [-a atol] [-g gpu] filebase \n\n");
      printf("-h for help \n");
      printf("-v for verbose \n");
      printf("-D for finite differences \n");
      printf("-P to supress dense phase output \n");
      printf("-o for stuart landau oscillators \n");
      printf("N is number of oscillators. Default 1536. \n");
      printf("2Pi L is linear system size. Default 192. \n");
      printf("c1i is initial CGLE Laplacian coefficient. Default 2.0. \n");
      printf("c1f is final CGLE Laplacian coefficient. Default 2.0. \n");
      printf("c3i is initial CGLE cubic coefficient. Default 0.85. \n");
      printf("c3f is final CGLE cubic coefficient. Default 0.85. \n");
      printf("t1 is total integration time. Default 1e2. \n");
      printf("t2 is time to stop quasistatic change. Default 0. \n");
      printf("t3 is time stop outputting dense timestep data. Default 0. \n");
      printf("t4 is time start averagine order parameter. Default 0. \n");
      printf("dt is the time between outputs. Default 1e0. \n");
      printf("seed is random seed. Default 1. \n");
      printf("diff is 0 for finite diff, 1 for pseudospectral. Default 1.\n");
      printf("rtol is relative error tolerance. Default 1e-6.\n");
      printf("atol is absolute error tolerance. Default 1e-6.\n");
      printf("gpu is index of the gpu. Default 0.\n");
      printf("filebase is base file name for output. \n");


      exit(0);
    }

    double t=0,h,t0;
    int i,j,steps=0;
    FILE *outlast, *outanimation, *outcharges, *outtimes, *out, *in;
    char file[256];
    strcpy(file,filebase);
    strcat(file, "phases.dat");
    outanimation = fopen(file,"w");
    strcpy(file,filebase);
    strcat(file, "charges.dat");
    outcharges = fopen(file,"w");
    strcpy(file,filebase);
    strcat(file,".out");
    out = fopen(file,"w");
    strcpy(file,filebase);
    strcat(file, "times.dat");
    outtimes = fopen(file,"w");
    double *omegasloc, *omegas, *Rloc, *R;
    cuDoubleComplex *yloc, *frequenciesloc;
    cuDoubleComplex *y, *f, *ytemp, *yfft, *yerr, *frequencies, *k1, *k2, *k3, *k4, *k5, *k6;

    for (int  i=0; i<argc; i++){
      fprintf(out, "%s ", argv[i]);
      if(verbose){
      	printf("%s ", argv[i]);
      }
    }
    fprintf(out,"\n");
    if(verbose){
        printf("\n");
    }

    cublasStatus_t stat;
    cusparseStatus_t stat2;
    cublasHandle_t handle;
    cusparseHandle_t sphandle;

    cudaSetDevice(gpu);
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    stat2 = cusparseCreate(&sphandle);
    if (stat2 != CUSPARSE_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    int *rowsloc, *colsloc, *rows, *cols;
    cuDoubleComplex *valsloc, *vals;
    cusparseSpMatDescr_t adj;
    yloc = (cuDoubleComplex*)calloc(N*N,sizeof(cuDoubleComplex));
    frequenciesloc = (cuDoubleComplex*)calloc(N*N,sizeof(cuDoubleComplex));
    omegasloc = (double*)calloc(N*N,sizeof(double));
    Rloc = (double*)calloc(N*N,sizeof(double));

    size_t fr, total;
    cudaMemGetInfo (&fr, &total);
    printf("GPU Memory: %li %li\n", fr, total);
    if(fr < 1000*N*N*sizeof(double)) {
      printf("GPU Memory low! \n");
      return 0;
    }
    cudaMalloc ((void**)&omegas, N*N*sizeof(double));
    cudaMalloc ((void**)&R, N*N*sizeof(double));
    cudaMalloc ((void**)&y, N*N*sizeof(cuDoubleComplex));
    cudaMalloc ((void**)&f, N*N*sizeof(cuDoubleComplex));
    cudaMalloc ((void**)&yerr, N*N*sizeof(cuDoubleComplex));
    cudaMalloc ((void**)&ytemp, N*N*sizeof(cuDoubleComplex));
    cudaMalloc ((void**)&yfft, N*N*sizeof(cuDoubleComplex));
    cudaMalloc ((void**)&frequencies, N*N*sizeof(cuDoubleComplex));
    cudaMalloc ((void**)&k1, N*N*sizeof(cuDoubleComplex));
    cudaMalloc ((void**)&k2, N*N*sizeof(cuDoubleComplex));
    cudaMalloc ((void**)&k3, N*N*sizeof(cuDoubleComplex));
    cudaMalloc ((void**)&k4, N*N*sizeof(cuDoubleComplex));
    cudaMalloc ((void**)&k5, N*N*sizeof(cuDoubleComplex));
    cudaMalloc ((void**)&k6, N*N*sizeof(cuDoubleComplex));


    for (i =0; i<N; i++){
      for (j =0; j<N; j++){
        frequenciesloc[N*i+j].x = -((N/2-abs(j-N/2))*(N/2-abs(j-N/2))+(N/2-abs(i-N/2))*(N/2-abs(i-N/2)))/(N*L*N*L);
        frequenciesloc[N*i+j].y = 0;
      }
    }
    cublasSetVector (N*N, sizeof(cuDoubleComplex), frequenciesloc, 1, frequencies, 1);
    cublasSetVector (N*N, sizeof(double), Rloc, 1, R, 1);

    cufftHandle plan;
    cufftPlan2d (&plan, N, N, CUFFT_Z2Z);

    fprintf(out,"%i %i %i %f %f %f %f %f %f %f %f %f %e %e %s\n", N, seed, diff, L, c1i, c1f, c3i, c3f, t1, t2, t3, dt, atl, rtl, filebase);
    fflush(out);

    int nnz=5*N*N;

    strcpy(file,filebase);
    strcat(file, "adj.dat");
    if ((in = fopen(file,"r")))
    {
        printf("Using adjacency from file\n");
        fprintf(out, "Using adjacency from file\n");
        size_t read=fread(&nnz,sizeof(int),1,in);
        rowsloc=(int*)calloc(nnz,sizeof(int));
        colsloc=(int*)calloc(nnz,sizeof(int));
        valsloc=(cuDoubleComplex*)calloc(nnz,sizeof(cuDoubleComplex));
        double *valstemp=(double*)calloc(nnz,sizeof(double));

        read=fread(rowsloc,sizeof(int),nnz,in);
        read=fread(colsloc,sizeof(int),nnz,in);
        read=fread(valstemp,sizeof(double),nnz,in);
        for (int i=0; i<nnz; i++){
          valsloc[i].x=valstemp[i]/(2*PI*L/N*2*PI*L/N);
          valsloc[i].y=0;
        }
        fclose(in);
        free(valstemp);
    }
    else {
        printf("Using nearest neighbors adjacency\n");
        fprintf(out, "Using nearest neighbors adjacency\n");
        rowsloc=(int*)calloc(nnz,sizeof(int));
        colsloc=(int*)calloc(nnz,sizeof(int));
        valsloc=(cuDoubleComplex*)calloc(nnz,sizeof(cuDoubleComplex));
        getadj(N, L, rowsloc, colsloc, valsloc, eqn);
    }
    printf("%i %i %i %f %f %i %i %f %f\n",nnz, rowsloc[0], colsloc[0], valsloc[0].x*(2*PI*L/N*2*PI*L/N), valsloc[0].y, rowsloc[nnz-1], colsloc[nnz-1], valsloc[nnz-1].x*(2*PI*L/N*2*PI*L/N), valsloc[nnz-1].y);
    cudaMalloc((void**)&rows,sizeof(int)*nnz);
    cudaMalloc((void**)&cols,sizeof(int)*nnz);
    cudaMalloc((void**)&vals,sizeof(cuDoubleComplex)*nnz);
    cublasSetVector (nnz, sizeof(int), rowsloc, 1, rows, 1);
    cublasSetVector (nnz, sizeof(int), colsloc, 1, cols, 1);
    cublasSetVector (nnz, sizeof(cuDoubleComplex), valsloc, 1, vals, 1);

    cusparseCreateCoo(&adj, N*N, N*N, nnz, rows, cols, vals, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);

    strcpy(file,filebase);
    strcat(file, "ic.dat");
    if ((in = fopen(file,"r")))
    {
        printf("Using initial conditions from file\n");
        fprintf(out, "Using initial conditions from file\n");
        size_t read=fread(yloc,sizeof(double),2*N*N,in);
        fclose(in);
        if (read!=2*N*N){
          printf("initial conditions file not compatible with N!");
          return 0;
        }
    }
    else {
        printf("Using random initial conditions\n");
        fprintf(out, "Using random initial conditions\n");
        srand(seed);
        for(j=0; j<N*N; j++) {
          yloc[j].x = (2.0/RAND_MAX*rand()-1);
          yloc[j].y = (2.0/RAND_MAX*rand()-1);
        }
    }

    strcpy(file,filebase);
    strcat(file, "frequencies.dat");
    if ((in = fopen(file,"r")))
    {
        printf("Using frequencies from file\n");
        fprintf(out, "Using frequencies from file\n");
        size_t read=fread(omegasloc,sizeof(double),N*N,in);
        fclose(in);
        if (read!=N*N){
          printf("frequencies not compatible with N!");
          return 0;
        }
    }
    else {
        printf("Using constant frequencies\n");
        fprintf(out, "Using constant frequencies\n");
        for(j=0; j<N*N; j++) {
          omegasloc[j] = 1.0;
        }
    }
    cublasSetVector (N*N, sizeof(double), omegasloc, 1, omegas, 1);

    gettimeofday(&start,NULL);
    h = dt/100;


    cublasSetVector (N*N, sizeof(cuDoubleComplex), yloc, 1, y, 1);
    int *nn, *np, *colp, *rowp, *coln, *rown;
    int *nnloc, *nploc, *colploc, *rowploc, *colnloc, *rownloc;
    nnloc=(int*)calloc(1,sizeof(int));
    nploc=(int*)calloc(1,sizeof(int));
    colploc=(int*)calloc(N*N,sizeof(int));
    rowploc=(int*)calloc(N*N,sizeof(int));
    colnloc=(int*)calloc(N*N,sizeof(int));
    rownloc=(int*)calloc(N*N,sizeof(int));
    cudaMalloc ((void**)&nn, 1*sizeof(int));
    cudaMalloc ((void**)&np, 1*sizeof(int));
    cudaMalloc ((void**)&colp, N*N*sizeof(int));
    cudaMalloc ((void**)&rowp, N*N*sizeof(int));
    cudaMalloc ((void**)&coln, N*N*sizeof(int));
    cudaMalloc ((void**)&rown, N*N*sizeof(int));

    size_t size=512*N*N*sizeof(double);
    void *externalBuffer;
    cusparseDnVecDescr_t invec, outvec;
    cusparseCreateDnVec(&invec, N*N, y, CUDA_C_64F);
    cusparseCreateDnVec(&outvec, N*N, f, CUDA_C_64F);
    cusparseSpMV_bufferSize(sphandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, adj, invec, &beta, outvec, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &size);
    cudaMalloc((void **)&externalBuffer, size*sizeof(char));

    int Rcount=1;

    while(t<t1+dt){
      t0=t;

      if(t>=t4){
        coupling(handle, t, t2, N, y, f, yfft, frequencies, omegas, c1i, c1f, c3i, c3f, plan, sphandle, adj, diff, eqn, externalBuffer, invec, outvec);
        average<<<(N*N+255)/256,256>>>(Rcount, y, f, R);
        Rcount++;
      }

      if(t>=t3){
        cublasGetVector(N*N, sizeof(cuDoubleComplex), y, 1, yloc, 1);
        if(phase==0){
          fwrite(yloc,sizeof(double),2*N*N,outanimation);
          fflush(outanimation);
        }
        int zero[1]={0};
        cublasSetVector (1, sizeof(int), zero, 1, np, 1);
        cublasSetVector (1, sizeof(int), zero, 1, nn, 1);
        findcharges<<<(N*N+255)/256,256>>>(N, y, np, nn, rowp, colp, rown, coln);
        cublasGetVector (1, sizeof(int), np, 1, nploc, 1);
        cublasGetVector (1, sizeof(int), nn, 1, nnloc, 1);
        cublasGetVector (*nploc, sizeof(int), colp, 1, colploc, 1);
        cublasGetVector (*nploc, sizeof(int), rowp, 1, rowploc, 1);
        cublasGetVector (*nnloc, sizeof(int), coln, 1, colnloc, 1);
        cublasGetVector (*nnloc, sizeof(int), rown, 1, rownloc, 1);
        fwrite(nploc,sizeof(int),1,outcharges);
        fwrite(rowploc,sizeof(int),*nploc,outcharges);
        fwrite(colploc,sizeof(int),*nploc,outcharges);
        fwrite(nnloc,sizeof(int),1,outcharges);
        fwrite(rownloc,sizeof(int),*nnloc,outcharges);
        fwrite(colnloc,sizeof(int),*nnloc,outcharges);
        fflush(outcharges);

        strcpy(file,filebase);
        strcat(file,"fs.dat");
        outlast=fopen(file,"w");
        fwrite(yloc,sizeof(double),2*N*N,outlast);
        fflush(outlast);
        fclose(outlast);
      }


      while(t<t0+dt){
        steps++;
        if(verbose) {
          gettimeofday(&end,NULL);
          printf("%.3f\t%1.3e\t%1.3e\t%f\t%i\n",t/t1, end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec), (end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec))/((t+h)/t1)*(1-t/t1), h, steps);
          fflush(stdout);
        }
        fwrite(&t,sizeof(double),1,outtimes);
        fflush(outtimes);

        if(t+h>t0+dt)
          h=t0+dt-t;

        rkf45 (handle, &t, &h, t2, N, c1i, c1f, c3i, c3f, y, ytemp, frequencies, omegas, k1, k2, k3, k4, k5, k6, yfft, atl, yerr, rtl, plan, sphandle, adj, diff, eqn, externalBuffer, invec, outvec);
      }
    }

    cublasGetVector (N*N, sizeof(cuDoubleComplex), y, 1, yloc, 1);
    strcpy(file,filebase);
    strcat(file,"fs.dat");
    outlast=fopen(file,"w");
    fwrite(yloc,sizeof(double),2*N*N,outlast);
    fflush(outlast);
    fclose(outlast);

    int zero[1]={0};
    cublasSetVector (1, sizeof(int), zero, 1, np, 1);
    cublasSetVector (1, sizeof(int), zero, 1, nn, 1);
    findcharges<<<(N*N+255)/256,256>>>(N, y, np, nn, rowp, colp, rown, coln);
    cublasGetVector (1, sizeof(int), np, 1, nploc, 1);
    cublasGetVector (1, sizeof(int), nn, 1, nnloc, 1);
    cublasGetVector (*nploc, sizeof(int), colp, 1, colploc, 1);
    cublasGetVector (*nploc, sizeof(int), rowp, 1, rowploc, 1);
    cublasGetVector (*nnloc, sizeof(int), coln, 1, colnloc, 1);
    cublasGetVector (*nnloc, sizeof(int), rown, 1, rownloc, 1);
    strcpy(file,filebase);
    strcat(file,"fc.dat");
    outlast=fopen(file,"w");
    fwrite(nploc,sizeof(int),1,outlast);
    fwrite(rowploc,sizeof(int),*nploc,outlast);
    fwrite(colploc,sizeof(int),*nploc,outlast);
    fwrite(nnloc,sizeof(int),1,outlast);
    fwrite(rownloc,sizeof(int),*nnloc,outlast);
    fwrite(colnloc,sizeof(int),*nnloc,outlast);
    fflush(outlast);
    fclose(outlast);

    cublasGetVector (N*N, sizeof(double), R, 1, Rloc, 1);
    strcpy(file,filebase);
    strcat(file,"order.dat");
    outlast=fopen(file,"w");
    fwrite(Rloc,sizeof(double),N*N,outlast);
    fflush(outlast);
    fclose(outlast);

    gettimeofday(&end,NULL);
    printf("\nruntime: %f\n",end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec));
    fprintf(out,"\nsteps: %i\n",steps);
    fprintf(out,"runtime: %f\n",end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec));

    fclose(outanimation);
    fclose(outcharges);
    fclose(outtimes);
    fclose(out);

    free(yloc);
    free(rowsloc);
    free(colsloc);
    free(valsloc);
    free(omegasloc);
    free(nploc);
    free(nnloc);
    free(colploc);
    free(colnloc);
    free(rowploc);
    free(rownloc);
    cudaFree(omegas);
    cudaFree(y);
    cudaFree(yerr);
    cudaFree(ytemp);
    cudaFree(yfft);
    cudaFree(frequencies);
    cudaFree(k1);
    cudaFree(k2);
    cudaFree(k3);
    cudaFree(k4);
    cudaFree(k5);
    cudaFree(k6);
    cudaFree(rows);
    cudaFree(cols);
    cudaFree(vals);
    cudaFree (nn);
    cudaFree(np);
    cudaFree(colp);
    cudaFree(coln);
    cudaFree(rowp);
    cudaFree(rown);
    cudaFree(externalBuffer);

    cufftDestroy(plan);
    cublasDestroy(handle);
    cusparseDestroy(sphandle);


    return 0;
}
