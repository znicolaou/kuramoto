//Zachary G. Nicolaou 2/10/2024
//Integrate the Kuramoto model with adaptive Runke Kutta timestepping on a gpu
//Default adjacency, initial conditions, and frequencies follow volcano
//nvcc -lcuda -lcublas -lcurand -O3 -o kuramoto dp45.cu kuramoto.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include "dp45.h"
#include "cublas_v2.h"
#include <curand_kernel.h>

typedef struct parameters
{
  cublasHandle_t handle;
  unsigned long int N;
  unsigned long int K;
  int A;
  float *y2;
  float *f2;
  float *f3;
  float *omegas;
  float *adj;
  float c1;
  float t0;
  float t1;
  int steps;
  int verbose;
  int dense;
  float *t_eval;
  int n_eval;
  int eval_i;
  float *yloc;
  float *ones;
  char *filebase;
  struct timeval start;
  curandStatePhilox4_32_10_t *state;
}parameters;

__global__ void tot_kuramoto (float* y, const unsigned long int N, float* y2, float* f, float* f2, float *omegas, const float c1) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<N){
    f[i]=omegas[i]+c1*(y2[2*i+1]*f2[2*i]-y2[2*i]*f2[2*i+1]);
  }
}

__global__ void makey2 (float* y, const unsigned long int N, float* y2, float *omegas, const float t) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<N){
    y2[2*i]=cos(y[i]);
    y2[2*i+1]=sin(y[i]);
  }
}

__global__ void makestate(curandStatePhilox4_32_10_t *state, int seed)
{
  curand_init(seed, 0, 0, state);
}

__global__ void m1(float* y, float* f, int N, int K, curandStatePhilox4_32_10_t *globalstate) {
  int m = blockIdx.x*blockDim.x+threadIdx.x;

  if (m < K) {
    float tmpSum = 0;
    float tmpSum2 = 0;
    curandStatePhilox4_32_10_t state = *globalstate;
    skipahead(m*N, &state);

    float s=pow(-1.0,m);

    for (int n = 0; n < N; n++) {
      float umn=(2*(curand(&state)%2)-1.0);
      tmpSum += y[2*n] * s*umn;
      tmpSum2 += y[2*n+1] * s*umn;
    }
    f[2*m] = tmpSum/N;
    f[2*m+1] = tmpSum2/N;
  }
}

__global__ void m2(float* y, float* f, int N, int K, curandStatePhilox4_32_10_t *globalstate) {
  int n = blockIdx.x*blockDim.x+threadIdx.x;

  if (n < N) {
    float tmpSum = 0;
    float tmpSum2 = 0;
    curandStatePhilox4_32_10_t state = *globalstate;
    skipahead(n, &state);

    for (int m = 0; m < K; m++) {
      float umn=(2*(curand(&state)%2)-1.0);
      skipahead(N-1, &state);
      tmpSum += y[2*m] * umn;
      tmpSum2 += y[2*m+1] * umn;
    }
    f[2*n] = tmpSum;
    f[2*n+1] = tmpSum2;
  }
}


void makecoupling(float t, float *y, float *f, void* pars){
  parameters *p = (parameters *)pars;
  if(p->A){
    m1<<<(p->K+255)/256, 256>>> (y,p->f3, p->N, p->K, p->state);
    m2<<<(p->N+255)/256, 256>>> (p->f3,f, p->N, p->K, p->state);
  }
  else{
    float alpha=1;
    float beta=0;
    cublasSgemv(p->handle, CUBLAS_OP_T, p->N, p->N, &alpha, p->adj, p->N, p->y2, 2, &beta, p->f2, 2);
    cublasSgemv(p->handle, CUBLAS_OP_T, p->N, p->N, &alpha, p->adj, p->N, (p->y2)+1, 2, &beta, (p->f2)+1, 2);
  }
}

void dydt (float t, float *y, float *f, void *pars){

  parameters *p = (parameters *)pars;

  makey2<<<(p->N+255)/256, 256>>>(y, p->N, p->y2, p->omegas, t);
  makecoupling(t,p->y2,p->f2,pars);
  tot_kuramoto<<<(p->N+255)/256, 256>>>(y, p->N, p->y2, f, p->f2, p->omegas, p->c1);

}

__global__ void makeu1 (float* u1, unsigned int *r, const unsigned long int N, const int K, curandStatePhilox4_32_10_t *globalstate) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<N*K){
    curandStatePhilox4_32_10_t state=*globalstate;
    skipahead(i, &state);
    u1[i] = (2*(curand(&state)%2)-1.0);
  }
}

__global__ void makeu2 (float* u1, float *u2, const unsigned long int N, const int K) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<N*K){
    int m=i/N;
    u2[i]=pow(-1,m)*u1[i]/N;
  }
}

void getadj(void *pars){
  parameters *p = (parameters *)pars;

  float *u1, *u2;
  const float alpha=1.0;
  const float beta=0.0;

  cudaMalloc ((void**)&u1, p->K*p->N*sizeof(float));
  cudaMalloc ((void**)&u2, p->K*p->N*sizeof(float));

  makeu1<<<(p->N*p->K+255)/256, 256>>>(u1, (unsigned int *)u2, p->N, p->K, p->state);
  makeu2<<<(p->N*p->K+255)/256, 256>>>(u1, u2, p->N, p->K);
  cublasSgemm(p->handle,CUBLAS_OP_N,CUBLAS_OP_T,p->N,p->N,p->K,&alpha,u2,p->N,u1,p->N,&beta,p->adj,p->N);

  cudaFree(u1);
  cudaFree(u2);
}


void step_eval(float t, float h, float* y, void *pars){
  parameters *p = (parameters *)pars;
  static char file[256];
  struct timeval end;

  p->steps++;
  if(p->verbose) {
    gettimeofday(&end,NULL);
    printf("%.3f\t%1.3e\t%1.3e\t%f\t%i\t\r",(t-p->t0)/(p->t1-p->t0), end.tv_sec-p->start.tv_sec + 1e-6*(end.tv_usec-p->start.tv_usec), (end.tv_sec-p->start.tv_sec + 1e-6*(end.tv_usec-p->start.tv_usec))/((t-p->t0+h)/(p->t1-p->t0))*(1-(t-p->t0)/(p->t1-p->t0)), h, p->steps);
    fflush(stdout);
    strcpy(file,p->filebase);
    strcat(file,".out");
    FILE *out = fopen(file,"ab");
    fprintf(out,"%.3f\t%1.3e\t%1.3e\t%f\t%i\t\n",(t-p->t0)/(p->t1-p->t0), end.tv_sec-p->start.tv_sec + 1e-6*(end.tv_usec-p->start.tv_usec), (end.tv_sec-p->start.tv_sec + 1e-6*(end.tv_usec-p->start.tv_usec))/((t-p->t0+h)/(p->t1-p->t0))*(1-(t-p->t0)/(p->t1-p->t0)), h, p->steps);
    fflush(out);
    fclose(out);
  }
  if(p->dense){
    strcpy(file,p->filebase);
    strcat(file, "times.dat");
    FILE *outtimes = fopen(file,"ab");
    fwrite(&t,sizeof(float),1,outtimes);
    fflush(outtimes);
    fclose(outtimes);
  }

  FILE *outanimation;
  if(p->dense){
    strcpy(file,p->filebase);
    strcat(file, "thetas.dat");
    outanimation = fopen(file,"ab");
  }


  float X,Y;
  static float *r=(float*)calloc(p->n_eval,sizeof(float));
  int eval_j=p->eval_i;
  while (t >= p->t_eval[eval_j] && eval_j<p->n_eval){
    eval_j++;
  }
  int num=eval_j-p->eval_i;
  int ind=0;

  while (t >= p->t_eval[p->eval_i] && p->eval_i<p->n_eval){
    float *y_eval;
    y_eval=dp45_eval(t,p->t_eval[p->eval_i]);

    makey2<<<(p->N+255)/256, 256>>>(y_eval, p->N, p->y2, p->omegas, t);
    cublasSdot(p->handle,p->N, p->y2, 2, p->ones, 1, &X);
    cublasSdot(p->handle,p->N, p->y2+1, 2, p->ones, 1, &Y);
    r[ind++]=pow((X/p->N*X/p->N+Y/p->N*Y/p->N),0.5);
    if(p->dense){
      cublasGetVector(p->N, sizeof(float), y_eval, 1, p->yloc, 1);
      fwrite(p->yloc,sizeof(float),p->N,outanimation);
      fflush(outanimation);
    }

    p->eval_i++;
  }
  if(p->dense){
    fclose(outanimation);
  }
  cublasGetVector(p->N, sizeof(float), y, 1, p->yloc, 1);

  if (num>0){
    strcpy(file,p->filebase);
    strcat(file,"order.dat");
    FILE *outorder=fopen(file,"ab");
    fwrite(r,sizeof(float),num,outorder);
    fflush(outorder);
    fclose(outorder);
  }

  strcpy(file,p->filebase);
  strcat(file,"fs.dat");
  FILE *outlast=fopen(file,"wb");

  fwrite(p->yloc,sizeof(float),p->N,outlast);
  fwrite(&t,sizeof(float),1,outlast);
  fwrite(&h,sizeof(float),1,outlast);
  fflush(outlast);
  fclose(outlast);
}

int main (int argc, char* argv[]) {
    struct timeval start,end;
    gettimeofday(&start,NULL);

    float c1, t1, dt;
    float freq, atl, rtl, I;
    int gpu, seed, fixed, A;
    unsigned long int N,K;
    char* filebase;

    N=5000;
    K=6;
    A=0;
    I=0;
    c1=3.0;
    freq=1.0;

    t1=1e2;
    dt=1e0;
    gpu=0;
    seed=1;
    int verbose=0;
    rtl=0;
    atl=1e-6;
    fixed=0;
    char c;
    int help=1;
    int dense=1;
    int normal=0;
    int reload=0;

    while (optind < argc) {
      if ((c = getopt(argc, argv, "N:K:I:c:g:t:d:f:s:r:a:hvDFnRA")) != -1) {
        switch (c) {
          case 'N':
              N = (int)atoi(optarg);
              break;
          case 'K':
              K = (int)atoi(optarg);
              break;
          case 'A':
              A = 1;
              break;
          case 'I':
              I = (float)atof(optarg);
              break;
          case 'c':
              c1 = (float)atof(optarg);
              break;
          case 'g':
              gpu = (float)atof(optarg);
              break;
          case 't':
              t1 = (float)atof(optarg);
              break;
          case 'd':
              dt = (float)atof(optarg);
              break;
          case 'f':
              freq = (float)atof(optarg);
              break;
          case 's':
              seed = (int)atoi(optarg);
              break;
          case 'r':
              rtl = (float)atof(optarg);
              break;
          case 'a':
              atl = (float)atof(optarg);
              break;
          case 'D':
              dense = 0;
              break;
          case 'R':
              reload = 1;
              break;
          case 'F':
              fixed = 1;
              break;
          case 'h':
              help=1;
              break;
          case 'v':
              verbose=1;
              break;
          case 'n':
              normal=1;
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
      printf("usage:\tkuramoto [-hvnDRFA] [-N N] [-K K]\n");
      printf("\t[-c c] [-t t] [-d dt] [-f f] [-s seed] \n");
      printf("\t[-I init] [-r rtol] [-a atol] [-g gpu] filebase \n\n");
      printf("-h for help \n");
      printf("-v for verbose \n");
      printf("-n for normal random frequencies (default is cauchy) \n");
      printf("-D to supress dense output \n");
      printf("-R to reload adjacency, frequencies, and initial conditions from files if possible\n");
      printf("-F for fixed timestep \n");
      printf("-A to regenerate volcano adjacency each step to save memory \n");
      printf("N is number of oscillators. Default 5000. \n");
      printf("K is rank of volcano adjacency. Default 5. \n");
      printf("c is the coupling coefficient. Default 3.0. \n");
      printf("t is total integration time. Default 1e2. \n");
      printf("dt is the time between outputs. Default 1e0. \n");
      printf("f is the scale of the frequencies. Default 1e0. \n");
      printf("seed is random seed. Default 1. \n");
      printf("init is uniform random initial condition scale. Default 0. \n");
      printf("rtol is relative error tolerance. Default 0.\n");
      printf("atol is absolute error tolerance. Default 1e-6.\n");
      printf("gpu is index of the gpu. Default 0.\n");
      printf("filebase is base file name for output. \n");


      exit(0);
    }

    float t=0,h;
    int j=0;
    FILE *out, *in;

    char file[256];
    strcpy(file,filebase);
    strcat(file,".out");
    out = fopen(file,"ab");

    float *omegasloc, *omegas, *yloc, *adjloc, *adj, *y2, *f2, *f3, *ones;
    yloc = (float*)calloc(N,sizeof(float));
    omegasloc = (float*)calloc(N,sizeof(float));
    adjloc = (float*)calloc(N*N,sizeof(float));

    cublasStatus_t stat;
    cublasHandle_t handle;

    cudaSetDevice(gpu);
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        fprintf (out,"CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    srand(seed);

    fprintf(out,"%lu %lu %f %f %f %i\n", N, K, t1, dt, c1, seed);
    for (int  i=0; i<argc; i++){
      fprintf(out, "%s ", argv[i]);
    }
    fprintf(out, "\n");

    size_t fr, total, req;
    cudaMemGetInfo (&fr, &total);
    if(A){
      req=100*N*sizeof(float);
    }
    else{
      req=(100+N+2*K)*N*sizeof(float);
    }
    printf("GPU Memory: %lu %lu %lu\n", fr, total, req);
    fprintf(out,"GPU Memory: %lu %lu %lu\n", fr, total, req);
    if(fr < req) {
      printf("GPU Memory low!\n");
      fprintf(out,"GPU Memory low!\n");
      return 0;
    }
    fflush(out);


    cudaMalloc ((void**)&omegas, N*sizeof(float));
    cudaMalloc ((void**)&y2, 2*N*sizeof(float));
    cudaMalloc ((void**)&f2, 2*N*sizeof(float));
    cudaMalloc ((void**)&f3, 2*K*sizeof(float));
    if(!A){
      cudaMalloc((void**)&adj,sizeof(float)*N*N);
    }
    cudaMalloc((void**)&ones,sizeof(float)*N);
    for(int i=0; i<N; i++){
      yloc[i]=1;
    }
    cublasSetVector(N, sizeof(float), yloc, 1, ones, 1);

    curandStatePhilox4_32_10_t *state;
    cudaMalloc((void**)&state,sizeof(curandStatePhilox4_32_10_t));
    makestate<<<1, 1>>>(state,seed);

    if (fixed){
      h = dt;
    }
    else{
      h = dt/100;
    }

    strcpy(file,filebase);
    strcat(file, "fs.dat");
    int reloaded=0;
    if (reload && (in = fopen(file,"r"))){
      reloaded=1;
      printf("Using initial conditions from file\n");
      fprintf(out, "Using initial conditions from file\n");
      size_t read=fread(yloc,sizeof(float),N,in);
      if (read!=N){
        printf("initial conditions file not compatible with N!\n");
        fprintf(out,"initial conditions file not compatible with N!\n");
        reloaded=0;
      }
      if(reloaded){
        read=fread(&t,sizeof(float),1,in);
        read=fread(&h,sizeof(float),1,in);
        if (read!=1){
          printf("Couldn't read start time and step!\n");
          fprintf(out,"Couldn't read start time and step!\n");
          reloaded=0;
        }
      }
      fclose(in);

      printf("Restarting at t=%f with h=%f\n",t,h);
      fprintf(out,"Restarting at t=%f with h=%f\n",t,h);
    }
    if (!reloaded) {
      printf("Using random initial conditions\n");
      fprintf(out, "Using random initial conditions\n");
      for(j=0; j<N; j++) {
        yloc[j] = I/2*(2.0/RAND_MAX*rand()-1);
      }
      in=fopen(file,"wb");
      fwrite(yloc,sizeof(float),N,in);
      fclose(in);

      cublasSetVector(N, sizeof(float), yloc, 1, omegas, 1);
      strcpy(file,filebase);
      strcat(file,"order.dat");
      FILE *outorder=fopen(file,"wb");
      makey2<<<(N+255)/256, 256>>>(omegas, N, y2, omegas, t);
      float X,Y;
      cublasSdot(handle,N, y2, 2, ones, 1, &X);
      cublasSdot(handle,N, y2+1, 2, ones, 1, &Y);
      float r=pow((X/N*X/N+Y/N*Y/N),0.5);
      fwrite(&r,sizeof(float),1,outorder);
      fflush(outorder);
      fclose(outorder);

      if(dense){
        strcpy(file,filebase);
        strcat(file,"thetas.dat");
        FILE *outanimation=fopen(file,"wb");
        fwrite(yloc,sizeof(float),N,outanimation);
        fflush(outanimation);
        fclose(outanimation);
        strcpy(file,filebase);
        strcat(file,"times.dat");
        FILE *outtimes=fopen(file,"wb");
        fclose(outtimes);
      }
    }

    strcpy(file,filebase);
    strcat(file, "frequencies.dat");
    if (reload && (in = fopen(file,"r")))
    {
        printf("Using frequencies from file\n");
        fprintf(out, "Using frequencies from file\n");
        size_t read=fread(omegasloc,sizeof(float),N,in);
        fclose(in);
        if (read!=N){
          printf("frequencies not compatible with N!");
          fprintf(out,"frequencies not compatible with N!");
          return 0;
        }
    }
    else {

        printf("Using random frequencies\n");
        fprintf(out, "Using random frequencies\n");
        for(j=0; j<N; j++) {
          if (normal){
            float u = rand() / (float)RAND_MAX;
            float v = rand() / (float)RAND_MAX;
            omegasloc[j] = freq*pow(-2*log(u),0.5)*cos(2*M_PI*v);

          }
          else{
            float u = rand() / (float)RAND_MAX;
            omegasloc[j] = freq*tan(M_PI * (u - 0.5));
          }
        }
        in=fopen(file,"wb");
        fwrite(omegasloc,sizeof(float),N,in);
        fclose(in);
    }

    int n0=int(t/dt)+1;
    int n1=int(t1/dt)+1;
    int n_eval=n1-n0;
    float *t_eval=(float *)calloc(n_eval,sizeof(float));
    int ind=0;
    for(int n=n0; n<n1; n++){
      t_eval[ind++]=dt*n;
    }

    cublasSetVector (N, sizeof(float), omegasloc, 1, omegas, 1);


    parameters pars={.handle=handle, .N=N, .K=K, .A=A, .y2=y2, .f2=f2, .f3=f3, .omegas=omegas, .adj=adj, .c1=c1, .t0=t, .t1=t1, .steps=0, .verbose=verbose, .dense=dense, .t_eval=t_eval, .n_eval=n_eval, .eval_i=0, .yloc=yloc, .ones=ones, .filebase=filebase,.start=start, .state=state};

    strcpy(file,filebase);
    strcat(file, "adj.dat");
    if (!A && reload && (in = fopen(file,"r")))
    {
        printf("Using adjacency from file\n");
        fprintf(out, "Using adjacency from file\n");
        size_t read=fread(adjloc,sizeof(float),N*N,in);
        fclose(in);
        cublasSetVector(N*N, sizeof(float), adjloc, 1, adj, 1);
    }
    else {
        printf("Using random adjacency matrix\n");
        fprintf(out, "Using random adjacency matrix\n");
        if(!A){
          getadj(&pars);
          if (dense){
            cublasGetVector(N*N, sizeof(float), adj, 1, adjloc, 1);
            in=fopen(file,"wb");
            fwrite(adjloc,sizeof(float),N*N,in);
            fclose(in);
          }
        }
    }
    fflush(out);
    fclose(out);


    float* y=dp45_init(N, atl, rtl, fixed, yloc, handle, &dydt);
    float *y_eval=dp45_run(&t, &h, t1, &pars, &step_eval);

    strcpy(file,filebase);
    strcat(file,".out");
    out = fopen(file,"ab");
    gettimeofday(&end,NULL);
    printf("\nruntime: %f\n",end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec));
    fprintf(out,"\nsteps: %i\n",pars.steps);
    fprintf(out,"runtime: %f\n",end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec));
    fflush(out);
    fclose(out);

    free(yloc);
    free(omegasloc);
    if(!A){
      free(adjloc);
      cudaFree(adj);
    }

    cudaFree(omegas);
    cudaFree(y2);
    cudaFree(f2);
    cudaFree(f3);
    cudaFree(state);

    dp45_destroy();

    cublasDestroy(handle);

    return 0;
}
