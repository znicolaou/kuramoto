//Zachary G. Nicolaou 2/10/2024
//Integrate the Kuramoto model with adaptive Runke Kutta timestepping on a gpu
//Default adjacency, initial conditions, and frequencies follow volcano
//nvcc -lcuda -lcublas -lcurand -O3 -o kuramoto_64 dp45_64.cu kuramoto_64.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include "dp45_64.h"
#include "cublas_v2.h"
#include <curand_kernel.h>

typedef struct parameters
{
  cublasHandle_t handle;
  unsigned long int N;
  unsigned long int K;
  int A;
  double *y2;
  double *f2;
  double *f3;
  double *floc;
  double *omegas;
  double *adj;
  double c1;
  double t0;
  double t1;
  int steps;
  int verbose;
  int dense;
  double *t_eval;
  int n_eval;
  int eval_i;
  double *yloc;
  double *ones;
  char *filebase;
  struct timeval start;
  curandStatePhilox4_32_10_t *state;
}parameters;

__global__ void tot_kuramoto (double* y, const unsigned long int N, double* y2, double* f, double* f2, double *omegas, const double c1) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<N){
    f[i]=omegas[i]+c1*(y2[2*i+1]*f2[2*i]-y2[2*i]*f2[2*i+1]);
  }
}

__global__ void makey2 (double* y, const unsigned long int N, double* y2, double *omegas, const double t) {
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

__global__ void m1(double* y, double* f, int N, int K, curandStatePhilox4_32_10_t *globalstate) {
  int m = blockIdx.x*blockDim.x+threadIdx.x;

  if (m < K) {
    double tmpSum = 0;
    double tmpSum2 = 0;
    curandStatePhilox4_32_10_t state = *globalstate;
    skipahead(m*N, &state);

    double s=pow(-1.0,m);

    for (int n = 0; n < N; n++) {
      double umn=(2*(curand(&state)%2)-1.0);
      tmpSum += y[2*n] * s*umn;
      tmpSum2 += y[2*n+1] * s*umn;
    }
    f[2*m] = tmpSum/N;
    f[2*m+1] = tmpSum2/N;
  }
}

__global__ void m2(double* y, double* f, int N, int K, curandStatePhilox4_32_10_t *globalstate) {
  int n = blockIdx.x*blockDim.x+threadIdx.x;

  if (n < N) {
    double tmpSum = 0;
    double tmpSum2 = 0;
    curandStatePhilox4_32_10_t state = *globalstate;
    skipahead(n, &state);

    for (int m = 0; m < K; m++) {
      double umn=(2*(curand(&state)%2)-1.0);
      skipahead(N-1, &state);
      tmpSum += y[2*m] * umn;
      tmpSum2 += y[2*m+1] * umn;
    }
    f[2*n] = tmpSum;
    f[2*n+1] = tmpSum2;
  }
}


void makecoupling(double t, double *y, double *f, void* pars){
  parameters *p = (parameters *)pars;
  if(p->A){
    m1<<<(p->K+255)/256, 256>>> (y,p->f3, p->N, p->K, p->state);
    m2<<<(p->N+255)/256, 256>>> (p->f3,f, p->N, p->K, p->state);
  }
  else{
    double alpha=1;
    double beta=0;
    cublasDgemv(p->handle, CUBLAS_OP_T, p->N, p->N, &alpha, p->adj, p->N, p->y2, 2, &beta, f, 2);
    cublasDgemv(p->handle, CUBLAS_OP_T, p->N, p->N, &alpha, p->adj, p->N, (p->y2)+1, 2, &beta, f+1, 2);
  }
}

void dydt (double t, double *y, double *f, void *pars){

  parameters *p = (parameters *)pars;

  makey2<<<(p->N+255)/256, 256>>>(y, p->N, p->y2, p->omegas, t);
  makecoupling(t,p->y2,p->f2,pars);
  tot_kuramoto<<<(p->N+255)/256, 256>>>(y, p->N, p->y2, f, p->f2, p->omegas, p->c1);

}

__global__ void makeu1 (double* u1, unsigned int *r, const unsigned long int N, const int K, curandStatePhilox4_32_10_t *globalstate) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<N*K){
    curandStatePhilox4_32_10_t state=*globalstate;
    skipahead(i, &state);
    u1[i] = (2*(curand(&state)%2)-1.0);
  }
}

__global__ void makeu2 (double* u1, double *u2, const unsigned long int N, const int K) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<N*K){
    int m=i/N;
    u2[i]=pow(-1,m)*u1[i]/N;
  }
}

void getadj(void *pars){
  parameters *p = (parameters *)pars;

  double *u1, *u2;
  const double alpha=1.0;
  const double beta=0.0;

  cudaMalloc ((void**)&u1, p->K*p->N*sizeof(double));
  cudaMalloc ((void**)&u2, p->K*p->N*sizeof(double));

  makeu1<<<(p->N*p->K+255)/256, 256>>>(u1, (unsigned int *)u2, p->N, p->K, p->state);
  makeu2<<<(p->N*p->K+255)/256, 256>>>(u1, u2, p->N, p->K);
  cublasDgemm(p->handle,CUBLAS_OP_N,CUBLAS_OP_T,p->N,p->N,p->K,&alpha,u2,p->N,u1,p->N,&beta,p->adj,p->N);

  cudaFree(u1);
  cudaFree(u2);
}


void step_eval(double t, double h, double* y, void *pars){
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
  if(p->dense>=1){
    strcpy(file,p->filebase);
    strcat(file, "times.dat");
    FILE *outtimes = fopen(file,"ab");
    fwrite(&t,sizeof(double),1,outtimes);
    fflush(outtimes);
    fclose(outtimes);
  }

  FILE *outanimation, *outcouplings;
  if(p->dense>=1){
    strcpy(file,p->filebase);
    strcat(file, "thetas.dat");
    outanimation = fopen(file,"ab");
  }
  if(p->dense>=2){
    strcpy(file,p->filebase);
    strcat(file, "couplings.dat");
    outcouplings = fopen(file,"ab");
  }


  double X,Y;
  static double *r=(double*)calloc(p->n_eval,sizeof(double));
  int eval_j=p->eval_i;
  while (t >= p->t_eval[eval_j] && eval_j<p->n_eval){
    eval_j++;
  }
  int num=eval_j-p->eval_i;
  int ind=0;

  while (t >= p->t_eval[p->eval_i] && p->eval_i<p->n_eval){
    double *y_eval;
    y_eval=dp45_eval(t,p->t_eval[p->eval_i]);

    makey2<<<(p->N+255)/256, 256>>>(y_eval, p->N, p->y2, p->omegas, t);

    cublasDdot(p->handle,p->N, p->y2, 2, p->ones, 1, &X);
    cublasDdot(p->handle,p->N, p->y2+1, 2, p->ones, 1, &Y);
    r[ind++]=pow((X/p->N*X/p->N+Y/p->N*Y/p->N),0.5);
    if(p->dense>=1){
      cublasGetVector(p->N, sizeof(double), y_eval, 1, p->yloc, 1);
      fwrite(p->yloc,sizeof(double),p->N,outanimation);
      fflush(outanimation);
    }
    if(p->dense>=2){
      makecoupling(t,p->y2,p->f2,pars);
      cublasGetVector(2*p->N, sizeof(double), p->f2, 1, p->floc, 1);
      fwrite(p->floc,sizeof(double),2*p->N,outcouplings);
      fflush(outcouplings);
    }

    p->eval_i++;
  }
  if(p->dense>=1){
    fclose(outanimation);
  }
  if(p->dense>=2){
    fclose(outcouplings);
  }

  if (num>0){
    strcpy(file,p->filebase);
    strcat(file,"order.dat");
    FILE *outorder=fopen(file,"ab");
    fwrite(r,sizeof(double),num,outorder);
    fflush(outorder);
    fclose(outorder);
  }
  cublasGetVector(p->N, sizeof(double), y, 1, p->yloc, 1);

  strcpy(file,p->filebase);
  strcat(file,"fs.dat");
  FILE *outlast=fopen(file,"wb");

  fwrite(p->yloc,sizeof(double),p->N,outlast);
  fwrite(&t,sizeof(double),1,outlast);
  fwrite(&h,sizeof(double),1,outlast);
  fwrite(p->floc,sizeof(double),2*p->N,outlast);
  fflush(outlast);
  fclose(outlast);
}

int main (int argc, char* argv[]) {
    struct timeval start,end;
    gettimeofday(&start,NULL);

    double c1, t1, dt;
    double freq, atl, rtl, I;
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
    int dense=3;
    int normal=0;
    int reload=0;

    while (optind < argc) {
      if ((c = getopt(argc, argv, "N:K:I:D:c:g:t:d:f:s:r:a:hvFnRA")) != -1) {
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
              I = (double)atof(optarg);
              break;
          case 'c':
              c1 = (double)atof(optarg);
              break;
          case 'g':
              gpu = (double)atof(optarg);
              break;
          case 't':
              t1 = (double)atof(optarg);
              break;
          case 'd':
              dt = (double)atof(optarg);
              break;
          case 'f':
              freq = (double)atof(optarg);
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
          case 'D':
              dense = (int)atoi(optarg);
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
      printf("usage:\tkuramoto [-hvnRFA] [-N N] [-K K] [-D D]\n");
      printf("\t[-c c] [-t t] [-d dt] [-f f] [-s seed] \n");
      printf("\t[-I init] [-r rtol] [-a atol] [-g gpu] filebase \n\n");
      printf("-h for help \n");
      printf("-v for verbose \n");
      printf("-n for normal random frequencies (default is cauchy) \n");
      printf("-R to reload adjacency, frequencies, and initial conditions from files if possible\n");
      printf("-F for fixed timestep \n");
      printf("-A to regenerate volcano adjacency each step to save memory \n");
      printf("D is the output density level. 0 for minimal, 1 for phases, 2 for phases and couplings, 3 for phases, couplings, and adjacency\n");
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

    double t=0,h;
    int j=0;
    FILE *out, *in;

    char file[256];
    strcpy(file,filebase);
    strcat(file,".out");
    out = fopen(file,"ab");

    double *omegasloc, *omegas, *yloc, *adjloc, *adj, *y2, *f2, *f3, *floc, *ones;
    yloc = (double*)calloc(N,sizeof(double));
    floc = (double*)calloc(2*N,sizeof(double));
    omegasloc = (double*)calloc(N,sizeof(double));
    adjloc = (double*)calloc(N*N,sizeof(double));

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
      req=100*N*sizeof(double);
    }
    else{
      req=(100+N+2*K)*N*sizeof(double);
    }
    printf("GPU Memory: %lu %lu %lu\n", fr, total, req);
    fprintf(out,"GPU Memory: %lu %lu %lu\n", fr, total, req);
    if(fr < req) {
      printf("GPU Memory low!\n");
      fprintf(out,"GPU Memory low!\n");
      return 0;
    }
    fflush(out);


    cudaMalloc ((void**)&omegas, N*sizeof(double));
    cudaMalloc ((void**)&y2, 2*N*sizeof(double));
    cudaMalloc ((void**)&f2, 2*N*sizeof(double));
    cudaMalloc ((void**)&f3, 2*K*sizeof(double));
    if(!A){
      cudaMalloc((void**)&adj,sizeof(double)*N*N);
    }
    cudaMalloc((void**)&ones,sizeof(double)*N);
    for(int i=0; i<N; i++){
      yloc[i]=1;
    }
    cublasSetVector(N, sizeof(double), yloc, 1, ones, 1);

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
      size_t read=fread(yloc,sizeof(double),N,in);
      if (read!=N){
        printf("initial conditions file not compatible with N!\n");
        fprintf(out,"initial conditions file not compatible with N!\n");
        reloaded=0;
      }
      if(reloaded){
        read=fread(&t,sizeof(double),1,in);
        read=fread(&h,sizeof(double),1,in);
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
      fwrite(yloc,sizeof(double),N,in);
      fclose(in);

      cublasSetVector(N, sizeof(double), yloc, 1, omegas, 1);
    }

    strcpy(file,filebase);
    strcat(file, "frequencies.dat");
    if (reload && (in = fopen(file,"r")))
    {
        printf("Using frequencies from file\n");
        fprintf(out, "Using frequencies from file\n");
        size_t read=fread(omegasloc,sizeof(double),N,in);
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
            double u = rand() / (double)RAND_MAX;
            double v = rand() / (double)RAND_MAX;
            omegasloc[j] = freq*pow(-2*log(u),0.5)*cos(2*M_PI*v);

          }
          else{
            double u = rand() / (double)RAND_MAX;
            omegasloc[j] = freq*tan(M_PI * (u - 0.5));
          }
        }
        in=fopen(file,"wb");
        fwrite(omegasloc,sizeof(double),N,in);
        fclose(in);
    }

    int n0=int(t/dt)+1;
    int n1=int(t1/dt)+1;
    int n_eval=n1-n0;
    double *t_eval=(double *)calloc(n_eval,sizeof(double));
    int ind=0;
    for(int n=n0; n<n1; n++){
      t_eval[ind++]=dt*n;
    }

    cublasSetVector (N, sizeof(double), omegasloc, 1, omegas, 1);


    parameters pars={.handle=handle, .N=N, .K=K, .A=A, .y2=y2, .f2=f2, .f3=f3, .floc=floc, .omegas=omegas, .adj=adj, .c1=c1, .t0=t, .t1=t1, .steps=0, .verbose=verbose, .dense=dense, .t_eval=t_eval, .n_eval=n_eval, .eval_i=0, .yloc=yloc, .ones=ones, .filebase=filebase,.start=start, .state=state};

    strcpy(file,filebase);
    strcat(file, "adj.dat");
    if (!A && reload && (in = fopen(file,"r")))
    {
        printf("Using adjacency from file\n");
        fprintf(out, "Using adjacency from file\n");
        size_t read=fread(adjloc,sizeof(double),N*N,in);
        fclose(in);
        cublasSetVector(N*N, sizeof(double), adjloc, 1, adj, 1);
    }
    else {
        printf("Using random adjacency matrix\n");
        fprintf(out, "Using random adjacency matrix\n");
        if(!A){
          getadj(&pars);
          if (dense){
            cublasGetVector(N*N, sizeof(double), adj, 1, adjloc, 1);
            if (dense>=3) {
              in=fopen(file,"wb");
              fwrite(adjloc,sizeof(double),N*N,in);
              fclose(in);
            }
          }
        }
    }
    fflush(out);
    fclose(out);

    double* y=dp45_init(N, atl, rtl, fixed, yloc, handle, &dydt);

    if(!reloaded){
      strcpy(file,filebase);
      strcat(file,"order.dat");
      FILE *outorder=fopen(file,"wb");

      makey2<<<(N+255)/256, 256>>>(y, N, y2, omegas, t);

      double X,Y;
      cublasDdot(handle,N, y2, 2, ones, 1, &X);
      cublasDdot(handle,N, y2+1, 2, ones, 1, &Y);
      double r=pow((X/N*X/N+Y/N*Y/N),0.5);
      fwrite(&r,sizeof(double),1,outorder);
      fflush(outorder);
      fclose(outorder);

      if(dense>=1){
        strcpy(file,filebase);
        strcat(file,"thetas.dat");
        FILE *outanimation=fopen(file,"wb");
        fwrite(yloc,sizeof(double),N,outanimation);
        fflush(outanimation);
        fclose(outanimation);
        strcpy(file,filebase);
        strcat(file,"times.dat");
        FILE *outtimes=fopen(file,"wb");
        fclose(outtimes);

      }
      if(dense>=2){
        makecoupling(t,y2,f2,&pars);
        cublasGetVector(2*N, sizeof(double), f2, 1, floc, 1);
        strcpy(file,filebase);
        strcat(file,"couplings.dat");
        FILE *outcouplings=fopen(file,"wb");
        fwrite(floc,sizeof(double),2*N,outcouplings);
        fflush(outcouplings);
        fclose(outcouplings);
      }

    }

    double *y_eval=dp45_run(&t, &h, t1, &pars, &step_eval);

    //final state output with coupling appended
    parameters *p = &pars;
    y_eval=dp45_eval(t,p->t_eval[p->n_eval-1]);
    makey2<<<(p->N+255)/256, 256>>>(y_eval, p->N, p->y2, p->omegas, t);
    makecoupling(t,p->y2,p->f2,p);
    cublasGetVector(p->N, sizeof(double), y_eval, 1, p->yloc, 1);
    cublasGetVector(2*p->N, sizeof(double), p->f2, 1, p->floc, 1);

    strcpy(file,p->filebase);
    strcat(file,"fs.dat");
    FILE *outlast=fopen(file,"wb");

    fwrite(p->yloc,sizeof(double),p->N,outlast);
    fwrite(&t,sizeof(double),1,outlast);
    fwrite(&h,sizeof(double),1,outlast);
    fwrite(p->floc,sizeof(double),2*p->N,outlast);
    fflush(outlast);
    fclose(outlast);

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
    free(floc);
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
