#!/usr/bin/env python
import sys
import os
import numpy as np
import timeit
from scipy.linalg import svd, eig, lu_factor, lu_solve,eigh
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import argparse
from dmd import *
import pysindy as ps


if __name__ == "__main__":

    #Command line arguments
    parser = argparse.ArgumentParser(description='Numerical integration of networks of phase oscillators.')
    parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output.')
    parser.add_argument("--filesuffix", type=str, required=False, dest='filesuffix', default='', help='Suffix string for file output.')
    parser.add_argument("--verbose", type=int, required=False, dest='verbose', default=1, help='Verbose printing.')
    parser.add_argument("--pcatol", type=float, required=False, dest='pcatol', default=1E-8, help='Reconstruction error cutoff for pca.')
    parser.add_argument("--resmax", type=float, required=False, dest='resmax', default=None, help='Maximum residue.')
    parser.add_argument("--minr", type=float, required=False, dest='minr', default=-0.05, help='Pseudospectra real scale.')
    parser.add_argument("--maxr", type=float, required=False, dest='maxr', default=0.01, help='Pseudospectra real scale.')
    parser.add_argument("--mini", type=float, required=False, dest='mini', default=-1.5, help='Pseudospectra imaginary scale.')
    parser.add_argument("--maxi", type=float, required=False, dest='maxi', default=1.5, help='Pseudospectra imaginary scale.')
    parser.add_argument("--nr", type=int, required=False, dest='nr', default=26, help='Number of real pseudospectra points.')
    parser.add_argument("--ni", type=int, required=False, dest='ni', default=26, help='Number of imaginary pseudospectra points.')
    parser.add_argument("--xskip", type=int, required=False, dest='xskip', default=1, help='Skips per space.')
    parser.add_argument("--tskip", type=int, required=False, dest='tskip', default=1, help='Skips per time.')
    parser.add_argument("--M", type=int, required=False, dest='M', default=1, help='Number of angle multiples to include in library.')
    parser.add_argument("--D", type=int, required=False, dest='D', default=0, help='Number of angle pairs to include in library.')
    parser.add_argument("--rank", type=int, required=False, dest='rank', default=None, help='Ritz rank for svd.')
    parser.add_argument("--savepca", type=int, required=False, dest='savepca', default=0, help='Save dense PCA data.')
    parser.add_argument("--runpseudo", type=int, required=False, dest='runpseudo', default=1, help='Run the pseudospectrum calculation.')
    parser.add_argument("--load", type=int, required=False, dest='load', default=0, help='Load data from previous runs.')
    args = parser.parse_args()

    print(*sys.argv,flush=True)
    filebase0 = args.filebase
    filesuffix = args.filesuffix
    verbose = args.verbose
    pcatol = args.pcatol
    resmax = args.resmax
    minr = args.minr
    maxr = args.maxr
    mini = args.mini
    maxi = args.maxi
    D = args.D
    M = args.M
    nr = args.nr
    ni = args.ni
    rank = args.rank
    save = args.savepca
    runpseudo = args.runpseudo
    load = args.load
    tskip=args.tskip
    xskip=args.xskip


    start=timeit.default_timer()
    file=open(filebase0+'.out')
    dat=file.readlines()
    print(dat[0])
    print(dat[-1])
    Nx=int(dat[1].split()[0])
    nx=Nx//xskip
    dt=float(dat[1].split(' ')[11])*tskip
    dx=float(dat[1].split(' ')[3])/Nx
    def diff(x,d):
        ret=x.copy()
        if d[0]>0:
            ret=ps.FiniteDifference(axis=1,d=d[0],periodic=True)._differentiate(ret,dx)
        if d[1]>0:
            ret=ps.FiniteDifference(axis=2,d=d[1],periodic=True)._differentiate(ret,dx)
        return ret

    X=np.fromfile(filebase0+'phases.dat',dtype=np.complex128).reshape((-1,Nx,Nx))[::tskip,::xskip,::xskip]

    pows=[[1,0],[0,1]]
    for i in range(M+1):
        for j in range(M+1):
            if i+j<=M and i+j>1:
                pows=pows+[[i,j]]
    library=[lambda x,d=d: (np.real(x)**d[0])*(np.imag(x)**d[1]) for d in pows]
    pows2=[]
    for i in range(D+1):
        for j in range(D+1):
            if i+j<=D and i+j>0:
                pows2=pows2+[[i,j]]

    library=library+[lambda x,d=d: np.real(diff(x,d)) for d in pows2]+[lambda x,d=d: np.imag(diff(x,d)) for d in pows2]
    if verbose:
        print('Library: %i terms'%len(library))
    filebase='%s%s'%(filebase0,filesuffix)
    start=timeit.default_timer()
    X=np.array([lib(X) for lib in library]).transpose(1,0,2,3).reshape(-1,nx*nx*len(library))

    Xinds=slice(0,len(X)-1,1)
    Yinds=slice(1,len(X),1)
    stop=timeit.default_timer()

    if verbose:
        print('Library build time:', stop-start, flush=True)
        print('shape:', X.shape, flush=True)


    filebase=filebase0+filesuffix

    s,u,v,errs=PCA(X[Xinds],filebase,verbose,rank=rank,save=save,load=load)

    r=int(errs[0][-1])
    try:
        f=interp1d(errs[0],errs[1])
        r=int(root_scalar(lambda x:f(x)-pcatol,bracket=(errs[0][0],errs[0][-1])).root)
    except:
        pass
    if verbose:
        print('rank:',r,flush=True)
        if(r==errs[0][-1]):
            print('Warning: numerical precision may be limiting achievable pcatol')
    U=u[:,:r].copy()
    del u
    V=v[:r,:].copy()
    del v
    S=s[:r].copy()
    del s
    evals,evecs,res,phis,bs,A=resDMD(U,V,S,X[Xinds],X[Yinds],filebase,verbose,load=load)

    if nr>1:
        murs=minr+(maxr-minr)*np.arange(nr)/(nr-1)
    else:
        murs=np.array([0])
    muis=mini+(maxi-mini)*np.arange(ni)/(ni-1)

    zs=np.exp((murs[:,np.newaxis]+1j*muis[np.newaxis,:]).ravel()*dt)

    if runpseudo:
        zs_prevs,pseudo,xis,its=resDMDpseudo(U,A,zs,evals,evecs,filebase,verbose,load=load)
    stop=timeit.default_timer()
    if verbose:
        printmem(locals())
    print('runtime:',stop-start)
