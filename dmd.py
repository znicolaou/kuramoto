#!/usr/bin/env python
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import timeit
from scipy.linalg import svdvals, svd, eig
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import argparse

###########################################################################
def PCA(X,filebase,verbose=False):
    if not os.path.exists(filebase+'s.dat'):
        start=timeit.default_timer()
        u,s,v=svd(X)
        stop=timeit.default_timer()
        if verbose:
            print('svd runtime:',stop-start,flush=True)

        try:
            rank=np.where(s<s.max() * max(X.shape[0],X.shape[1]) * np.finfo(X.dtype).eps)[0][0]
        except:
            rank=min(X.shape[0],X.shape[1])
        print('full rank:', rank)
        errs=[]
        if rank>10:
            ranks=np.arange(rank//10,rank,rank//10)
        else:
            ranks=np.arange(1,rank)
        for r in ranks:
            errs=errs+[np.linalg.norm(X-u[:,:r].dot(s[:r,np.newaxis]*v[:r]))/np.linalg.norm(X)]
        s[:rank].tofile(filebase+'s.dat')
        u[:,:rank].tofile(filebase+'u.dat')
        v[:rank,:].tofile(filebase+'v.dat')
        errs=np.array([ranks,errs])
        errs.tofile(filebase+'errs.dat')

    else:
        s=np.fromfile(filebase+'s.dat')
        rank=len(s)
        u=np.fromfile(filebase+'u.dat').reshape((-1,rank))
        v=np.fromfile(filebase+'v.dat').reshape((rank,-1))
        errs=np.fromfile(filebase+'errs.dat').reshape((2,-1))

    return s,u,v,errs
###########################################################################

###########################################################################
def resDMD(U,V,S,X,Y,filebase,verbose=False):
    if not os.path.exists(filebase+'res.dat'):
        start=timeit.default_timer()
        A=Y.dot(np.conjugate(V).T*1/S)
        B=X.dot(np.conjugate(V).T*1/S)
        Ktilde=np.conjugate(U.T).dot(A)
        evals,evecs=np.linalg.eig(Ktilde)
        stop=timeit.default_timer()
        if verbose:
            print('eig runtime:',stop-start,flush=True)

        start=timeit.default_timer()
        res=np.linalg.norm(A.dot(evecs)-evals[np.newaxis,:]*B.dot(evecs),axis=0)/np.linalg.norm(B.dot(evecs),axis=0)
        stop=timeit.default_timer()
        if verbose:
            print('residue runtime:',stop-start,flush=True)
        res.tofile(filebase+'res.dat')
        evals.tofile(filebase+'evals.dat')
        evecs.tofile(filebase+'evecs.dat')
    else:
        res=np.fromfile(filebase+'res.dat')
        evals=np.fromfile(filebase+'evals.dat',dtype=np.complex128)
        evecs=np.fromfile(filebase+'evecs.dat',dtype=np.complex128).reshape((evals.shape[0],evals.shape[0]))
    return evals,evecs,res
###########################################################################

###########################################################################
def resDMDpseudo(U,V,S,X,Y,zs,evals,evecs,filebase,verbose):
    n0=0
    zs_prev=[]
    zs_new=zs
    vals=[]
    its=[]
    xis=[]
    start=timeit.default_timer()

    if os.path.exists(filebase+'zs.dat'):
        zs_prev=np.fromfile(filebase+'zs.dat',dtype=np.complex128).tolist()
        zs_new=np.setdiff1d(zs,zs_prev)
        vals=np.fromfile(filebase+'pseudo.dat',dtype=np.float64).tolist()
        its=np.fromfile(filebase+'its.dat',dtype=np.int64).tolist()
        xis=list(np.fromfile(filebase+'xis.dat',dtype=np.complex128).reshape((len(zs_prev),-1)))

    if len(zs_new)>0:
        A=Y.dot(np.conjugate(V).T*1/S)
        B=X.dot(np.conjugate(V).T*1/S)
        #inverse power iteration with C2
        for n in range(len(zs_new)):
            z=zs_new[n]
            print('%f\t%f\t%f'%(n/len(zs_new),np.real(z),np.imag(z)),end='\n',flush=True)
            i=np.argmin(np.abs(z-evals))
            xi=evecs[:,i]
            A2=(A-z*B)
            C2=np.conjugate(A2).T.dot(A2)
            residue=np.linalg.norm(A2.dot(xi))
            for m in range(100):
                xi=np.linalg.solve(C2,xi)
                xi=xi/np.linalg.norm(xi)
                newres=np.linalg.norm(A2.dot(xi))
                if np.linalg.norm((residue-newres)/residue)<1E-3:
                    residue=newres
                    break
                residue=newres
            zs_prev=zs_prev+[z]
            vals=vals+[residue]
            xis=xis+[xi]
            its=its+[m]
            np.array(zs_prev).tofile(filebase+'zs.dat')
            np.array(vals).tofile(filebase+'pseudo.dat')
            np.array(xis).tofile(filebase+'xis.dat')
            np.array(its).tofile(filebase+'its.dat')
        #singular value solution with C
        # L=np.conjugate(A).T.dot(A)
        # G=np.conjugate(A).T.dot(B)
        # E=np.conjugate(B).T.dot(B)
        # vals=[]
        # for i in range(len(zs)):
        #     z=zs[i]
        #     if verbose:
        #         print('%f'%(i/len(zs)),end='\r',flush=True)
        #     C=L-G*z-np.conjugate(G).T*np.conjugate(z)+np.abs(z)**2*E
        #     u,s,vh=svd(C)
        #     vals=vals+[np.linalg.norm((A-z*B).dot(np.conjugate(vh[-1])))]
    stop=timeit.default_timer()
    if verbose:
        print()
        print('pseudospectra runtime:',stop-start,flush=True)


    return zs_prev,vals,xis,its
###########################################################################

if __name__ == "__main__":

    #Command line arguments
    parser = argparse.ArgumentParser(description='Numerical integration of networks of phase oscillators.')
    parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output.')
    parser.add_argument("--verbose", type=int, required=False, dest='verbose', default=1, help='Verbose printing.')
    parser.add_argument("--pcatol", type=float, required=False, dest='pcatol', default=1E-7, help='Reconstruction error cutoff for pca.')
    parser.add_argument("--resmax", type=float, required=False, dest='resmax', default=None, help='Maximum residue.')
    parser.add_argument("--scaler", type=float, required=False, dest='scaler', default=0.6, help='Maximum residue.')
    parser.add_argument("--scalei", type=float, required=False, dest='scalei', default=3, help='Maximum residue.')
    parser.add_argument("--nr", type=int, required=False, dest='nr', default=26, help='Maximum residue.')
    parser.add_argument("--ni", type=int, required=False, dest='ni', default=26, help='Maximum residue.')
    args = parser.parse_args()

    print(*sys.argv,flush=True)
    filebase0 = args.filebase
    verbose = args.verbose
    pcatol = args.pcatol
    resmax = args.resmax
    scaler = args.scaler
    scalei = args.scalei
    nr = args.nr
    ni = args.ni

    start=timeit.default_timer()
    thetas=[]
    n=0
    while os.path.exists('%s%i.out'%(filebase0,n)):
        filebase='%s%i'%(filebase0,n)
        file=open(filebase+'.out')
        lines=file.readlines()
        N,K,t1,dt,c,seed=np.array(lines[0].split(),dtype=np.float64)
        N=int(N)
        K=int(K)
        if verbose:
            print(lines[1])
            print(lines[-1])
        file.close()

        omega=np.fromfile(filebase+'frequencies.dat',dtype=np.float64)
        N=len(omega)
        theta=np.fromfile(filebase+'thetas.dat',dtype=np.float64).reshape((-1,N))
        theta=theta-np.mean(omega)*dt*np.arange(theta.shape[0])[:,np.newaxis]
        thetas=thetas+[theta]
        n=n+1

    X=np.concatenate([theta[:-1] for theta in thetas],axis=0)
    X=np.concatenate([np.sin(X), np.cos(X), np.ones((X.shape[0],1))],axis=1)
    Y=np.concatenate([theta[1:] for theta in thetas],axis=0)
    Y=np.concatenate([np.sin(Y), np.cos(Y), np.ones((Y.shape[0],1))],axis=1)

    if verbose:
        print('shape:', X.shape, flush=True)

    s,u,v,errs=PCA(X,filebase0,verbose)
    f=interp1d(errs[0],errs[1])
    r=int(root_scalar(lambda x:f(x)-pcatol,bracket=(errs[0][0],errs[0][-1])).root)
    if verbose:
        print('rank:',r,flush=True)
    evals,evecs,res=resDMD(u[:,:r],v[:r,:],s[:r],X,Y,filebase0,verbose)

    if nr>1:
        murs=-scaler+2*scaler*np.arange(nr)/(nr-1)
    else:
        murs=np.array([0])
    muis=-scalei+2*scalei*np.arange(ni)/(ni-1)

    inds=np.intersect1d(np.where(np.abs(np.real(np.log(evals)/dt))<scaler)[0],np.where(np.abs(np.imag(np.log(evals)/dt))<scalei)[0])
    zs=np.concatenate([evals[inds],np.exp((murs[:,np.newaxis]+1j*muis[np.newaxis,:]).ravel()*dt)])

    zs_prevs,vals,xis,its=resDMDpseudo(u[:,:r],v[:r,:],s[:r],X,Y,zs,evals,evecs,filebase0,verbose)
    stop=timeit.default_timer()
    print('runtime:',stop-start)
