#!/usr/bin/env python
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import timeit
from scipy.linalg import svdvals, svd, eig, lu_factor, lu_solve
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import argparse

def PCA(X,filebase,verbose=False, reload=False):
    if not reload or not os.path.exists(filebase+'s.dat'):
        start=timeit.default_timer()
        u,s,v=svd(X)
        stop=timeit.default_timer()
        if verbose:
            print('svd runtime:',stop-start,flush=True)
    else:
        s=np.load(filebase+'s.npy')
        u=np.load(filebase+'u.npy')
        v=np.load(filebase+'v.npy')

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
    np.save(filebase+'s.npy',s)
    np.save(filebase+'u.npy',u)
    np.save(filebase+'v.npy',v)
    errs=np.array([ranks,errs])
    np.save(filebase+'errs.npy',errs)


    return s,u,v,errs

def resDMD(U,V,S,X,Y,filebase,verbose=False,reload=False):
    if not reload or not os.path.exists(filebase+'res.dat'):
        start=timeit.default_timer()
        A=Y.dot(np.conjugate(V).T*1/S)
        Ktilde=np.conjugate(U.T).dot(A)
        evals,evecs=np.linalg.eig(Ktilde)
        stop=timeit.default_timer()
        if verbose:
            print('eig runtime:',stop-start,flush=True)

        start=timeit.default_timer()
        res=np.linalg.norm(A.dot(evecs)-evals[np.newaxis,:]*U.dot(evecs),axis=0)
        stop=timeit.default_timer()
        if verbose:
            print('residue runtime:',stop-start,flush=True)

        phis=(np.conjugate(V).T*1/S).dot(evecs)
        bs=X.dot(phis)/np.linalg.norm(Y.dot(phis),axis=0)
        np.save(filebase+'res.npy',res)
        np.save(filebase+'evals.npy',evals)
        np.save(filebase+'evecs.npy',evecs)
        np.save(filebase+'phis.npy',phis)
        np.save(filebase+'bs.npy',bs)
    else:
        res=np.load(filebase+'res.npy')
        evals=np.load(filebase+'evals.npy')
        evecs=np.load(filebase+'evecs.npy')
        phis=np.load(filebase+'phis.npy')
        bs=np.load(filebase+'bs.npy')
    return evals,evecs,res,phis,bs

def resDMDpseudo(U,V,S,X,Y,zs,evals,evecs,filebase,verbose,reload=False):
    n0=0
    zs_prev=[]
    zs_new=zs
    vals=[]
    its=[]
    xis=[]
    start=timeit.default_timer()

    if reload and os.path.exists(filebase+'zs.dat'):
        zs_prev=np.load(filebase+'zs.npy').tolist()
        zs_new=np.setdiff1d(zs,zs_prev)
        vals=np.load(filebase+'pseudo.npy').tolist()
        its=np.load(filebase+'its.npy').tolist()
        xis=list(np.load(filebase+'xis.npy').reshape((len(zs_prev),-1)))

    if len(zs_new)>0:
        A=Y.dot(np.conjugate(V).T*1/S)
        for n in range(len(zs_new)):
            z=zs_new[n]
            if verbose:
                print('%f\t%f\t%f'%(n/len(zs_new),np.real(z),np.imag(z)),end='\r',flush=True)
            i=np.argmin(np.abs(z-evals))
            xi=evecs[:,i]
            A2=(A-z*U)
            C2=np.conjugate(A2).T.dot(A2)
            lu,piv=lu_factor(C2)
            residue=np.linalg.norm(A2.dot(xi))

            for m in range(100):
                xi=lu_solve((lu,piv),xi)
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
            np.save(filebase+'zs.npy',np.array(zs_prev))
            np.save(filebase+'pseudo.npy',np.array(vals))
            np.save(filebase+'xis.npy',np.array(xis))
            np.save(filebase+'its.npy',np.array(its))
    stop=timeit.default_timer()
    if verbose:
        print()
        print('pseudospectra runtime:',stop-start,flush=True)


    return zs_prev,vals,xis,its

if __name__ == "__main__":

    #Command line arguments
    parser = argparse.ArgumentParser(description='Numerical integration of networks of phase oscillators.')
    parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output.')
    parser.add_argument("--verbose", type=int, required=False, dest='verbose', default=1, help='Verbose printing.')
    parser.add_argument("--pcatol", type=float, required=False, dest='pcatol', default=1E-10, help='Reconstruction error cutoff for pca.')
    parser.add_argument("--resmax", type=float, required=False, dest='resmax', default=None, help='Maximum residue.')
    parser.add_argument("--scaler", type=float, required=False, dest='scaler', default=20, help='Pseudospectra real scale.')
    parser.add_argument("--scalei", type=float, required=False, dest='scalei', default=25, help='Pseudospectra imaginary scale.')
    parser.add_argument("--nr", type=int, required=False, dest='nr', default=26, help='Number of real pseudospectra points.')
    parser.add_argument("--ni", type=int, required=False, dest='ni', default=26, help='Number of imaginary pseudospectra points.')
    parser.add_argument("--num_traj", type=int, required=False, dest='num_traj', default=0, help='Number of trajectories.')
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
    num_traj = args.num_traj

    start=timeit.default_timer()
    thetas=[]
    lengths=[]
    n=0
    read_traj=True
    while read_traj:
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
        orders=np.fromfile(filebase+'order.dat',dtype=np.float64)
        #lastind=np.min([len(theta)-1,5*np.where(orders<np.sqrt(np.pi/(4*N)))[0][1]])
        lastind=len(theta)
        lengths=lengths+[lastind]
        print(lastind)
        theta=theta[:lastind]
        theta=theta-np.mean(omega)*dt*np.arange(theta.shape[0])[:,np.newaxis]
        thetas=thetas+[theta]
        n=n+1
        if num_traj == 0:
            read_traj=os.path.exists('%s%i.out'%(filebase0,n))
        elif n>=num_traj:
            read_traj=False


    X=np.concatenate([thetas[i][:-1] for i in range(len(thetas))],axis=0)
    X=np.concatenate([np.cos(X),np.sin(X)],axis=1)

    Y=np.concatenate([thetas[i][1:] for i in range(len(thetas))],axis=0)
    Y=np.concatenate([np.cos(Y),np.sin(Y)],axis=1)
    if verbose:
        print('shape:', X.shape, flush=True)

    np.save(filebase0+'n0s.npy',np.array(lengths))
    s,u,v,errs=PCA(X,filebase0,verbose)
    f=interp1d(errs[0],errs[1])
    r=int(root_scalar(lambda x:f(x)-pcatol,bracket=(errs[0][0],errs[0][-1])).root)
    if verbose:
        print('rank:',r,flush=True)
    evals,evecs,res,phis,bs=resDMD(u[:,:r],v[:r,:],s[:r],X,Y,filebase0,verbose)

    if nr>1:
        murs=-scaler+2*scaler*np.arange(nr)/(nr-1)
    else:
        murs=np.array([0])
    muis=-scalei+2*scalei*np.arange(ni)/(ni-1)

    zs=np.exp((murs[:,np.newaxis]+1j*muis[np.newaxis,:]).ravel()*dt)

    zs_prevs,vals,xis,its=resDMDpseudo(u[:,:r],v[:r,:],s[:r],X,Y,zs,evals,evecs,filebase0,verbose)
    stop=timeit.default_timer()
    print('runtime:',stop-start)
