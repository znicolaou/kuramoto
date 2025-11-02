#!/usr/bin/env python
import sys
import os
import numpy as np
import timeit
from scipy.linalg import svd, eig, lu_factor, lu_solve,eigh
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from sklearn.utils.extmath import randomized_svd
from memory_profiler import profile
import argparse
import dask.array as da
from dask.distributed import Client, LocalCluster
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

@profile
def PCA(X,filebase,verbose=False,rank=None,load=False,save=False):
    if not load or not os.path.exists(filebase+'s.npy'):
        start=timeit.default_timer()
        if rank is None:
            u,s,v=svd(X,full_matrices=False,check_finite=False)
        else:
            # u,s,v=randomized_svd(X, n_components=rank, n_oversamples=rank, random_state=0)
            u,sda,v=da.linalg.svd_compressed(X, rank, n_oversamples=rank, compute=False)
            s=sda.compute()
            #since we'll use u and v many times, compute and stope them here
            if not os.path.exists(filebase+'u'):
                os.mkdir(filebase+'u')
            da.to_npy_stack(filebase+'u',u)
            if not os.path.exists(filebase+'v'):
                os.mkdir(filebase+'v')    
            da.to_npy_stack(filebase+'v',v)
            u=da.from_npy_stack(filebase+'u')
            v=da.from_npy_stack(filebase+'v')

        stop=timeit.default_timer()
        if verbose:
            print('svd runtime:',stop-start,flush=True)
    else:
        s=np.load(filebase+'s.npy')
        u=da.from_npy_stack(filebase+'u')
        v=da.from_npy_stack(filebase+'v')

    if rank is None:
        try:
            rank=np.where(s<s.max() * max(X.shape[0],X.shape[1]) * np.finfo(X.dtype).eps)[0][0]
        except:
            rank=min(X.shape[0],X.shape[1])
    else:
        #The normal matrix eigenvalues/squared singular values can only be computed to numerical precision
        #so the numerical rank and svd precision is reduced for rank not None
        #rank=len(np.where(evals>evals.max() * min(X.shape[0],X.shape[1]) * np.finfo(X.dtype).eps)[0])a
        #but we can allow bigger Ritz space, which continues to decrease pca error for a bit more before saturating
        try:
            rank=np.where(s<s.max() * max(X.shape[0],X.shape[1]) * np.finfo(X.dtype).eps)[0][0]
        except:
            pass

    print('numerical rank:', rank,flush=True)
    if not load or not os.path.exists(filebase+'errs.npy'):
        start=timeit.default_timer()
        errs=[]
        if rank>10:
            ranks=np.arange((rank//10),rank+1,rank//10)
        else:
            ranks=np.arange(1,rank)
        for r in ranks:
            Xtilde=(u[:,:r]*s[:r]).dot(v[:r])
            err=da.linalg.norm(X-Xtilde)/da.linalg.norm(X)
            errs=errs+[err.compute()]
        errs=np.array([ranks,errs])

        np.save(filebase+'s.npy',s)
        np.save(filebase+'errs.npy',errs)
        stop=timeit.default_timer()
        if verbose:
            print('errs runtime:',stop-start,flush=True)
    else:
        errs=np.load(filebase+'errs.npy')
    
    return s,u,v,errs

@profile
def resDMD(U,V,S,X,Yinds,binds,filebase,verbose=False,load=False,save=True,dense_amplitudes=False):
    if not load or not os.path.exists(filebase+'res.npy'):
        start=timeit.default_timer()
        A=X[Yinds].dot(np.conjugate(V).T*1/S).compute()
        stop=timeit.default_timer()
        if verbose:
            print('A runtime:',stop-start,flush=True)
        start=timeit.default_timer()
        Ktilde=np.conjugate(U.T).dot(A).compute()
        stop=timeit.default_timer()
        if verbose:
            print('Ktilde runtime:',stop-start,flush=True)
        start=timeit.default_timer()
        evals,levecs,revecs=eig(Ktilde,left=True,right=True)
        stop=timeit.default_timer()
        if verbose:
            print('eig runtime:',stop-start,flush=True)

        phis=(np.conjugate(V).T*1/S).dot(revecs)

        start=timeit.default_timer()
        bs=(X[binds].dot(phis)/np.linalg.norm(X[Yinds[binds]].dot(phis),axis=0)).compute()

        stop=timeit.default_timer()
        if verbose:
            print('amplitude runtime:',stop-start,flush=True)

        start=timeit.default_timer()
        res=np.linalg.norm(A.dot(revecs)-evals[np.newaxis,:]*U.dot(revecs).compute(),axis=0)
        stop=timeit.default_timer()
        if verbose:
            print('res runtime:',stop-start,flush=True)

        start=timeit.default_timer()
        phis=phis.compute()
        stop=timeit.default_timer()
        if verbose:
            print('phis runtime:',stop-start,flush=True)
        start=timeit.default_timer()
        diag=np.sum(np.conjugate(levecs)*revecs,axis=0)
        revecsinv=1/diag[:,np.newaxis]*(np.conjugate(levecs).T)
        stop=timeit.default_timer()
        if verbose:
            print('revecsinv runtime:',stop-start,flush=True)
        start=timeit.default_timer()
        phitildes=V.T.dot(S*revecsinv.T).T.compute()
        stop=timeit.default_timer()
        if verbose:
            print('phitildes runtime:',stop-start,flush=True)
        
            
        if save:
            np.save(filebase+'A.npy',A)
            np.save(filebase+'res.npy',res)
            np.save(filebase+'evals.npy',evals)
            np.save(filebase+'revecs.npy',revecs)
            np.save(filebase+'levecs.npy',levecs)
            np.save(filebase+'phis.npy',phis)
            np.save(filebase+'phitildes.npy',phitildes)
            np.save(filebase+'bs.npy',bs)

    else:
        res=np.load(filebase+'res.npy')
        evals=np.load(filebase+'evals.npy')
        revecs=np.load(filebase+'evecs.npy')
        phis=np.load(filebase+'phis.npy')
        bs=np.load(filebase+'bs.npy')
        A=np.load(filebase+'A.npy')
    
    return evals,revecs,res,phis,bs,A

def resDMDpseudo(U,A,zs,evals,evecs,filebase,verbose,load=False,save=True):
    n0=0
    zs_prev=[]
    zs_new=zs
    pseudo=[]
    its=[]
    xis=[]
    start=timeit.default_timer()

    if load and os.path.exists(filebase+'zs.npy'):
        zs_prev=np.load(filebase+'zs.npy').tolist()
        zs_new=np.setdiff1d(zs,zs_prev)
        pseudo=np.load(filebase+'pseudo.npy').tolist()
        its=np.load(filebase+'its.npy').tolist()
        xis=list(np.load(filebase+'xis.npy').reshape((len(zs_prev),-1)))

    if len(zs_new)>0:
        for n in range(len(zs_new)):
            z=zs_new[n]
            if verbose:
                print('%f\t%f\t%f'%(n/len(zs_new),np.real(z),np.imag(z)),end='\r',flush=True)
            i=np.argmin(np.abs(z-evals))
            xi=evecs[:,i]
            A2=(A-z*U)
            C2=np.conjugate(A2).T.dot(A2)
            lu,piv=lu_factor(C2)
            residue=np.linalg.norm(A2.dot(xi)).compute()

            for m in range(100):
                xi=lu_solve((lu,piv),xi)
                xi=xi/np.linalg.norm(xi)
                newres=np.linalg.norm(A2.dot(xi)).compute()
                if np.linalg.norm((residue-newres)/residue)<1E-3:
                    residue=newres
                    break
                residue=newres

            zs_prev=zs_prev+[z]
            pseudo=pseudo+[residue]
            xis=xis+[xi]
            its=its+[m]
            if save:
                np.save(filebase+'zs.npy',np.array(zs_prev))
                np.save(filebase+'pseudo.npy',np.array(pseudo))
                np.save(filebase+'xis.npy',np.array(xis))
                np.save(filebase+'its.npy',np.array(its))
    stop=timeit.default_timer()
    if verbose:
        print()
        print('pseudospectra runtime:',stop-start,flush=True)

    return zs_prev,pseudo,xis,its

@profile 
def load_data(filebase0,filesuffix,verbose=False,num_traj=0,D=0,M=1,load=False):
    start=timeit.default_timer()
    thetas=[]
    lengths=[]
    n=0
    filebase=filebase0+filesuffix

    if load:
        X=da.from_npy_stack(filebase+'X')
        lengths=np.load(filebase+'n0s.npy')
    else:
        read_traj=True    
        while read_traj:
            filebase1='%s%i'%(filebase0,n)
            file=open(filebase1+'.out')
            lines=file.readlines()
            N,K,t1,dt,c,seed0=np.array(lines[0].split(),dtype=np.float64)
            N=int(N)
            length=int(t1/dt)
            shape=(length,N)
            lengths=lengths+[length]
            if verbose:
                print(lines[1])
                print(lines[-1])
            file.close()
        
            omega=da.from_array(np.memmap(filebase1+'frequencies.dat',dtype=np.float64,mode='r+',shape=(N)),name=False)
            theta=da.from_array(np.memmap(filebase1+'thetas.dat',dtype=np.float64,mode='r+',shape=shape),name=False)
        
            theta=theta-np.mean(omega)*dt*np.arange(theta.shape[0])[:,np.newaxis]
            thetas=thetas+[theta]
            n=n+1
            if num_traj == 0:
                read_traj=os.path.exists('%s%i.out'%(filebase0,n))
            elif n>=num_traj:
                read_traj=False
        
        X0=da.vstack(tuple(thetas))
        
        if not os.path.exists(filebase+'X0'):
            os.mkdir(filebase+'X0')
        da.to_npy_stack(filebase+'X0',X0)
        X0=da.from_npy_stack(filebase+'X0')
        
        includes=np.random.choice(np.arange(N*(N-1)//2),size=D,replace=False)
        ls=np.array([(i+1)*N-i*(i-1)//2-1-2*i for i in range(N)],dtype=int)
        Nt=thetas[0].shape[0]
        
        includes=np.random.choice(np.arange(N*(N-1)//2),size=D,replace=False)
        ls=np.array([(i+1)*N-i*(i-1)//2-1-2*i for i in range(N)],dtype=int)
        
        ii=[]
        jj=[]
        for m in range(D):
            i=np.where(ls>=includes[m])[0][0]
            if i==0:
                j=includes[m]
            else:
                j=includes[m]-ls[i-1]+i
            ii=ii+[i]
            jj=jj+[j]
        X=da.hstack(((np.arange(1,M+1).reshape(1,1,-1)*X0.reshape(list(X0.shape)+[1])).reshape((-1,M*N)),X0[:,np.array(ii)]-X0[:,np.array(jj)]))
        X=da.hstack((da.cos(X),da.sin(X)))
        
        if not os.path.exists(filebase+'X'):
            os.mkdir(filebase+'X')
        da.to_npy_stack(filebase+'X',X)
        X=da.from_npy_stack(filebase+'X')

        if not os.path.exists(filebase+'lengths.npy'):
            np.save(filebase+'lengths.npy',np.array(lengths))
    stop=timeit.default_timer()
    if verbose:
        print('data load runtime:',stop-start,flush=True)
    return X,lengths,filebase,dt

if __name__ == "__main__":

    #Command line arguments
    parser = argparse.ArgumentParser(description='Numerical integration of networks of phase oscillators.')
    parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output.')
    parser.add_argument("--filesuffix", type=str, required=False, dest='filesuffix', default='', help='Suffix string for file output.')
    parser.add_argument("--verbose", type=int, required=False, dest='verbose', default=1, help='Verbose printing.')
    parser.add_argument("--pcatol", type=float, required=False, dest='pcatol', default=1E-7, help='Reconstruction error cutoff for pca.')
    parser.add_argument("--resmax", type=float, required=False, dest='resmax', default=None, help='Maximum residue.')
    parser.add_argument("--minr", type=float, required=False, dest='minr', default=-3, help='Pseudospectra real scale.')
    parser.add_argument("--maxr", type=float, required=False, dest='maxr', default=1, help='Pseudospectra real scale.')
    parser.add_argument("--mini", type=float, required=False, dest='mini', default=-15, help='Pseudospectra imaginary scale.')
    parser.add_argument("--maxi", type=float, required=False, dest='maxi', default=15, help='Pseudospectra imaginary scale.')
    parser.add_argument("--nr", type=int, required=False, dest='nr', default=26, help='Number of real pseudospectra points.')
    parser.add_argument("--ni", type=int, required=False, dest='ni', default=26, help='Number of imaginary pseudospectra points.')
    parser.add_argument("--num_traj", type=int, required=False, dest='num_traj', default=0, help='Number of trajectories.')
    parser.add_argument("--order", type=int, required=False, dest='order', default=1, help='Number of trajectories.')
    parser.add_argument("--seed", type=int, required=False, dest='seed', default=1, help='Random seed for library.')
    parser.add_argument("--M", type=int, required=False, dest='M', default=1, help='Number of angle multiples to include in library.')
    parser.add_argument("--D", type=int, required=False, dest='D', default=0, help='Number of angle pairs to include in library.')
    parser.add_argument("--rank", type=int, required=False, dest='rank', default=None, help='Ritz rank for svd.')
    parser.add_argument("--savepca", type=int, required=False, dest='savepca', default=0, help='Save dense PCA data.')
    parser.add_argument("--runpseudo", type=int, required=False, dest='runpseudo', default=0, help='Run the pseudospectrum calculation.')
    parser.add_argument("--dense_amplitudes", type=int, required=False, dest='dense_amplitudes', default=0, help='Save dense amplitude data.')
    parser.add_argument("--load", type=int, required=False, dest='load', default=0, help='Load data from previous runs.')
    parser.add_argument("--cpus", type=int, required=False, dest='cpus', default=8, help='Number of tasks for dask.')
    parser.add_argument("--mem", type=str, required=False, dest='mem', default='20GB', help='Memory limit for dask.')
    args = parser.parse_args()

    print(*sys.argv,flush=True)
    # Create a LocalCluster with a memory limit of 4GB per worker
    cluster = LocalCluster(n_workers=1, processes=False, memory_limit=args.mem)
    client = Client(cluster)

    filebase0 = args.filebase
    filesuffix = args.filesuffix
    verbose = args.verbose
    pcatol = args.pcatol
    resmax = args.resmax
    minr = args.minr
    maxr = args.maxr
    mini = args.mini
    maxi = args.maxi
    order = args.order
    seed = args.seed
    D = args.D
    M = args.M
    nr = args.nr
    ni = args.ni
    num_traj = args.num_traj
    rank = args.rank
    save = args.savepca
    runpseudo = args.runpseudo
    load = args.load

    start=timeit.default_timer()
    X,lengths,filebase,dt=load_data(filebase0,filesuffix,verbose,num_traj,D,M,load)
    

    Xinds=np.setdiff1d(np.arange(np.sum(lengths)),np.cumsum(lengths)-1)
    Yinds=np.setdiff1d(np.arange(np.sum(lengths)),np.concatenate([[0],np.cumsum(lengths)[:-1]]))
    if args.dense_amplitudes:
        binds=Xinds
    else:
        binds=np.array([0]+list(np.cumsum(lengths)[:-1]))
    if verbose:
        print('shape:', X[Xinds].shape, flush=True)

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
    evals,evecs,res,phis,bs,A=resDMD(u[:,:r],v[:r],s[:r],X,Yinds,binds,filebase,verbose,load=load)


    if runpseudo:
        if nr>1:
            murs=minr+(maxr-minr)*np.arange(nr)/(nr-1)
        else:
            murs=np.array([0])
        muis=mini+(maxi-mini)*np.arange(ni)/(ni-1)

        zs=np.exp((murs[:,np.newaxis]+1j*muis[np.newaxis,:]).ravel()*dt)
        zs_prevs,pseudo,xis,its=resDMDpseudo(u[:,:r],A,zs,evals,evecs,filebase,verbose,load=load)
    stop=timeit.default_timer()
    
    print('runtime:',stop-start)
