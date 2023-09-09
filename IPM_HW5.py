import numpy as np

def df(x):
    temp=np.zeros((x0.size,1))
    temp[0]=2*x[0]-2*x[1]-2
    temp[1]=-2*x[0]+4*x[1]-6
    return temp

def fun(x,lamb,nu,A,b,mu):
    #Build F
    F=np.zeros((2*x.size+b.size,1))
    X=np.diag(x.ravel())
    Lamb=np.diag(lamb.ravel())
    ones=np.ones((x0.size,1))
    F[:x.size]=df(x)+A.T@nu-lamb
    F[x.size:x0.size+b.size]=A@x-b
    F[x.size+b.size:]=X@Lamb@ones-mu*ones
    return F

def IPM(x0,A,b,mu=1,error=1e-2):
    temp=np.zeros((2*x0.size+b.size,1))#temp=[x;lambda;nu].T, lambda0=1,nu0=0
    x=temp[:x0.size]=x0
    lamb=temp[x0.size:2*x0.size]=mu/x0
    nu=temp[2*x0.size:]

    #Componeents of dF
    minus_eye=-np.eye(x0.size)
    Hessian=np.zeros((x0.size,x0.size))
    Hessian[0,:2]=[2,-2]
    Hessian[1,:2]=[-2,4]
    zero1=np.zeros(A.shape)
    zero2=np.zeros((A.shape[0],A.shape[0]))
    zero3=np.zeros(A.shape[::-1])
    trans_A=A.T

    while True:
        X=np.diag(x.ravel())
        Lamb=np.diag(lamb.ravel())
        F=fun(x,lamb,nu,A,b,mu)
        if np.all(np.absolute(F)<=error):
            return x
        else:
            dF=np.bmat("Hessian,minus_eye,trans_A;A,zero1,zero2;Lamb,X,zero3")
            step=-(np.linalg.inv(dF)@F).A
            norm=np.linalg.norm(F)
            
            #Backward Line Search
            alpha=0.49
            beta=0.8
            while True:
                x1=x+alpha*step[:x0.size]
                lamb1=lamb+alpha*step[x0.size:2*x0.size]
                nu1=nu+alpha*step[2*x0.size:]
                F1=fun(x1,lamb1,nu1,A,b,mu)
                norm1=np.linalg.norm(F1)
                if norm1<=0.99*norm and np.all(x1>=0):
                    (x,lamb,nu)=(x1,lamb1,nu1)
                    mu=0.1/x.size*(x.T@lamb)
                    break
                else:
                    alpha*=beta

if __name__=='__main__':
    A=np.array([[0.5,0.5,1,0],[-1,2,0,1]])
    b=np.array([1,2]).reshape(2,1)
    x0=np.array([1,1,1,1]).reshape(4,1)
    x=IPM(x0,A,b)