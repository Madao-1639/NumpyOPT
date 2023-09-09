from libsvmdata import fetch_libsvm
import numpy as np
import matplotlib.pyplot as plt

def dh(a,b,x):
    #Derivative of Loss Function
    return (-b*a.T/(1+np.exp(b*a@x))).A

def SGD(A,b,x0,step=1e-3,batchsize=1):
    result=[f(A,b,x0)]
    global lamb
    sum_dhi=np.zeros((x0.size,1))
    x=x0
    for k in range(10000):
        batch=np.random.randint(0,b.size,size = batchsize)
        for i in batch:
            sum_dhi+=dh(A[i],b[i],x)
        x=x-step*(sum_dhi/batchsize+2*lamb*x)
        sum_dhi=0
        result.append(f(A,b,x))
    return result

def f(A,b,x):
    b=b.reshape(A.shape[0],1)
    temp=np.exp(-b*(A@x))
    y=np.log(1+temp).sum(axis=0)
    global lamb
    y=y/A.shape[0]+lamb*(x.T@x)
    return y[0,0]

if __name__=='__main__':
    np.random.seed(42)
    data=fetch_libsvm("a9a")
    A=data[0]
    b=data[1]
    x0=np.zeros((A.shape[1],1))
    lamb=0.1/x0.size
    result=SGD(A,b,x0)

    plt.figure()
    plt.plot(result)
    plt.savefig('1.jpg')