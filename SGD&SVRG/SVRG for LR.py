from libsvmdata import fetch_libsvm
import numpy as np
import matplotlib.pyplot as plt

def dh(a,b,x):
    #Derivative of Loss Function
    return (-b*a.T/(1+np.exp(b*a@x))).A

def SGD(A,b,x0,step=1e-3,freq=100):
    result=[f(A,b,x0)]
    global lamb
    x=x0
    for k in range(100):
        avg_df=0
        for i in range(A.shape[0]):
            avg_df+=dh(A[i],b[i],x)
        avg_df=avg_df/A.shape[0]+2*lamb*x
        x_=x
        sum_x_=x_
        for i in range(freq):
            it=np.random.randint(0,A.shape[0])
            x_=x_-step*(dh(A[it],b[it],x_)-dh(A[it],b[it],x)+2*lamb*(x_-x)+avg_df)
            sum_x_+=x_
        x=sum_x_/freq#Average option|   Last option:x=x_    |   Random option:x=(random)x_
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
    plt.savefig('3.jpg')