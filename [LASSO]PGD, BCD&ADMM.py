import numpy as np
import matplotlib.pyplot as plt

def soft(z,theshold):
    #np.apply_along_axis(lambda x:sign(x)*max(np.absolute(x)-theshold,1,z).reshape(-1,1)
    return np.sign(z)*max(np.abs(z)-theshold,0)

def f(x):
    global A,b,lam
    temp=A@x-b
    nor_x = np.linalg.norm(x, ord=1)
    answer=0.5*(temp.T@temp)+lam*nor_x
    return answer[0,0]

def gradient(x):
    return A.T@((A@x)-b)

def PGD(x0,err=1e-2):
    #Proximal Gradient Descent
    result=[[x0],[f(x0)]]
    x=x0
    for i in range(1,int(np.ceil(1/err))):
        s=10/(10000+i)
        x=np.apply_along_axis(lambda x:soft(x,s*lam),axis=1,arr=x-s*gradient(x)).reshape(-1,1)
        result[0].append(x)
        result[1].append(f(x))
    return result

def BCD(x0,iteration=100):
    #Block Coordinate Descent
    result=[[x0],[f(x0)]]
    x=x0
    list1=list(range(x0.size))
    for k in range(iteration):
        for i in list1:
            list2=[j for j in list1 if j!=i]
            temp=A[:,list2]@x[list2]
            temp=A[:,i].T@(b-temp)
            temp=temp[0]
            x[i]=soft(temp,lam)/(A[:,i].T@A[:,i])
        result[0].append(x)
        result[1].append(f(x))
    return result

def ADMM(x0,z0,mu0,rho,iteration=100):
    #(Alternating Direction Method of Multipliers
    result=[[x0],[f(x0)]]
    x,z,mu=x0,z0,mu0
    for k in range(iteration):
        temp=(A.T@A)+rho
        temp=np.linalg.inv(temp)
        x=temp@((A.T@b)+z-mu)
        z=np.apply_along_axis(lambda x:soft(x,lam/rho),axis=1,arr=x+mu).reshape(-1,1)
        mu=mu+x-z
        result[0].append(x)
        result[1].append(f(x))
    return result

if __name__=='__main__':
    #Create a random dataset
    np.random.seed(2021)
    A = np.random.rand(500, 100)
    x_ = np.zeros([100, 1])
    x_[:5, 0] += np.array([i+1 for i in range(5)])
    b = np.matmul(A, x_) + np.random.randn(500, 1) * 0.1 
    lam = 10

    #Implement algorithms
    PGD_result=PGD(x_,err=1e-3)
    BCD_result=BCD(x_,iteration=1000)
    mu_=z_=x_
    ADMM_result=ADMM(x_,z_,mu_,rho=0.1,iteration=1000)
    ADMM_result1=ADMM(x_,z_,mu_,rho=1,iteration=1000)
    ADMM_result2=ADMM(x_,z_,mu_,rho=10,iteration=1000)

    #Draw pics
    plt.figure(figsize=(12, 10), dpi=80)
    pic1=plt.subplot(2,2,1)
    pic1.plot(PGD_result[1][:100],color="k",linestyle = "--")
    pic1.set_title('Proximal Gradient Descent')

    pic2=plt.subplot(2,2,2)
    pic2.plot(BCD_result[1][:100],color="b",linestyle = "-.")
    pic2.set_title('BCD')

    pic3=plt.subplot(2,2,3)
    pic3.plot(ADMM_result[1],color="g",linestyle = "--",label='ρ=0.1')
    pic3.plot(ADMM_result1[1],color="r",linestyle = "-",label='ρ=1')
    pic3.plot(ADMM_result2[1],color="b",linestyle = "-.",label='ρ=10')
    pic3.legend()
    pic3.set_title('ADMM')

    pic4=plt.subplot(2,2,4)
    pic4.plot(PGD_result[1],color="k",linestyle = "--",label='PGD')
    pic4.plot(BCD_result[1],color="b",linestyle = "-",label='BCD')
    pic4.plot(ADMM_result1[1],color="r",linestyle = "-.",label='ADMM(ρ=1)')
    pic4.legend()
    pic4.set_title('Comparison')

    plt.savefig('LASSO result.jpg')