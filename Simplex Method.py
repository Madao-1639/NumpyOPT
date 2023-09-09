import numpy as np

def simplex_method(x0,c0,A0,b,base):
    x=x0
    while True:
        not_base=[i for i in range(c0.size) if i not in base]
        order=base+not_base
        B=A0[:,base]
        N=A0[:,not_base]
        c=c0[order]

        zeros=np.zeros((N.shape[1],B.shape[1]))
        eye=np.eye(N.shape[1])
        inv_B=np.linalg.inv(B)
        temp=-inv_B@N
        inv_M=np.bmat("inv_B,temp;zeros,eye")

        for i in range(len(base),c0.size):
            dq=inv_M[:,i]
            if c.T@dq<0:
                if dq[dq<0].size==0:
                    return None#the LP is unbounded 
                else:
                    list1=[]
                    for j in range(c0.size):
                        if dq[j]<0:
                            list1.append(-x[order[j],0]/dq[j,0])
                        else:
                            list1.append(None)
                    list2=[x for x in list1 if x !=None]
                    step=min(list2)
                    in_base=order[i]
                    out_base=order[list1.index(step)]
                    break
            elif i == c0.size-1:
                return x

        base.remove(out_base)
        base.append(in_base)
        x[order]=x[order]+step*dq


if __name__=='__main__':
    c=np.array([-5,-1,0,0]).reshape(4,1)
    A=np.array([[1,1,1,0],[2,0.5,0,1]])
    b=np.array([5,8]).reshape(2,1)
    x0=np.array([0,0,5,8]).reshape(4,1)
    x=simplex_method(x0,c,A,b,[2,3])