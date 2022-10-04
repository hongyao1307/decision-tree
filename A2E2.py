import numpy as np
from sympy import*
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

x=np.array([[2,1],[3,1],[1,2],[3,2]])
y=np.array([[1],[1],[-1],[-1]])

def y_hat(w,x,b):
    wt=np.transpose(w)
    yhat=np.matmul(wt,x)+b
    #print('yhat=',yhat)
    return yhat[0][0]

def al_SGD(x,y,max_pass,step,C):
    b=0
    w=np.zeros((len(x[0]),1))
    for i in range(max_pass):
        for j in range(len(x)):
            x1=x[j][:, None]
            y1=y[j][0]
            yh=y_hat(w,x1,b)
            #print(yh)
            if y1*yh<=1:
                dw=-2*C*y1*x1*(1-y1*yh)
                db=-2*C*y1*(1-y1*yh)
                w=(w-step*dw)*(1+step)
                b=b-step*db
            w=w/(1+step)
        w=w-step*w
    return w,b

def main():
    max_pass=1000
    step=0.001
    (w,b)=al_SGD(x,y,max_pass,step,120)
    print('w=',w)
    print('b=',b)

if __name__ == '__main__':
    main()
