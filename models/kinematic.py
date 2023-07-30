import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

class KinematicModel():

    def __init__(self,lf=0.971,lr=1.566) -> None:
        self.lf = lf        # distance between the CG and the front axis
        self.lr = lr        # distance between the CG and the rear axis

    def sideslip_angle(self,df,dr):

        argument = (self.lf*np.tan(dr) + self.lr*np.tan(df)) / (self.lf + self.lr)
        return np.arctan(argument)
    
    def velocity(self,vf,vr,df,dr,beta):

        argument_nom = vf*np.cos(df) + vr*np.cos(dr)
        return argument_nom/(2*np.cos(beta))
    
    def f(self,t,x,u):

        # t .... times
        # x .... states     [x,y,psi]
        # u .... inputs     [vf,vr,df,dr]

        vf,vr,df,dr = u[0:4]
        psi = x[2]

        beta = self.sideslip_angle(df,dr)
        v = self.velocity(vf,vr,df,dr,beta)

        dx = v*np.cos(psi+beta)
        dy = v*np.sin(psi+beta) 
        arg_nom = v*np.cos(beta)*(np.tan(df)+np.tan(dr))
        dpsi = arg_nom/(self.lf+self.lr)

        return np.array([dx,dy,dpsi])
    

if __name__=="__main__":

    # Example of usage

    K = KinematicModel()
    test_data = np.loadtxt("data/kinematic_data.csv",dtype=float,delimiter=",")

    start_time = 0
    max_time = 20000

    t = test_data[start_time:max_time,0]

    x_ode = np.zeros((3,len(t)))
    x_prev = np.zeros(3)

    for i in range(len(t)-1):
        v = spi.solve_ivp(K.f,[t[i],t[i+1]],x_prev,args=(test_data[i,1:],))
        x_ode[:,i+1] = v.y[:,-1]
        x_prev = v.y[:,-1]

    plt.plot(x_ode[0,:],x_ode[1,:],'-')
    plt.axis('equal')

    plt.show()
