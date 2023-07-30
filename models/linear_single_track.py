import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

class LinearSingleTrackModel():

    # linearization around beta = 0, yaw_rate = 0
    # longitudal forces Fxf =0 and Fxr = 0
    # linear tire models ... Fyf = cf * alphaf; Fyr = cr * alphar
    # assuming alphaf = deltaf - atan(vyf/vxf); alphar = deltar - atan(vyr/vxr);

    # v is constant
    # x,y,psi are left non-linear for visualization purposes

    def __init__(self,m=1463,Iz=1968,lf=0.971,lr=1.566,rad=0.306,Iw=20):
        self.m = m      # vehicle mass
        self.Iz = Iz    # moment of inertia
        self.lf = lf
        self.lr = lr
        self.rad = rad
        self.Iw = Iw

        self.Caf = 11.9*self.m*9.81*self.lr/(self.lf+self.lr)
        self.Car = 13.6*self.m*9.81*self.lf/(self.lf+self.lr)
        self.Clf = 33*self.m*9.81*self.lr/(self.lf+self.lr)
        self.Clr = 29.6*self.m*9.81*self.lf/(self.lf+self.lr)

    def f(self,t,x,u):

        # t .... times
        # x .... states     [x,y,psi,v,beta,r] (pos x, pos y, heading, sideslip angle, yaw rate)
        # u .... inputs     [df,dr] (front and rear steering angles)

        df = u[0]
        dr = u[1]

        v,beta,r = x[3:6]
        
        if v != 0:

            alphaf = df - beta - self.lf*r/v
            alphar = dr - beta + self.lr*r/v

            Fyf = self.Caf*alphaf
            Fyr = self.Car*alphar

            dv = ((-df+beta)*Fyf + (-dr+beta)*Fyr)/self.m           # here we are missing input torques
            dbeta = (((beta*df+1)*Fyf + (beta*dr+1)*Fyr)/(self.m*v))-r
            dr = (self.lf*Fyf - self.lr*Fyr)/self.Iz

        else:

            dv = 0
            dbeta = 0
            dr = 0

        dv = 0


        psi,v,beta = x[2:5]

        dx = v*np.cos(psi+beta)
        dy = v*np.sin(psi+beta)
        arg_nom = v*np.cos(beta)*(np.tan(u[0])+np.tan(u[1]))
        dpsi = arg_nom/(self.lf+self.lr)

        return np.array([dx,dy,dpsi,dv,dbeta,dr])



if __name__ == "__main__":

    S = LinearSingleTrackModel()
    test_data = np.loadtxt("data/single_track_data.csv",dtype=float,delimiter=",")
    
    max_time = 20000
    start_time = 1500
    t = test_data[start_time:max_time,0]

    x_ode = np.zeros(len(t))
    y_ode = np.zeros(len(t))
    psi_ode = np.zeros(len(t))
    v_ode = np.zeros(len(t))
    beta_ode = np.zeros(len(t))
    r_ode = np.zeros(len(t))

    x_prev = np.array([0,0,0,22.6,0,0])

    v_ode[0] = x_prev[3]

    for i in range(len(t)-1):
        v = spi.solve_ivp(S.f,[t[i],t[i+1]],x_prev,args=(test_data[start_time+i,1:],))
        x_ode[i+1] = v.y[0][-1]
        y_ode[i+1] = v.y[1][-1]
        psi_ode[i+1] = v.y[2][-1]
        v_ode[i+1] = v.y[3][-1]
        beta_ode[i+1] = v.y[4][-1]
        r_ode[i+1] = v.y[5][-1]

        for j in range(6):
            x_prev[j] = v.y[j][-1]
            
    plt.plot(x_ode,y_ode,'-')
    plt.axis('equal')

    plt.show()