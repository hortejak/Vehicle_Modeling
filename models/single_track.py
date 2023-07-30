import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt


class SingleTrackModel():

    def __init__(self,m=1463,Iz=1968,lf=0.971,lr=1.566,rad=0.306,Iw=20):
        self.m = m      # vehicle mass
        self.Iz = Iz    # moment of inertia
        self.lf = lf
        self.lr = lr
        self.rad = rad
        self.Iw = Iw

        self.Fxf_max = 1*self.m*9.81*self.lr/(self.lf+self.lr)
        self.Fyf_max = 0.95*self.m*9.81*self.lr/(self.lf+self.lr)
        self.Fxr_max = 1.12*self.m*9.81*self.lf/(self.lf+self.lr)
        self.Fyr_max = 1.05*self.m*9.81*self.lf/(self.lf+self.lr)

        self.Caf = 11.9*self.m*9.81*self.lr/(self.lf+self.lr)
        self.Car = 13.6*self.m*9.81*self.lf/(self.lf+self.lr)
        self.Clf = 33*self.m*9.81*self.lr/(self.lf+self.lr)
        self.Clr = 29.6*self.m*9.81*self.lf/(self.lf+self.lr)


    def f(self,t,x,u):

        # t .... times
        # x .... states     [x,y,psi,v,beta,r,wf,wr] (pos x, pos y, heading, sideslip angle, yaw rate, front angular velocity, rear angular velocity)
        # u .... inputs     [df,dr,tdf,tbf,tdr,tbr] (front and rear steering angles, driving and braking torques on front and rear axle)

        af,ar,vxf,vxr = self.wheel_kinematics(x,u)
        lambdaf,lambdar = self.slip_ratios(vxf,vxr,x)
        Fxf,Fyf,Fxr,Fyr = self.tire_models(af,ar,lambdaf,lambdar)
        Fx,Fy,Mz = self.steering_angle_projection(Fxf,Fyf,Fxr,Fyr,u)
        dbeta,dv,dr = self.rigid_body_dynamics(Fx,Fy,Mz,x)
        dwf,dwr = self.wheel_dynamics(x,u,Fxf,Fxr)

        psi,v,beta = x[2:5]

        dx = v*np.cos(psi+beta)
        dy = v*np.sin(psi+beta)
        arg_nom = v*np.cos(beta)*(np.tan(u[0])+np.tan(u[1]))
        dpsi = arg_nom/(self.lf+self.lr)

        return np.array([dx,dy,dpsi,dv,dbeta,dr,dwf,dwr])


    def rigid_body_dynamics(self,Fx,Fy,Mz,x):

        # represents the vehicle chassis

        v,beta,r = x[3:6]

        D = np.zeros(3)     # dbeta, dv, dr

        if v != 0:

            A = np.diag([ 1/(self.m*v), 1/self.m, 1/self.Iz])
            c, s = np.cos(beta), np.sin(beta)
            rot = np.array(((-s,c,0),(c,s,0),(0,0,1)))

            F = np.array([[Fx],[Fy],[Mz]])
            R = np.array([[r],[0],[0]])
        
            D = (A @ rot @ F) - R

            D = np.transpose(D)[0]
            
        return D
    
    def steering_angle_projection(self,Fxf,Fyf,Fxr,Fyr,u):

        # computes longitudal and lateral forces acting on the rigin body as well as the angular momentum

        # returns F = [Fx,Fy.Mz]

        df,dr = u[0:2]

        F_pre = np.array([[Fxf],[Fyf],[Fxr],[Fyr]])
        cf,cr,sf,sr = np.cos(df), np.cos(dr), np.sin(df), np.sin(dr)
        A = np.array([[cf,-sf,cr,-sr],[sf,cf,sr,cr],[self.lf*sf,self.lf*cf,-self.lr*sr,-self.lr*cr]])
        F = A @ F_pre

        return np.transpose(F)[0]
    

    def tire_models(self,af,ar,lambdaf,lambdar):

        # calculates tire forces from slip variables

        # two line model, can be improved by simplified Pacejka

        F = np.zeros(4)

        m = np.min((self.Fxf_max,self.Clf*lambdaf))
        F[0] = np.max((m,-self.Fxf_max))

        m = np.min((self.Fyf_max,self.Caf*af))
        F[1] = np.max((m,-self.Fyf_max))

        m = np.min((self.Fxr_max,self.Clr*lambdar))
        F[2] = np.max((m,-self.Fxr_max))

        m = np.min((self.Fyr_max,self.Car*ar))
        F[3] = np.max((m,-self.Fyr_max))

        return F
    
    def wheel_kinematics(self,x,u):

        # calculates sideslip angles and wheel speeds

        df,dr = u[0:2]
        v,beta,r = x[3:6]

        V = np.zeros(4)

        Af = np.array([[np.cos(df),np.sin(df)],[-np.sin(df),np.cos(df)]])
        Ar = np.array([[np.cos(dr),np.sin(dr)],[-np.sin(dr),np.cos(dr)]])

        vx = v*np.cos(beta)
        vy = v*np.sin(beta)

        Vf_pre = np.array([[vx],[vy+self.lf*r]])
        Vr_pre = np.array([[vx],[vy-self.lr*r]])

        Vf = np.transpose(Af @ Vf_pre)[0]
        Vr = np.transpose(Ar @ Vr_pre)[0]

        V[0] = -np.arctan2(Vf[1],np.abs(Vf[0]))     # alphaF --> angle between the wheel's F and v
        V[1] = -np.arctan2(Vr[1],np.abs(Vr[0]))
        V[2] = Vf[0]
        V[3] = Vr[0]

        return V
    
    def wheel_dynamics(self,x,u,Fxf,Fxr):

        # calculates wheel rotational dynamics

        tdf,tbf,tdr,tbr = u[2:6]
        wf,wr = x[6:8]

        dwf = (tdf - self.rad*Fxf - np.sign(wf)*tbf)/self.Iw
        dwr = (tdr - self.rad*Fxr - np.sign(wr)*tbr)/self.Iw

        return dwf,dwr
    
    def slip_ratios(self,vxf,vxr,x):

        # computes how each wheel spins

        wf,wr = x[6:8]
        lambdaf = (wf*self.rad-vxf)/np.max((np.abs(wf*self.rad),np.abs(vxf)))
        lambdar = (wr*self.rad-vxr)/np.max((np.abs(wr*self.rad),np.abs(vxr)))

        return lambdaf,lambdar

if __name__ == "__main__":

    S = SingleTrackModel()
    test_data = np.loadtxt("data/single_track_data.csv",dtype=float,delimiter=",")

    start_time = 1500    
    max_time = 20000
    
    t = test_data[start_time:max_time,0]

    x_ode = np.zeros((8,len(t)))
    x_prev = np.array([0,0,0,1,0,0,1/S.rad,1/S.rad])

    x_ode[:,0] = x_prev

    for i in range(len(t)-1):
        v = spi.solve_ivp(S.f,[t[i],t[i+1]],x_prev,args=(test_data[start_time+i,1:],))
        x_ode[:,i+1] = v.y[:,-1]
        x_prev = v.y[:,-1]

    plt.plot(x_ode[0,:],x_ode[1,:],'-')
    plt.axis('equal')

    plt.show()

    

