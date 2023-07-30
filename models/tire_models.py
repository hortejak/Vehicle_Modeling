import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm


class SimplifiedPacejkaModel():

    def __init__(self,Dx=1,Bx=11.2758,Cx=1.2355,Ex=-0.9193,Dy=None,By=None,Cy=None,Ey=None):

        self.Dx = Dx
        self.Bx = Bx
        self.Cx = Cx
        self.Ex = Ex

        self.Dy = Dx if Dy is None else Dy
        self.By = Bx if By is None else By
        self.Cy = Cx if Cy is None else Cy        
        self.Ey = Ex if Ey is None else Ey

    def calculate_longitudal_force(self,Fz,slip):

        F = self.Dx*Fz*np.sin(self.Cx*np.arctan(self.Bx*slip-self.Ex*(self.Bx*slip-np.arctan(self.Bx*slip))))
        return F
    
    def calculate_lateral_force(self,Fz,slip):

        F = self.Dy*Fz*np.sin(self.Cy*np.arctan(self.By*slip-self.Ey*(self.By*slip-np.arctan(self.By*slip))))
        return F
    
    def calculate_traction_elypse(self,Fz,slip_alpha,slip_lambda):

        Fx_raw = self.calculate_longitudal_force(Fz,slip_lambda)
        Fy_raw = self.calculate_lateral_force(Fz,slip_alpha)

        if slip_alpha == 0 and slip_lambda == 0:
            return [0,0]

        beta_star = np.arccos(np.abs(slip_lambda)/np.sqrt(np.power(slip_lambda,2)+np.power(np.sin(slip_alpha),2)))

        mu_x_tmp = Fx_raw/Fz
        mu_y_tmp = Fy_raw/Fz

        if mu_x_tmp == 0:
            Fx = 0
        else:
            mu_x = 1/(np.sqrt(np.power(1/mu_x_tmp,2)+np.power(np.tan(beta_star)/self.Dy,2)))
            Fx = Fx_raw*np.abs(mu_x/mu_x_tmp)

        if mu_y_tmp == 0:
            Fy = 0
        else:
            mu_y = 1/(np.sqrt(np.power(1/self.Dx,2)+np.power(np.tan(beta_star)/mu_y_tmp,2)))
            Fy = Fy_raw*np.abs(mu_y/mu_y_tmp)       
        

        F = np.array([Fx,Fy])

        return F


if __name__ == "__main__":

    s = SimplifiedPacejkaModel(Dx=-1.05,Bx=10.8,Cx=1.2,Ex=-1,Dy=-1,By=30,Cy=-1.1,Ey=-0.1)

    slip = np.linspace(-0.6,0.6,1000)

    f = s.calculate_longitudal_force(8600,slip)
    
    plt.plot(slip,f,'-')
    plt.show()

    slip_alpha = np.linspace(-np.pi/2,np.pi/2,63)
    slip_lambda = np.linspace(-1,1,41)

    X,Y = np.meshgrid(slip_lambda,slip_alpha)

    Fx = np.zeros(X.shape)
    Fy = np.zeros(Y.shape)

    Fz = 1

    for i in range(len(slip_alpha)):
        for j in range(len(slip_lambda)):
            Fx[i,j],Fy[i,j] = s.calculate_traction_elypse(Fz,X[i,j],Y[i,j])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Y, Fx, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    