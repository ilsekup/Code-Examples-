import fvis3 as FVis
import matplotlib.pyplot as plt
import numpy as np

class convection:
    def __init__(self):
        """
        define variables
        """
        self.G = 6.6742e-11 #N m^2 kg^-2
        self.M_star = 1.989e30 #kg
        self.R_star = 6.96e8 #m
        self.m_u = 1.660539066e-27
        self.kb = 1.38064852e-23 #m^2 kg s^-2 K-1
        self.g = -self.G*self.M_star/self.R_star**2
        self.gamma = 5/3
        self.mu = 0.61
        self.nx = 300
        self.ny = 100
        self.xlen = 12*1e6 #m
        self.ylen = 4*1e6 #m
        self.Dx = self.xlen/(self.nx-1)
        self.Dy = self.ylen/(self.ny-1)
        self.x = np.linspace((self.nx-1)*self.Dx, 0, self.nx)
        self.y = np.linspace((self.ny-1)*self.Dy, 0, self.ny)
        self.rho = np.zeros((self.ny, self.nx))
        self.e = np.zeros((self.ny, self.nx))
        self.u = np.zeros((self.ny, self.nx))
        self.w = np.zeros((self.ny, self.nx))
        self.P = np.zeros((self.ny, self.nx))
        self.T = np.zeros((self.ny, self.nx))


    def initialise(self, Pertubation):
        """
        initialise temperature, pressure, density and internal energy
        """
        nabla = 0.4001
        self.T[-1, :] = 5778 #K
        self.P[-1, :] = 1.8e8 #Pa (kg m^−1 s^−2)
        self.e[-1, :] = self.P[-1, :]/(self.gamma - 1)
        self.rho[-1, :] = self.e[-1, :]*(self.gamma - 1)*self.mu*self.m_u/(self.kb*self.T[-1,:])
        for j in range(self.ny-2, -1, -1):
            self.T[j, :] = self.T[-1, :]- (nabla)*self.g*self.mu*self.m_u/self.kb*self.y[j]
            self.P[j, :] = self.P[-1, :]*(self.T[j, :]/self.T[-1, :])**(1/nabla)
        #If we want a pertubation we have added a gassian function to the intitial temperature
        if Pertubation == True:
            x = np.linspace(0, 1, self.nx)
            y = np.linspace(0, 1, self.ny)
            x0 = int(self.nx/2)
            y0 = int(self.ny/2)
            X, Y = np.meshgrid(x, y)
            self.T +=  10000*np.exp(-0.5*15**2*((X- x[x0])**2 +(Y- y[y0])**2))
        for j in range(self.ny-2, -1, -1):
            self.e[j, :] = self.P[j, :]/(self.gamma - 1)
            self.rho[j, :] = self.e[j, :]*(self.gamma - 1)*self.mu*self.m_u/(self.kb*self.T[j,:])



    def timestep(self, drhodt, dudt, dwdt, dedt):
        """
        calculate timestep
        """
        rel_rho = np.amax(abs(drhodt*1/self.rho))
        rel_x = np.amax(abs(self.u/self.Dx))
        rel_y = np.amax(abs(self.w/self.Dy))
        rel_e = np.amax(abs(dedt*1/self.e))
        #Taking into account the fact that the velocities can be zero in order
        #to not devide by zero.
        uiszero =np.isclose(self.u,0, rtol=1e-5, atol=1e-5)
        wiszero =np.isclose(self.w, 0)
        w_nonzero = abs(self.w) >= 1e-5
        u_nonzero = abs(self.u) >= 1e-5
        if np.all(uiszero) and np.all(wiszero):
            return 0.01
        elif np.all(uiszero) and not np.all(wiszero):
            rel_w = np.amax(abs(dwdt[w_nonzero]*1/(self.w[w_nonzero])))
            delta = max([rel_rho, rel_e, rel_w, rel_x, rel_y])
        elif np.all(wiszero) and not np.all(uiszero):
            rel_u = np.amax(abs(dudt[u_nonzero]*1/(self.u[u_nonzero])))
            delta = max([rel_rho, rel_e, rel_u, rel_x, rel_y])
        else:
            rel_w = np.amax(abs(dwdt[w_nonzero]*1/(self.w[w_nonzero])))
            rel_u = np.amax(abs(dudt[u_nonzero]*1/(self.u[u_nonzero])))
            delta = max([rel_rho, rel_e, rel_u, rel_w, rel_x, rel_y])
        p =0.1
        dt = p/delta
        #en extra if-test just so the timestep will not be too small for animation
        if dt < 1e-2:
            dt = 1e-2
        return dt


    def boundary_conditions(self):
        """
        boundary conditions for energy, density and velocity
        """
        C = self.g*self.mu*self.m_u/self.kb
        self.w[0, :] = self.w[self.ny-1, :] = 0
        self.u[0,:] = (-self.u[0+2, :] + 4*self.u[0+1, :])/3
        self.u[self.ny-1,:] = (-self.u[self.ny-3, :] + 4*self.u[self.ny-2, :])/3
        self.e[0, :] = (-self.e[0+2, :]+ 4*self.e[0+1, :])/(3 + C/self.T[0, :]*2*self.Dy)
        self.e[self.ny-1, :]= (-self.e[self.ny-3, :]+ 4*self.e[self.ny-2, :])/(3 - C/self.T[self.ny-1, :]*2*self.Dy)
        self.rho[0,:] = (self.gamma -1)*self.mu*self.m_u/self.kb*self.e[0,:]/self.T[0, :]
        self.rho[self.ny-1,:] = (self.gamma -1)*self.mu*self.m_u/self.kb*self.e[self.ny-1,:]/self.T[self.ny-1, :]


    def central_x(self,func):
        """
        central difference scheme in x-direction
        """
        dfuncdx = (np.roll(func, -1, 1) - np.roll(func, 1, 1))/(2*self.Dx)
        return dfuncdx

    def central_y(self,func):
        """
        central difference scheme in y-direction
        """
        dfuncdy = (np.roll(func, -1, 0) - np.roll(func, 1, 0))/(2*self.Dy)
        return dfuncdy

    def upwind_x(self,func,u):
        """
        upwind difference scheme in x-direction
        """
        dfuncdx = np.zeros((self.ny, self.nx))
        greater_roll = np.roll(func, 1, 1)
        less_roll = np.roll(func, -1, 1)
        g_y, g_x = np.where(u >= 0)
        l_y, l_x = np.where(u < 0)
        dfuncdx[g_y, g_x] = (func[g_y, g_x]- greater_roll[g_y, g_x])/self.Dx
        dfuncdx[l_y, l_x] = (less_roll[l_y, l_x] - func[l_y, l_x])/self.Dx
        return dfuncdx

    def upwind_y(self,func,u):
        """
        upwind difference scheme in y-direction
        """
        dfuncdy =np.zeros((self.ny, self.nx))
        greater_roll = np.roll(func, 1, 0)
        less_roll = np.roll(func, -1, 0)
        g_y, g_x = np.where(u >= 0)
        l_y, l_x = np.where(u < 0)
        dfuncdy[g_y, g_x] = (func[g_y, g_x] - greater_roll[g_y, g_x])/self.Dy
        dfuncdy[l_y, l_x] = (func[l_y, l_x] - less_roll[l_y, l_x])/self.Dy
        return dfuncdy

    def hydro_solver(self):
        """
        hydrodynamic equations solver
        """
        drhodt = -self.rho*(self.central_x(self.u)+ self.central_y(self.w)) \
            - self.u*self.upwind_x(self.rho, self.u) - self.w*self.upwind_y(self.rho, self.w)

        drhoudt = -self.rho*self.u*(self.upwind_x(self.u, self.u) \
            + self.upwind_y(self.w, self.u)) - self.u*self.upwind_x(self.rho*self.u, self.u) \
            - self.w*self.upwind_y(self.rho*self.u, self.w) - self.central_x(self.P)

        drhowdt = -self.rho*self.w*(self.upwind_x(self.u, self.w) \
            + self.upwind_y(self.w, self.w)) - self.w*self.upwind_y(self.rho*self.w, self.w) \
            - self.u*self.upwind_x(self.rho*self.w, self.u) - self.central_y(self.P) \
            + self.rho*self.g

        dedt = -self.e*self.central_x(self.u) - self.u*self.upwind_x(self.e, self.u) \
            -self.e*self.central_y(self.w) - self.w*self.upwind_y(self.e, self.w) \
                - self.P*(self.central_x(self.u) + self.central_y(self.w))
        #Find derivative of u and w with time to use for timestep
        dudt = self.u/self.rho*drhodt - 1/self.rho*drhoudt
        dwdt = self.w/self.rho*drhodt - 1/self.rho*drhowdt
        #update all variable matrices by using the timestep dt
        dt = self.timestep(drhodt, dudt, dwdt, dedt)
        urho = self.rho*self.u
        wrho = self.rho*self.w
        self.rho[:] = self.rho + drhodt*dt
        self.u[:] = (urho + drhoudt*dt)/self.rho
        self.w[:] = (wrho + drhowdt*dt)/self.rho
        self.e[:] = self.e + dedt*dt
        self.boundary_conditions()
        self.P[:] = (self.gamma -1 )*self.e
        self.T[:] = self.e*(self.gamma - 1)*(self.mu*self.m_u)/(self.rho*self.kb)
        return dt

solver = convection()
#pertubation = False for hydrostatic equilibrium
solver.initialise(Pertubation=True)
dt = solver.hydro_solver()
vis =FVis.FluidVisualiser()
vis.save_data(200, solver.hydro_solver, rho=solver.rho, u = solver.u, w=solver.w, e=solver.e, T=solver.T, P=solver.P, sim_fps=1.0)
vis.animate_2D('T', save=True)
vis.delete_current_data()