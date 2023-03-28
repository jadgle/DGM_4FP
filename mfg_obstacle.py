#!/usr/bin/python
## Author - Matteo Butano 

import numpy as np
import time

# Main Parameters
xi = 0.3
c_s = 0.2
Lx = 6
Ly = 12
Nx = 300
Ny = 600

# Constants
R = 0.37
s = 0.6
m_0 = 2.5
mu = 1
V = -10e2
g = -(2*c_s**2)/m_0
sigma = np.sqrt(2*xi*c_s)
lam = -g*m_0 

#Define grid 
Lx = 6
Ly = 12
Nx = 300
Ny = 600
dx = (2*Lx)/(Nx-1)
dy = (2*Ly)/(Ny-1)
x = np.linspace(-Lx,Lx,Nx)
y = np.linspace(-Ly,Ly,Ny)
X,Y = np.meshgrid(x,y)

def norm(u,v):
    return np.sqrt(u**2+v**2)

# Define cylindrical potentials
C = np.zeros((Ny,Nx))
for i in range(Ny):
    for j in range(Nx):
        if np.sqrt((X[i,j])**2+ Y[i,j]**2) < R:
            C[i,j] = V

def L2_error(p, pn):
    return np.sqrt(np.sum((p - pn)**2)/np.sum(pn**2))       

# Main algorithm 
def jacobi(s,m,p):
    p[:,0]=np.sqrt(m_0)
    p[0,:] = np.sqrt(m_0)
    p[-1,:] = np.sqrt(m_0)
    p[:,-1] = np.sqrt(m_0)
    l2_target = 1e-7
    l2norm = 1
    while l2norm > l2_target:
        pn = p.copy()
        A = -2*mu*sigma**4/(dx*dy) + lam + (g*m[1:-1,1:-1] + C[1:-1,1:-1])
        Q = pn[1:-1,2:] + pn[1:-1, :-2] + pn[2:, 1:-1] + pn[:-2, 1:-1]
        S = (-(0.5*mu*Q*sigma**4)/(dx*dy)+0.5*mu*(sigma**2)*s*(pn[2:,1:-1] - pn[:-2, 1:-1])/dy)
        p[1:-1,1:-1] = S/A
        l2norm = L2_error(p,pn)
    return p

# Self-consistence loop
def simulation(m,alpha,s):
    start = time.time()
    p = np.zeros((Ny,Nx))
    q = np.flip(p,0)
    l2_target = 1e-7
    l2norm = 1
    while l2norm > l2_target:
        mn = m.copy()
        p = jacobi(s,mn,p)
        q = np.flip(p,0)
        m = alpha*p*q + (1-alpha)*mn
        l2norm = L2_error(m,mn)
        print(l2norm)
    end = time.time()
    run = int(end-start)
    print('Program executed in ',run//3600,'h ',run//60,'m ',run%60,'s')
    return m,p,q

def vel(m,p,q):
    phi_grad_x = (p[1:-1,2:]-p[1:-1,:-2])/(2*dx)
    phi_grad_y = (p[2:,1:-1]-p[:-2,1:-1])/(2*dy)
    gamma_grad_x = (q[1:-1,2:]-q[1:-1,:-2])/(2*dx)
    gamma_grad_y = (q[2:,1:-1]-q[:-2,1:-1])/(2*dy)
    v_x = sigma**2*(q[1:-1,1:-1]*phi_grad_x-p[1:-1,1:-1]*gamma_grad_x)/(2*m[1:-1,1:-1])
    v_y = sigma**2*(q[1:-1,1:-1]*phi_grad_y-p[1:-1,1:-1]*gamma_grad_y)/(2*m[1:-1,1:-1])-s
    return np.array([v_x,v_y])

guess = np.full((Ny,Nx),m_0) 
sim = simulation(guess,0.01,s)
m,p,q = sim[0],sim[1],sim[2]
v = vel(m,p,q)
vx,vy = v[0],v[1] 

np.savetxt(r'mass',m)
np.savetxt(r'value',p)
#np.savetxt(r'data/vx_Nx='+str(Nx)+'_Ny='+str(Ny)+'_Lx='+str(Lx)+'_Ly='+str(Ly)+'_xi='+str(round(xi,2))+'_c_s='+str(round(c_s,2))+'.txt',vx)
#np.savetxt(r'data/vy_Nx='+str(Nx)+'_Ny='+str(Ny)+'_Lx='+str(Lx)+'_Ly='+str(Ly)+'_xi='+str(round(xi,2))+'_c_s='+str(round(c_s,2))+'.txt',vy)
