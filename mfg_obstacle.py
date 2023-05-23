#!/usr/bin/python

import numpy as np
import time
import json

# Main Parameters

with open('config.json') as f:
    var = json.loads(f.read())
     
xi = var['mfg_params']['xi']
c_s = var['mfg_params']['c_s']
Lx = var['room']['lx']
Ly = var['room']['ly']
Nx = var['room']['nx']
Ny = var['room']['ny']

# Constants
R = var['room']['R']
s = var['room']['s']
m_0 = var['room']['m_0']
mu = var['mfg_params']['mu']
V = var['mfg_params']['V']
g = -(2*c_s**2)/m_0
sigma = np.sqrt(2*xi*c_s)
lam = -g*m_0 

#Define grid 
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
        print('Error = {:10.3e}'.format(l2norm))
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

np.savetxt('IC/phi.txt',p.reshape((Nx*Ny,1)))
np.savetxt('IC/gamma.txt',q.reshape((Nx*Ny,1)))
np.savetxt('IC/points.txt',np.concatenate([X.reshape((Nx*Ny,1)),Y.reshape((Nx*Ny,1))],axis = 1))
