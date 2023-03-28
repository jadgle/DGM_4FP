# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 23:03:53 2023

@author: sarab

checking if the approximated solution are good w.r.t. the loss function
"""

import matplotlib
from dgm_library import sample_room, init_DGM, train, env_initializing, thick_room, sample_room_polar, train_TFC, warmstart_TFC, warmstart_prova
from DGM import * # NOQA
import numpy as np
import keras
import pandas as pd
import tensorflow as tf

V = 10e2

sigma = 0.35
g     = -0.005
mu    = 1
gamma = 0
m0    = 2.5
alpha = 0
l     = -((g*m0)/(1+alpha*m0))+(gamma*mu*sigma**2*np.log(np.sqrt(m0)))
u_b   = -mu*sigma**2*np.log(np.sqrt(m0))
R     = 0.37
#s     =  tf.constant([0, -0.6],  dtype=DTYPE, shape=(1, 2))
v0 = m0**((-mu*sigma**2)/2)

Lx = 6
Ly = 12
Nx = 300
Ny = 600
dx = (2*Lx)/(Nx-1)
dy = (2*Ly)/(Ny-1)
x = np.linspace(-Lx,Lx,Nx)
y = np.linspace(-Ly,Ly,Ny)
X,Y = np.meshgrid(x,y)

# Constants of the agents
Xi = np.sqrt(np.abs((mu*sigma**4)/(2*g*m0)))
Cs = np.sqrt(np.abs((g*m0)/(2*mu)))

u_theta = init_DGM(RNN_layers = 0, FNN_layers=2, nodes_per_layer=100,activation="relu")
m_theta = init_DGM(RNN_layers = 0, FNN_layers=2, nodes_per_layer=100,activation="relu")

x_s = tf.convert_to_tensor(np.reshape(X,(300*600,1)),dtype='float32')
y_s = tf.convert_to_tensor(np.reshape(Y,(300*600,1)),dtype='float32')
X_s = tf.concat([x_s, y_s], axis=1)
X0 = pd.DataFrame(X_s)
F0m = pd.read_csv('mass.csv',dtype='float32',index_col=0,header=0)
F0u = pd.read_csv('value.csv',dtype='float32',index_col=0,header=0)

m_theta, u_theta = warmstart_prova(m_theta,u_theta, X0,F0m,F0u)