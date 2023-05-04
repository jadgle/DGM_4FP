##################################################################################################################
#########################                                                            #############################       
#########################               Main - DGM for pedestrian MFG                #############################
#########################                                                            #############################
##################################################################################################################

import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
#from dgm_library import warmstart, sample_room, init_DGM, train, env_initializing, thick_room, warmstart_room, train_room
#from dgm_library_wTFC import env_initializing, sample_room_polar, train_TFC, warmstart_TFC
from dgm_library_ergodic import env_initializing, init_DGM, train, warmstart_sol

from DGM import * # NOQA
import numpy as np
import pandas as pd
from tensorflow.python.ops.numpy_ops import np_config

env_initializing()

#######################################################################################################################
DTYPE = 'float32'

Phi_theta = init_DGM(RNN_layers = 0, FNN_layers=1, nodes_per_layer=100,activation="tanh")
Gamma_theta = init_DGM(RNN_layers = 0, FNN_layers=1, nodes_per_layer=100,activation="tanh")

phi = pd.read_csv('phi.txt', header = None)
gamma = pd.read_csv('gamma.txt', header = None)

Phi_theta,Gamma_theta, X0  = warmstart_sol(Phi_theta,Gamma_theta,phi,gamma,2)



np_config.enable_numpy_behavior()
Phi = Phi_theta(X0)
Gamma = Gamma_theta(X0)
m = Phi*Gamma
x = X0.iloc[:,0]
y = X0.iloc[:,1]

usual = matplotlib.cm.hot_r(np.arange(256))
saturate = np.ones((int(256 / 20), 4))
for i in range(3):
    saturate[:, i] = np.linspace(usual[-1, i], 0, saturate.shape[0])
cmap1 = np.vstack((usual, saturate))
cmap1 = matplotlib.colors.ListedColormap(cmap1, name='myColorMap', N=cmap1.shape[0])

fig = plt.figure(figsize=(7,6))
plt.scatter(x, y, c=m, cmap=cmap1, marker='.', alpha=0.3)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.colorbar()
plt.show()

Phi_theta,Gamma_theta,X_b, X_s, X_c = train(Phi_theta,Gamma_theta,2)
###
# X0 = pd.concat([X_b,X_s,X_c], 0)
# x = X0.iloc[:,0]
# y = X0.iloc[:,1]
# m = Gamma_theta(X0)*Phi_theta(X0)


# usual = matplotlib.cm.hot_r(np.arange(256))
# saturate = np.ones((int(256 / 20), 4))
# for i in range(3):
#     saturate[:, i] = np.linspace(usual[-1, i], 0, saturate.shape[0])
# cmap1 = np.vstack((usual, saturate))
# cmap1 = matplotlib.colors.ListedColormap(cmap1, name='myColorMap', N=cmap1.shape[0])

# fig = plt.figure(figsize=(7,6))
# plt.scatter(x, y, c=m, cmap=cmap1, marker='.', alpha=0.3)
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.colorbar()
# plt.show()


# ############################################################################################
# N_s = 500
# x = np.linspace(-6, 6, N_s)
# y = np.linspace(-12, 12, N_s)
# X, Y = np.meshgrid(x, y)
# x_ = X.reshape((N_s**2,1))
# y_ = Y.reshape((N_s**2,1))
# X0 = pd.DataFrame(np.concatenate([x_,y_],1))    
# X0 = X0.astype(dtype = DTYPE)
# usual = matplotlib.cm.hot_r(np.arange(256))
# saturate = np.ones((int(256 / 20), 4))
# for i in range(3):
#     saturate[:, i] = np.linspace(usual[-1, i], 0, saturate.shape[0])
# cmap1 = np.vstack((usual, saturate))
# cmap1 = matplotlib.colors.ListedColormap(cmap1, name='myColorMap', N=cmap1.shape[0])

# fig = plt.figure(figsize=(7,6))
# plt.scatter(x, y, c=m, cmap=cmap1, marker='.', alpha=0.3)

# np_config.enable_numpy_behavior()
# x_ = x_.reshape(500,500)
# y_ = y_.reshape(500,500)
# Phi = Phi_theta(X0)
# Gamma = Gamma_theta(X0)
# m = Phi*Gamma
# np_config.enable_numpy_behavior()
# m = m.reshape(500,500)
# plt.contourf(x_, y_, m, 100, cmap=cmap1)
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.colorbar()
# plt.show()