##################################################################################################################     
#########################                                                            #############################  
#########################           Core methods - DGM for pedestrian MFG            #############################
#########################                                                            #############################
##################################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import numpy as np
import random
import os

from DGM import  DGMNet

########################################################################################################################
DTYPE = 'float64'

# Problem parameters
sigma = 0.35
g     = -0.005
mu    = 1
gamma = 0
m0    = 2.5
alpha = 0
R     = 0.37
V_const = 10e3

# room limits (changed on march 2023 to be bigger)
xmin = -6.
xmax = 6.
ymin = -12.
ymax = 12.
initial_tot_mass = tf.constant(2.5 * (xmax-xmin) * (ymax-ymin),dtype=DTYPE)

l     = -((g*m0)/(1+alpha*m0))+(gamma*mu*sigma**2*np.log(np.sqrt(m0)))
u_b   = -mu*sigma**2*np.log(np.sqrt(m0))
s     =  tf.constant([0, -0.6],  dtype=DTYPE, shape=(1, 2))
v0    = m0**((-mu*sigma**2)/2)

# Constants of the agents
Xi = np.sqrt(np.abs((mu*sigma**4)/(2*g*m0)))
Cs = np.sqrt(np.abs((g*m0)/(2*mu)))

# Radius of the circle for the solution of the mfg in polar coordinates (used for TFC)
B = 10

# Number of sampled points for the euclidean mfg 
N_0 = 200
N_b = 200
N_s1 = 1000
N_s2 = 5000
N_s = N_s1 + N_s2
########################################################################################################################
# This is the environment initializer, used for reproducibility as well
def env_initializing():
    # Seed value
    seed_value = 0
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    from keras import backend as K
    # jobs = 256 # it means number of cores
    # session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=jobs,
    #                           inter_op_parallelism_threads=jobs,
    #                           allow_soft_placement=True,
    #                           device_count={'CPU': jobs})
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=0,
                               inter_op_parallelism_threads=0)

    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    K.set_session(sess)
    # Set data type
    tf.keras.backend.set_floatx(DTYPE)
    return


########################################################################################################################

# Potential V entering the HJB equation. The value V_const is a parameter defined above
def V(m, x):
    U0 = tf.zeros(shape = (x.shape[0],1),dtype=DTYPE)
    U0 = tf.unstack(U0)
    for i in range(x.shape[0]):
        if tf.less_equal(tf.norm(x[i],'euclidean'),R): # for points in the cilinder
            U0[i] = tf.constant(V_const,dtype=DTYPE)   # we have higher cost
        else:
            U0[i] = tf.constant(0,dtype=DTYPE)
    U0 = tf.stack(U0)
    return g * m + U0 # formula for the potential from reference paper



########################################################################################################################

def sample_room(verbose):
    
    # number of points
    N_0 = 200
    N_b = 500
    N_s = 1000
    
    # room limits (changed on march 2023 to be bigger)
    xmin = -6.
    xmax = 6.
    ymin = -12.
    ymax = 12.

    # Lower bounds
    lb = tf.constant([xmin, ymin], dtype=DTYPE)
    # Upper bounds
    ub = tf.constant([xmax, ymax], dtype=DTYPE)

    # Draw uniform sample points for data in the domain
    x_s = tf.random.uniform((N_s, 1), lb[0], ub[0], dtype=DTYPE)
    y_s = tf.random.uniform((N_s, 1), lb[1], ub[1], dtype=DTYPE)
    X_s = tf.concat([x_s, y_s], axis=1)

    # UPDATE 28 MARCH : we don't erase points inside the cilinder
    # erasing points inside the cilinder 
    ind = tf.where(tf.norm(X_s, axis=1) > R)
    X_s = tf.gather(X_s, ind)
    N_s = X_s.shape[0]
    X_s = tf.squeeze(X_s)
    x_s = X_s.numpy()[:, 0]
    y_s = X_s.numpy()[:, 1]
    
    m_s = m0 * (np.sqrt(x_s ** 2 + y_s ** 2) > R)
    
    # what if we impose the initial condition on the mass as a warm start?
    #m_s = np.ones((N_s, 1)) * m0  # constant mass everywhere
    #m_s = np.multiply(m_s,(x_s ** 2 + y_s ** 2 >= R ** 2)) # no mass where the cilinder is
    # we impose some guassian warm start for the value function
    u_s = u_b + np.exp(-(x_s ** 2 + y_s ** 2) / 2 * 0.1 ** 2) / (2 * np.pi * 0.1 ** 2)


    # Boundary data (square - outside walls)
    x_b1 = lb[0] + (ub[0] - lb[0]) * tf.keras.backend.random_bernoulli((N_b, 1), 0.5, dtype=DTYPE)
    y_b1 = tf.random.uniform((N_b, 1), lb[1], ub[1], dtype=DTYPE)
    y_b2 = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((N_b, 1), 0.5, dtype=DTYPE)
    x_b2 = tf.random.uniform((N_b, 1), lb[0], ub[0], dtype=DTYPE)
    x_b = tf.concat([x_b1, x_b2], axis=0)
    y_b = tf.concat([y_b1, y_b2], axis=0)
    X_b = tf.concat([x_b, y_b], axis=1)
    
    # boundary conditions from the reference paper
    X_b0m = np.ones((2 * N_b, )) * m0
    X_b0u = np.ones((2 * N_b, )) * u_b
    
    
    # # UPDATE 28 MARCH : we don't erase points inside the cilinder, and we don't impose boundary conditions
    # # Boundary data (cilinder)
    # theta_r = tf.random.uniform((N_0, 1), 0, 2 * np.pi, dtype=DTYPE)
    # x_r = R * np.cos(theta_r)
    # y_r = R * np.sin(theta_r)
    # X_r = tf.concat([x_r, y_r], axis=1)
    # X_r0m = np.ones((N_0, )) * 0
    # X_r0u = np.ones((N_0, )) * u_b
    # # collecting all the information in DataFrames for the training
    # X0 = pd.DataFrame(np.concatenate((X_r.numpy(), X_s.numpy(), X_b.numpy())))
    # F0m = pd.DataFrame(np.concatenate((X_r0m, m_s, X_b0m)))
    # F0u = pd.DataFrame(np.concatenate((X_r0u, u_s, X_b0u)))
    
    # collecting all the information in DataFrames for the training
    X0 = pd.DataFrame(np.concatenate((X_s.numpy(), X_b.numpy())))
    F0m = pd.DataFrame(np.concatenate((m_s, X_b0m)))
    F0u = pd.DataFrame(np.concatenate((u_s, X_b0u)))
    
    # optional plot of the target for the warm start
    if verbose > 0:
        x = X0.iloc[:, 0]
        y = X0.iloc[:, 1]
        usual = matplotlib.cm.hot_r(np.arange(256))
        saturate = np.ones((int(256 / 20), 4))
        for i in range(3):
            saturate[:, i] = np.linspace(usual[-1, i], 0, saturate.shape[0])
        cmap1 = np.vstack((usual, saturate))
        cmap1 = matplotlib.colors.ListedColormap(cmap1, name='myColorMap', N=cmap1.shape[0])
        
        # mass
        plt.figure(figsize=(7, 6))
        plt.scatter(x, y, c=F0m, cmap=cmap1, marker='.', alpha=0.5)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Positions of collocation points and boundary data')
        #plt.clim(0, 3.5)
        plt.colorbar()
        plt.show()
        
        # value function
        plt.figure(figsize=(7, 6))
        plt.scatter(x, y, c=F0u, cmap=cmap1, marker='.', alpha=0.5)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Positions of collocation points and boundary data')
        plt.colorbar()
        plt.show()
    return X0, F0m, F0u

def thick_room(verbose):
    N_0 = 1000
    N_b = 1000
    N_s = 75000

    # room limits (changed on march 2023 to be bigger)
    xmin = -6.
    xmax = 6.
    ymin = -12.
    ymax = 12.


    # Lower bounds
    lb = tf.constant([xmin, ymin], dtype=DTYPE)
    # Upper bounds
    ub = tf.constant([xmax, ymax], dtype=DTYPE)
    
    # Sample room
    x_s1 = tf.random.uniform((N_s,1), lb[0], ub[0], dtype=DTYPE)
    y_s1 = tf.random.uniform((N_s,1), lb[1], ub[1], dtype=DTYPE)
    X_s1 = tf.concat([x_s1, y_s1], axis=1)
    # # erasing points inside the cilinder - NOT ANYMORE SINCE 28 MARCH 2023
    ind = tf.where(tf.norm(X_s1,axis=1)>R)
    X_s1 = tf.gather(X_s1,ind)
    X_s1 = tf.squeeze(X_s1)


    # Boundary data (square - room walls)
    x_b1 = lb[0] + (ub[0] - lb[0]) * tf.keras.backend.random_bernoulli((N_b, 1), 0.5, dtype=DTYPE)
    y_b1 = tf.random.uniform((N_b, 1), lb[1], ub[1], dtype=DTYPE)
    y_b2 = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((N_b, 1), 0.5, dtype=DTYPE)
    x_b2 = tf.random.uniform((N_b, 1), lb[0], ub[0], dtype=DTYPE)
    x_b = tf.concat([x_b1, x_b2], axis=0)
    y_b = tf.concat([y_b1, y_b2], axis=0)
    X_b = tf.concat([x_b, y_b], axis=1)


    # Boundary data (cilinder) - NOT ANYMORE SINCE 28 MARCH 2023
    # theta_r = tf.random.uniform((N_0, 1), 0, 2 * np.pi, dtype=DTYPE)
    # x_r = R * np.cos(theta_r)
    # y_r = R * np.sin(theta_r)
    # X_r = tf.concat([x_r, y_r], axis=1)

    # X0 = pd.DataFrame(np.concatenate((X_r.numpy(), X_s1.numpy(), X_b.numpy())))

    X0 = pd.DataFrame(np.concatenate((X_s1.numpy(), X_b.numpy())))
    
    return X0

########################################################################################################################

def init_DGM(RNN_layers, FNN_layers, nodes_per_layer, activation):
    # Initialize the deep galerking network
    model = DGMNet(nodes_per_layer, RNN_layers, FNN_layers, 2, activation) # input_dim = 2
    return model

########################################################################################################################

def train(u_theta,m_theta, verbose):
    # the learning rate and the optimizer are hyperparameters, we can change them to obtain better results
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [1e-1, 5e-2, 1e-2])
    optimizer_u = tf.optimizers.Adam(learning_rate=learning_rate)
    optimizer_m = tf.optimizers.Adam(learning_rate=learning_rate)


    # Train network - we sample a new room for each sampling stage 
    sampling_stages = 10
    steps_per_sample = 1001
    hist = []
    print('round-it           loss_u           loss_m')
    for i in range(sampling_stages):
    
            # sample from the required regions
            X0, F0m, F0u = sample_room(0)
            print('-----------------------------------------------')
            # for a given sample, take the prescribed number of training steps
            for j in range(steps_per_sample):
                loss_u = train_step_u(u_theta,m_theta, optimizer_u, X0, F0m)
                loss_m = train_step_m(u_theta,m_theta, optimizer_m, X0, F0m)
                hist.append(loss_m.numpy())
                if verbose > 0:
                    if j % 50 == 0:
                        print(' {:01d}-{:04d}          {:10.4e}          {:10.4e}'.format(i+1,j,loss_u,loss_m))
            
    if verbose > 1: # optional plotting of the loss function
        plt.figure(figsize=(5,5))
        plt.plot(range(len(hist[50:-1])), hist[50:-1],'k-')
        plt.xlabel('$n_{epoch}$')
        plt.title('Loss: residuals of the PDEs')
        plt.show()
    return u_theta, m_theta


def train_room(u_theta,m_theta, verbose,X0):
    # the learning rate and the optimizer are hyperparameters, we can change them to obtain better results
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([3000, 6000], [5e-2, 1e-2, 5e-3])
    optimizer_u = tf.optimizers.RMSprop(learning_rate=learning_rate)
    optimizer_m = tf.optimizers.RMSprop(learning_rate=learning_rate)
    
    x_s = X0.iloc[:, 0]
    y_s = X0.iloc[:, 1]
    m_s = m0 * (np.sqrt(x_s ** 2 + y_s ** 2) > R)
    F0m = pd.DataFrame(m_s)
    # Train network - we sample a new room for each sampling stage 
    sampling_stages = 1
    steps_per_sample = 10001
    hist = []
    print('round-it           loss_u           loss_m')
    for i in range(sampling_stages):
    
            # sample from the required regions
            #X0, F0m, F0u = sample_room(0)
            print('-----------------------------------------------')
            # for a given sample, take the prescribed number of training steps
            for j in range(steps_per_sample):
                loss_u = train_step_u(u_theta,m_theta, optimizer_u, X0, F0m)
                loss_m = train_step_m(u_theta,m_theta, optimizer_m, X0, F0m)
                hist.append(loss_m.numpy())
                if verbose > 0:
                    if j % 50 == 0:
                        print(' {:01d}-{:05d}          {:10.4e}          {:10.4e}'.format(i+1,j,loss_u,loss_m))
            
    if verbose > 1: # optional plotting of the loss function
        plt.figure(figsize=(5,5))
        plt.plot(range(len(hist[50:-1])), hist[50:-1],'k-')
        plt.xlabel('$n_{epoch}$')
        plt.title('Loss: residuals of the PDEs')
        plt.show()
    return m_theta, u_theta
########################################################################################################################
# function that computes the derivatives using automatic differentiation
# and uses them to evaluate the residuals of the FP and HJB eqs
def get_r(u_theta,m_theta, X_s):
    x = X_s
    with tf.GradientTape(persistent=True) as tape1:
        x_unstacked = tf.unstack(x, axis=1)
        tape1.watch(x_unstacked)

        # Using nested GradientTape for calculating higher order derivatives
        with tf.GradientTape() as tape2:
            # Re-stack x before passing it into f
            x_stacked = tf.stack(x_unstacked, axis=1)  # shape = (k,n)
            tape2.watch(x_stacked)
            m = m_theta(x_stacked)

        # Calculate gradient of m_theta with respect to x
        dm = tape2.batch_jacobian(m, x_stacked)  # shape = (k,n)

        # Turn df/dx into a list of n tensors of shape (k,)
        dm_unstacked = tf.unstack(dm, axis=1)

    laplacian_m = tf.zeros(shape=(x.shape[0],), dtype=DTYPE)
    for df_dxi, xi in zip(dm_unstacked, x_unstacked):
        # Take 2nd derivative of each dimension separately and sum for the laplacian
        laplacian_m = tf.math.add(tape1.gradient(df_dxi, xi), laplacian_m)  # d/dx_i (df/dx_i)

    with tf.GradientTape(persistent=True) as tape3:
        x_unstacked = tf.unstack(x, axis=1)
        tape3.watch(x_unstacked)

        # Using nested GradientTape for calculating higher order derivatives
        with tf.GradientTape() as tape4:
            # Re-stack x before passing it into u
            x_stacked = tf.stack(x_unstacked, axis=1)  # shape = (k,n)
            tape4.watch(x_stacked)
            u = u_theta(x_stacked)  # shape = (k,)

        # Calculate gradient of u with respect to x
        du = tape4.batch_jacobian(u, x_stacked)  # shape = (k,n)
        # Turn df/dx into a list of n tensors of shape (k,)
        du_unstacked = tf.unstack(du, axis=1)
    # Calculate laplacian
    
    laplacian_u = tf.zeros(shape=(x.shape[0],), dtype=DTYPE)
    for df_dxi, xi in zip(du_unstacked, x_unstacked):
        # Take 2nd derivative of each dimension separately and sum for the laplacian
        laplacian_u = tf.math.add(tape3.gradient(df_dxi, xi), laplacian_u)  # d/dx_i (df/dx_i)
    
    # reshaping vectors to be consistent for required operations
    m = tf.reshape(m,shape=(m.shape[0],1))
    u = tf.reshape(u,shape=(m.shape[0],1))
    laplacian_u = tf.reshape(laplacian_u,shape=(laplacian_u.shape[0],1))
    laplacian_m = tf.reshape(laplacian_m, shape=(laplacian_m.shape[0], 1))
    du = tf.cast(tf.reshape(du, shape=(du.shape[0], 2)), dtype=DTYPE)
    dm = tf.cast(tf.reshape(dm, shape=(dm.shape[0], 2)), dtype=DTYPE)
    
    resHJB, M_HJB = res_HJB(x,m,dm,laplacian_m,u,du,laplacian_u)
    resFP, M_FP = res_FP(m,dm,laplacian_m,du,laplacian_u)
    
    # the weights M_HJB and M_FP are the ones inspired from Anastasia's paper
    return resHJB, M_HJB, resFP, M_FP


########################################################################################################################

# residual of the Fokker Plank
def res_FP(m, m_x, m_xx, u_x, u_xx):
    
    term = tf.math.reciprocal(1+alpha*m) 
    term1m = ((sigma**2)/2)*m_xx 
    divergence = 1/mu * tf.add(tf.reduce_sum(tf.multiply(tf.math.square(term), tf.multiply(m_x, u_x)),1), 
                                                                                   tf.multiply(tf.multiply(m, term), u_xx)) 
    resFP = term1m + tf.reduce_sum(tf.multiply(s, m_x),1) + divergence  
    
    # weight for the loss 
    M_FP = tf.square(tf.norm(tf.multiply(m_x, u_x),'euclidean') + tf.norm(u_xx,'euclidean') + tf.norm(term1m,'euclidean')) 
    
    return tf.reduce_mean(tf.square(resFP)),M_FP
# residual of the HJB
def res_HJB(points, m, m_x, m_xx, u, u_x, u_xx):
    term = tf.math.reciprocal(2*mu*(1+alpha*m)) 
    term1 =  ((sigma**2)/2)*u_xx 
    term2 = tf.multiply(-tf.square(tf.norm(u_x),'euclidean'), term)

    resHJB = tf.add(term1, term2) - l - tf.reduce_sum(tf.multiply(s, u_x),1) - gamma*u - V(m, points)
    # weight for the loss 
    M_HJB = tf.square(tf.norm(term1,'euclidean') + tf.norm(-tf.square(tf.norm(u_x),'euclidean'),'euclidean') + 
                      tf.norm(l,'euclidean') + tf.norm(u_x,'euclidean') + tf.norm(V(m, points)))
    
    return tf.reduce_mean(tf.square(resHJB)), M_HJB
########################################################################################################################

def compute_loss(u_theta,m_theta, X0,F0m):
    X_s = X0[N_0:-N_b]
    X_b = X0[-N_b:-1]
    
    r_u, M_u, r_m, M_m = get_r(u_theta,m_theta, X_s) # we compute the residuals
    
    # # UPDATE 28 MARCH - we temporarely don't consider the optimal weighting
    # U_boundary = tf.square(tf.norm(u_b,'euclidean')) 
    # U_boundary = tf.cast(U_boundary,dtype=DTYPE)
    # M_boundary = tf.square(tf.norm(m0,'euclidean'))
    # M_boundary = tf.cast(M_boundary,dtype=DTYPE)   
    #M_FP = M_boundary/(M_m + M_boundary)
    #M_HJB  = U_boundary/(M_u + U_boundary)
    #gamma = (M_u + U_boundary)/((M_m + M_boundary) + (M_u + U_boundary))
    
    # # UPDATE 28 MARCH - we temporarely don't consider the boundary of the cilinder
    #m_cilinder = m_theta(X_r)
    #Lm_bC = tf.reduce_mean(tf.square(m_cilinder))
    
    # # UPDATE 28 MARCH - we consider the weights used on the report on overleaf
    m_boundary = m_theta(X_b) 
    Lm_bR = (tf.reduce_mean(tf.square(m0 - m_boundary))) # boundary discrepancy
    #loss_m = M_FP*r_m + (1-M_FP)*Lm_bR
    
    
    current_total_mass =  tf.math.reduce_mean(m_theta(X0)) * (xmax-xmin) * (ymax-ymin)
    # mass constraint 
    # total mass in the initial condition
    mass_conservation = tf.square(current_total_mass-initial_tot_mass)
    loss_m = r_m + 0.5*Lm_bR+0.1*mass_conservation
    
    u_boundary = u_theta(X_b) 
    Lu_bR = (tf.reduce_mean(tf.square(u_b - u_boundary))) # boundary discrepancy
    #loss_u = M_HJB*r_u + (1-M_HJB)*Lu_bR 
    loss_u = r_u + 0.5*Lu_bR 
    return loss_u+loss_m

#######################################################################################################################
# WARMSTART SECTION

def warmstart_loss_m(m_theta, X0, F0m):
    m_pred = m_theta(X0)
    loss = tf.reduce_mean(tf.math.square(m_pred - F0m))
    return loss


def warmstart_loss_u(u_theta, X0, F0u):
    u_pred = u_theta(X0)
    loss = tf.reduce_mean(tf.math.square(u_pred - F0u))
    return loss

def grad_warmstart_u(u_theta, X0, F0u):
    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with
        # respect to trainable variables
        theta = u_theta.trainable_variables
        tape.watch(theta)
        loss = warmstart_loss_u(u_theta, X0, F0u)

    g = tape.gradient(loss, theta)
    del tape
    return loss, g

def grad_warmstart_m(m_theta, X0, F0m):
    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with
        # respect to trainable variables
        tape.watch(m_theta.trainable_variables)
        loss = warmstart_loss_m(m_theta, X0, F0m)

    g = tape.gradient(loss, m_theta.trainable_variables)
    del tape
    return loss, g


def train_warmstart_u(u_theta, optim, X0, F0u):
    # Compute current loss and gradient w.r.t. parameters
    loss, grad_theta = grad_warmstart_u(u_theta, X0, F0u)
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, u_theta.trainable_variables))
    return loss

def train_warmstart_m(m_theta, optim, X0, F0m):
    # Compute current loss and gradient w.r.t. parameters
    loss, grad_theta = grad_warmstart_m(m_theta, X0, F0m)
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, m_theta.trainable_variables))
    return loss

def warmstart(u_theta,m_theta,verbose):

    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [5e-1, 1e-1, 5e-3])
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    #  Train network
    # for each sampling stage
    sampling_stages = 1
    steps_per_sample = 501
    
    hist = []
    if verbose > 0:
        print('------ preprocessing for u_theta ------')
    # preprocessing for u
    for i in range(sampling_stages):

        # sample from the required regions
        X0, F0m, F0u = sample_room(0)
        loss = warmstart_loss_u(u_theta, X0, F0u)
        hist.append(loss.numpy())

        # for a given sample, take the required number of SGD steps
        for j in range(steps_per_sample):
            loss = train_warmstart_u(u_theta, optimizer,X0, F0u)
            if verbose > 0:
                if j % 50 == 0:
                    print('u_theta It {:05d}: loss = {:10.8e}'.format(j, loss))
                    
    hist = []
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [5e-1, 1e-1, 5e-2])
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    if verbose > 0:
        print('------ preprocessing  for m_theta------')
    # preprocessing for m
    for i in range(sampling_stages):

        # sample from the required regions
        X0,F0m,F0u = sample_room(0)
        loss_m = warmstart_loss_m(m_theta, X0, F0m)
        hist.append(loss_m.numpy())

        # for a given sample, take the required number of training steps
        for j in range(steps_per_sample):
            loss = train_warmstart_m(m_theta, optimizer, X0, F0m)
            if verbose > 0:
                if j % 50 == 0:
                    print('m_theta It {:05d}: loss = {:10.8e}'.format(j, loss))
    return m_theta, u_theta


def warmstart_room(u_theta,m_theta,verbose,X0):

    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [5e-1, 1e-1, 5e-3])
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    x_s = X0.iloc[:, 0]
    y_s = X0.iloc[:, 1]
    m_s = m0 * (np.sqrt(x_s ** 2 + y_s ** 2) > R)
    F0m = pd.DataFrame(m_s)
    u_s = u_b + np.exp(-(x_s ** 2 + y_s ** 2) / 2 * 0.1 ** 2) / (2 * np.pi * 0.1 ** 2)
    F0u = pd.DataFrame(u_s)

    #  Train network
    # for each sampling stage
    sampling_stages = 1
    steps_per_sample = 501
    
    hist = []
    if verbose > 0:
        print('------ preprocessing for u_theta ------')
    # preprocessing for u
    for i in range(sampling_stages):
        # for a given sample, take the required number of SGD steps
        for j in range(steps_per_sample):
            loss = train_warmstart_u(u_theta, optimizer,X0, F0u)
            if verbose > 0:
                if j % 50 == 0:
                    print('u_theta It {:05d}: loss = {:10.8e}'.format(j, loss))
                    
    hist = []
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [5e-1, 1e-1, 5e-2])
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    if verbose > 0:
        print('------ preprocessing  for m_theta------')
    # preprocessing for m
    for i in range(sampling_stages):

        # sample from the required regions
        X0,F0m,F0u = sample_room(0)
        loss_m = warmstart_loss_m(m_theta, X0, F0m)
        hist.append(loss_m.numpy())

        # for a given sample, take the required number of training steps
        for j in range(steps_per_sample):
            loss = train_warmstart_m(m_theta, optimizer, X0, F0m)
            if verbose > 0:
                if j % 50 == 0:
                    print('m_theta It {:05d}: loss = {:10.8e}'.format(j, loss))
    return m_theta, u_theta

#######################################################################################################################
# This function was used to train a network approximating FD solutions from reference paper (Matteo's). 
# This was just a check and this part is not used in the main algorithm
def warmstart_prova(m_theta,u_theta, X_s,m_s,u_s):

    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [5e-2, 1e-2, 5e-4])
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    #  Train network
    # for each sampling stage
    sampling_stages = 1
    steps_per_sample = 1001
    hist = []
    if True:
        print('------ preprocessing for m_theta ------')
    # preprocessing for m
    for i in range(sampling_stages):

        # sample from the required region
        loss = warmstart_loss_m(m_theta, X_s, m_s)
        hist.append(loss.numpy())

        # for a given sample, take the required number of SGD steps
        for j in range(steps_per_sample):
            loss = train_warmstart_m(m_theta, optimizer, X_s, m_s)
            if True:
                if j % 50 == 0:
                    print('m_theta It {:05d}: loss = {:10.8e}'.format(j, loss))
    hist = []
    if True:
        print('------ preprocessing for u_theta ------')
    # preprocessing for u
    for i in range(sampling_stages):

        
        loss = warmstart_loss_u(u_theta, X_s, u_s)
        hist.append(loss.numpy())

        # for a given sample, take the required number of SGD steps
        for j in range(steps_per_sample):
            loss = train_warmstart_u(u_theta, optimizer, X_s, u_s)
            if True:
                if j % 50 == 0:
                    print('u_theta It {:05d}: loss = {:10.8e}'.format(j, loss))
    return m_theta, u_theta
########################################################################################################################

# gradient of the loss function with respect to the unknown variables in the model, also called `trainable variables`
def get_grad(u_theta,m_theta, X0,F0m, target):
    if target == 'FP':
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(m_theta.trainable_variables)
            loss = compute_loss(u_theta,m_theta, X0, F0m)

        g = tape.gradient(loss, m_theta.trainable_variables)
        del tape
        return loss, g
    elif target == 'HJB':
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(u_theta.trainable_variables)
            loss = compute_loss(u_theta, m_theta, X0, F0m)

        g = tape.gradient(loss, u_theta.trainable_variables)
        del tape
        return loss, g
########################################################################################################################

# Define one training step as a TensorFlow function to increase speed of training
@tf.function
def train_step_u(u,m, optim, data, F0m):
    # Compute current loss and gradient w.r.t. parameters
    loss_u, grad_theta_u = get_grad(u,m, data,F0m, 'HJB')
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta_u, u.trainable_variables))
    return loss_u

@tf.function
def train_step_m(u,m, optim, data,F0m):
    # Compute current loss and gradient w.r.t. parameters
    loss_m, grad_theta_m = get_grad(u,m, data,F0m, 'FP')
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta_m, m.trainable_variables))
    return loss_m


