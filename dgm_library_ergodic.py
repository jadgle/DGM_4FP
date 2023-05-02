
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
DTYPE = 'float32'

# Problem parameters
sigma = 0.35
g     = -0.005
mu    = 1
gamma = 0
m0    = 2.5
alpha = 0
R     = 0.37
V_const = -10e2

# room limits (changed on march 2023 to be bigger)
xmin = -6.
xmax = 6.
ymin = -12.
ymax = 12.
initial_tot_mass = tf.constant(2.5 * (xmax-xmin) * (ymax-ymin),dtype=DTYPE)

l     = -((g*m0)/(1+alpha*m0))+(gamma*mu*sigma**2*np.log(np.sqrt(m0)))
s     =  tf.constant([0, -0.6],  dtype=DTYPE, shape=(1, 2))

# Constants of the agents
Xi = np.sqrt(np.abs((mu*sigma**4)/(2*g*m0)))
Cs = np.sqrt(np.abs((g*m0)/(2*mu)))

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
def V(Phi,Gamma, x):
    U0 = tf.zeros(shape = (x.shape[0],1),dtype=DTYPE)
    U0 = tf.unstack(U0)
    for i in range(x.shape[0]):
        if tf.less_equal(tf.norm(x[i],'euclidean'),R): # for points in the cilinder
            U0[i] = tf.constant(V_const,dtype=DTYPE)   # we have higher cost
        else:
            U0[i] = tf.constant(0,dtype=DTYPE)
    U0 = tf.stack(U0)
    return g * tf.multiply(Phi,Gamma) + U0 # formula for the potential from reference paper



########################################################################################################################

def sample_room(verbose):
    
    # number of points
    N_0 = 200
    N_b = 500
    N_s = 5000
    
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
    #erasing points inside the cilinder 

    ind1 = tf.where(tf.norm(X_s, axis=1) <= R)
    ind = tf.where(tf.norm(X_s, axis=1) > R)
    X_c = tf.gather(X_s, ind1)
    X_s = tf.gather(X_s, ind)
    N_s = X_s.shape[0]
    X_s = tf.squeeze(X_s)
    X_c = tf.squeeze(X_c)

    # Boundary data (square - outside walls)
    x_b1 = lb[0] + (ub[0] - lb[0]) * tf.keras.backend.random_bernoulli((N_b, 1), 0.5, dtype=DTYPE)
    y_b1 = tf.random.uniform((N_b, 1), lb[1], ub[1], dtype=DTYPE)
    y_b2 = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((N_b, 1), 0.5, dtype=DTYPE)
    x_b2 = tf.random.uniform((N_b, 1), lb[0], ub[0], dtype=DTYPE)
    x_b = tf.concat([x_b1, x_b2], axis=0)
    y_b = tf.concat([y_b1, y_b2], axis=0)
    X_b = tf.concat([x_b, y_b], axis=1)

    
    # Boundary data (cilinder) 
    # theta_r = tf.random.uniform((N_0, 1), 0, 2 * np.pi, dtype=DTYPE)
    # x_r = R * np.cos(theta_r)
    # y_r = R * np.sin(theta_r)
    # X_r = tf.concat([x_r, y_r], axis=1)
    
    
    return pd.DataFrame(X_b.numpy()), pd.DataFrame(X_s.numpy()), pd.DataFrame(X_c.numpy())



def init_DGM(RNN_layers, FNN_layers, nodes_per_layer, activation):
    # Initialize the deep galerking network
    model = DGMNet(nodes_per_layer, RNN_layers, FNN_layers, 2, activation) # input_dim = 2
    return model

########################################################################################################################

def train(Phi_theta,Gamma_theta,verbose):
    # the learning rate and the optimizer are hyperparameters, we can change them to obtain better results
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([200, 300, 1200], [1e-1, 7e-2, 5e-2, 1e-2])
    optimizer_Phi = tf.optimizers.Adam(learning_rate=learning_rate)
    optimizer_Gamma = tf.optimizers.Adam(learning_rate=learning_rate)


    # Train network - we sample a new room for each sampling stage 
    sampling_stages = 5
    steps_per_sample = 1001
    hist = []
    print('round-it           loss')
    for i in range(sampling_stages):
            X_b, X_s, X_c = sample_room(0)
            
            
            print('-----------------------------------------------')
            # for a given sample, take the prescribed number of training steps
            for j in range(steps_per_sample):
                loss_Phi = train_step_Phi(Phi_theta,Gamma_theta, optimizer_Phi, X_b, X_s, X_c)
                loss_Gamma = train_step_Gamma(Phi_theta,Gamma_theta, optimizer_Gamma, X_b, X_s, X_c)
                hist.append(loss_Gamma.numpy())
                if verbose > 0:
                    if j % 50 == 0:
                        print(' {:01d}-{:04d}          {:10.4e}'.format(i+1,j,loss_Gamma))
            
    if verbose > 1: # optional plotting of the loss function
        plt.figure(figsize=(5,5))
        plt.plot(range(len(hist[50:-1])), hist[50:-1],'k-')
        plt.xlabel('$n_{epoch}$')
        plt.title('Loss: residuals of the PDEs')
        plt.show()
    return Phi_theta,Gamma_theta


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
    
    resHJB = res_HJB(x,m,dm,laplacian_m,u,du,laplacian_u)
    resFP = res_FP(x,m,dm,laplacian_m,u,du,laplacian_u)
    
    # the weights M_HJB and M_FP are the ones inspired from Anastasia's paper
    return resHJB, resFP


########################################################################################################################

# residual of the Fokker Plank
def res_FP(points, Gamma, Gamma_x, Gamma_xx, Phi, Phi_x, Phi_xx):
    term1 = mu*sigma**2*tf.reduce_sum(tf.multiply(s, Gamma_x),1) 
    term2 = ((mu*sigma**4)/2)*Gamma_xx 
    term_pot = tf.multiply(V(Gamma,Phi,points),Gamma)
    term_log = 0#gamma*mu*sigma**2*tf.multiply(Gamma,tf.math.log(Phi))
    
    resFP = l*Gamma + term1  +term2 +term_pot + term_log
    
    return tf.reduce_mean(tf.square(resFP))

# residual of the HJB
def res_HJB(points, Gamma, Gamma_x, Gamma_xx, Phi, Phi_x, Phi_xx):
    term1 = -mu*sigma**2*tf.reduce_sum(tf.multiply(s, Phi_x),1) 
    term2 = ((mu*sigma**4)/2)*Phi_xx 
    term_pot = tf.multiply(V(Gamma,Phi,points),Phi)
    term_log = 0#-gamma*mu*sigma**2*tf.multiply(Phi,tf.math.log(Phi))
    
    resHJB = l*Phi + term1  +term2 +term_pot + term_log
    
    return tf.reduce_mean(tf.square(resHJB))
########################################################################################################################

def compute_loss(Phi_theta,Gamma_theta, X_b, X_s, X_c):
    
    r_Phi, r_Gamma = get_r(Phi_theta,Gamma_theta, X_s) # we compute the residuals
    
    #  we consider the weights used on the report on overleaf
    m_boundary = tf.multiply(Gamma_theta(X_b),Phi_theta(X_b))
    m_bRoom = (tf.reduce_mean(tf.square(m0 - m_boundary))) # boundary discrepancy
    # Gamma_cilinder = Gamma_theta(X_r)
    # Gamma_bC = tf.reduce_mean(tf.square(Gamma_cilinder))


    #current_total_mass =  tf.math.reduce_mean(tf.multiply(Gamma_theta(X_s),Phi_theta(X0))) * (xmax-xmin) * (ymax-ymin)
    # mass constraint 
    # total mass in the initial condition
    
    m_Cyl = tf.reduce_mean(tf.multiply(Gamma_theta(X_c),Phi_theta(X_c)))
    
    #mass_conservation = tf.square(current_total_mass-initial_tot_mass)
    
    return r_Phi+r_Gamma+m_bRoom + m_Cyl# + mass_conservation


########################################################################################################################

# gradient of the loss function with respect to the unknown variables in the model, also called `trainable variables`
def get_grad(u_theta,m_theta, X_b, X_s, X_c, target):
    if target == 'FP':
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(m_theta.trainable_variables)
            loss = compute_loss(u_theta,m_theta, X_b, X_s, X_c)

        g = tape.gradient(loss, m_theta.trainable_variables)
        del tape
        return loss, g
    elif target == 'HJB':
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(u_theta.trainable_variables)
            loss = compute_loss(u_theta, m_theta, X_b, X_s, X_c)

        g = tape.gradient(loss, u_theta.trainable_variables)
        del tape
        return loss, g
########################################################################################################################

# Define one training step as a TensorFlow function to increase speed of training
@tf.function
def train_step_Phi(u,m, optim, X_b, X_s, X_c):
    # Compute current loss and gradient w.r.t. parameters
    loss_u, grad_theta_u = get_grad(u,m, X_b, X_s, X_c, 'HJB')
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta_u, u.trainable_variables))
    return loss_u

@tf.function
def train_step_Gamma(u,m, optim, X_b, X_s, X_c):
    # Compute current loss and gradient w.r.t. parameters
    loss_m, grad_theta_m = get_grad(u,m, X_b, X_s, X_c, 'FP')
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta_m, m.trainable_variables))
    return loss_m


