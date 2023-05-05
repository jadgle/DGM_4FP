
##################################################################################################################     
#########################                                                            #############################  
#########################           Core methods - DGM for pedestrian MFG            #############################
#########################                                                            #############################
##################################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import os
from tensorflow.python.ops.numpy_ops import np_config
from DGM import  DGMNet
import matplotlib

########################################################################################################################
DTYPE = 'float32'

# Problem parameters
V_const = -10e2

sigma =  0.35# 0.35 # 0.28 (grid -4,4)
g     = -0.005#-0.005 # -0.03
mu    = 1
gamma = 0
m0    = 2.5
alpha = 0
l     = -((g*m0)/(1+alpha*m0))+(gamma*mu*sigma**2*np.log(np.sqrt(m0))) # 0.08
u_b   = -mu*sigma**2*np.log(np.sqrt(m0))
R     = 0.37
s     =  tf.constant([0, -0.6],  dtype=DTYPE, shape=(1, 2))#tf.constant([0, -0.3],  dtype=DTYPE, shape=(1, 2)) # -0.3
v0 = m0**((-mu*sigma**2)/2)

# Constants of the agents
Xi = np.sqrt(np.abs((mu*sigma**4)/(2*g*m0)))
Cs = np.sqrt(np.abs((g*m0)/(2*mu)))

# room limits (changed on march 2023 to be bigger)
xmin = -6.
xmax = 6.
ymin = -6.
ymax = 6.
initial_tot_mass = tf.constant(m0 * (xmax-xmin) * (ymax-ymin),dtype=DTYPE)



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
# def V(Phi,Gamma, x):
#     U0 = np.zeros(shape = (x.shape[0],1))
#     for i in range(x.shape[0]):
#         if tf.less_equal(tf.norm(x[i],'euclidean'),R): # for points in the cilinder
#             U0[i] = V_const   # we have higher cost
#     U0 = tf.convert_to_tensor(U0, dtype=DTYPE)
#     return g * tf.multiply(Phi,Gamma) + U0 # formula for the potential from reference paper

def V(Phi,Gamma, x):
    U0 = tf.zeros(shape = (x.shape[0],1),dtype=DTYPE)
    U0 = tf.unstack(U0)
    for i in range(x.shape[0]):
        if tf.less_equal(tf.norm(x[i],'euclidean'),R): # for points in the cilinder
            U0[i] = tf.constant(V_const,dtype=DTYPE)   # we have higher cost
        else:
            U0[i] = tf.constant(0,dtype=DTYPE)   # we have higher cost
    U0 = tf.stack(U0)
    return g * tf.multiply(Phi,Gamma) + U0 # formula for the potential from reference paper





########################################################################################################################

def sample_room(verbose):
    # number of points
    N_b = 200
    N_s = 5000
    
    # room limits (changed on march 2023 to be bigger)
    xmin = -6.
    xmax = 6.
    ymin = -6.
    ymax = 6.
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
    #learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([100, 500], [1e-1, 7e-2, 5e-2])
    optimizer_Phi = tf.optimizers.Adam()
    optimizer_Gamma = tf.optimizers.Adam()


    # Train network - we sample a new room for each sampling stage 
    sampling_stages = 10
    steps_per_sample = 2001
    hist = []
    print('round-it           loss')
    for i in range(sampling_stages):
            X_b, X_s, X_c = sample_room(0)
            print('-----------------------------------------------')
            # for a given sample, take the prescribed number of training steps
            for j in range(steps_per_sample):
                loss_Phi, r_Phi, r_Gamma, m_bRoom   = train_step_Phi(Phi_theta,Gamma_theta, optimizer_Phi, X_b, X_s, X_c)
                loss_Gamma, r_Phi, r_Gamma, m_bRoom = train_step_Gamma(Phi_theta,Gamma_theta, optimizer_Gamma, X_b, X_s, X_c)
                hist.append(np.mean(np.array([loss_Gamma,loss_Phi])))
                if verbose > 0:
                    if j % 50 == 0:
                        print(' {:01d}-{:04d}          {:10.4e}'.format(i+1,j,loss_Gamma))
                        print('--- residual_Phi= ' + str(r_Phi.eval()) + ' --- residual_Gamma= '+ str(r_Gamma.eval()) + ' --- bordo= ' + str(m_bRoom.eval()))

    if verbose > 1: # optional plotting of the loss function
        plt.figure(figsize=(5,5))
        plt.plot(range(len(hist[50:-1])), hist[50:-1],'k-')
        plt.xlabel('$n_{epoch}$')
        plt.title('Loss: residuals of the PDEs')
        plt.show()
    return Phi_theta,Gamma_theta, X_b, X_s, X_c


########################################################################################################################
def get_derivatives(f_theta, x): # function that computes the derivatives using automatic differentiation
    with tf.GradientTape(persistent=True) as tape1:
        x_unstacked = tf.unstack(x, axis=1)
        tape1.watch(x_unstacked)

        # Using nested GradientTape for calculating higher order derivatives
        with tf.GradientTape() as tape2:
            # Re-stack x before passing it into f
            x_stacked = tf.stack(x_unstacked, axis=1)  # shape = (k,n)
            tape2.watch(x_stacked)
            f = f_theta(x_stacked)

        # Calculate gradient of m_theta with respect to x
        grad_f = tape2.batch_jacobian(f, x_stacked)  # shape = (k,n)

        # Turn df/dx into a list of n tensors of shape (k,)
        df_unstacked = tf.unstack(grad_f, axis=1)

    laplacian_f = []
    for df_dxi, xi in zip(df_unstacked, x_unstacked):
        # Take 2nd derivative of each dimension separately and sum for the laplacian
        laplacian_f.append(tape1.gradient(df_dxi, xi))  # d/dx_i (df/dx_i)
    laplacian_f = sum(laplacian_f)
    return grad_f, laplacian_f


# function to evaluate the residuals of the PDEs 
def get_residuals(Phi_theta,Gamma_theta, x):
    Phi = Phi_theta(x)
    Gamma = Gamma_theta(x)
    
    grad_Phi, laplacian_Phi = get_derivatives(Phi_theta, x)
    grad_Gamma, laplacian_Gamma = get_derivatives(Gamma_theta, x)
    
    #print(grad_Phi.shape)
    #print(laplacian_Phi.shape)
    
    # reshaping vectors to be consistent for required operations
    Phi   = tf.reshape(Phi,shape=(Phi.shape[0],1))
    Gamma = tf.reshape(Gamma,shape=(Gamma.shape[0],1))
    
    laplacian_Phi = tf.reshape(laplacian_Phi,shape=(laplacian_Phi.shape[0],1))
    laplacian_Gamma = tf.reshape(laplacian_Gamma, shape=(laplacian_Gamma.shape[0], 1))
    
    grad_Phi = tf.reshape(grad_Phi, shape=(grad_Phi.shape[0], 2))
    grad_Gamma = tf.reshape(grad_Gamma, shape=(grad_Gamma.shape[0], 2))
    #print(grad_Phi.shape)
    #print(laplacian_Phi.shape)
    res_Phi = residual_Phi(x, Gamma, grad_Gamma, laplacian_Gamma, Phi, grad_Phi, laplacian_Phi)
    res_Gamma= residual_Gamma(x, Gamma, grad_Gamma, laplacian_Gamma, Phi, grad_Phi, laplacian_Phi)

    return res_Phi, res_Gamma


########################################################################################################################

# residual of the Fokker Plank
def residual_Gamma(points, Gamma, Gamma_x, Gamma_xx, Phi, Phi_x, Phi_xx):
    term1 = mu*sigma**2*tf.reduce_sum(tf.multiply(s, Gamma_x),1) 
    term2 = ((mu*sigma**4)/2)*Gamma_xx 
    term_pot = tf.multiply(V(Gamma,Phi,points),Gamma)
    term_log = 0#gamma*mu*sigma**2*tf.multiply(Gamma,tf.math.log(Phi))
    
    resFP = l*Gamma + term1  +term2 +term_pot + term_log
    #print('calcolo residuo Gamma')
    return tf.norm(resFP)

# residual of the HJB
def residual_Phi(points, Gamma, Gamma_x, Gamma_xx, Phi, Phi_x, Phi_xx):
    term1 = -mu*sigma**2*tf.reduce_sum(tf.multiply(s, Phi_x),1) 
    term2 = ((mu*sigma**4)/2)*Phi_xx 
    term_pot = tf.multiply(V(Gamma,Phi,points),Phi)
    term_log = 0#-gamma*mu*sigma**2*tf.multiply(Phi,tf.math.log(Phi))
    
    resHJB = l*Phi + term1  +term2 +term_pot + term_log
    #print('calcolo residuo Phi')
    return tf.norm(resHJB)
########################################################################################################################

def compute_loss(Phi_theta,Gamma_theta, X_b, X_s, X_c):
    
    r_Phi, r_Gamma = get_residuals(Phi_theta,Gamma_theta, X_s) # we compute the residuals
    
    #  we consider the weights used on the report on overleaf
    m_bRoom = tf.norm(tf.sqrt(m0) - Gamma_theta(X_b)) + tf.norm(tf.sqrt(m0) - Phi_theta(X_b)) # boundary discrepancy
    # Gamma_cilinder = Gamma_theta(X_r)
    # Gamma_bC = tf.reduce_mean(tf.square(Gamma_cilinder))

    #current_total_mass =  tf.math.reduce_mean(tf.multiply(Gamma_theta(X_s),Phi_theta(X0))) * (xmax-xmin) * (ymax-ymin)
    # mass constraint 
    # total mass in the initial condition
    
    m_Cyl = tf.norm(Gamma_theta(X_c))+tf.norm(Phi_theta(X_c))
    
    #mass_conservation = tf.square(current_total_mass-initial_tot_mass)
    return r_Phi + r_Gamma + m_bRoom, r_Phi, r_Gamma, m_bRoom + m_Cyl# + mass_conservation


########################################################################################################################

# gradient of the loss function with respect to the unknown variables in the model, also called `trainable variables`
def get_grad(Phi,Gamma, X_b, X_s, X_c, target):
    if target == 'Gamma':
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(Gamma.trainable_variables)
            loss, r_Phi, r_Gamma, m_bRoom = compute_loss(Phi,Gamma, X_b, X_s, X_c)

        g = tape.gradient(loss, Gamma.trainable_variables)
        del tape
        return loss, g, r_Phi, r_Gamma, m_bRoom
    elif target == 'Phi':
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(Phi.trainable_variables)
            loss, r_Phi, r_Gamma, m_bRoom = compute_loss(Phi,Gamma, X_b, X_s, X_c)

        g = tape.gradient(loss, Phi.trainable_variables)
        del tape
        return loss, g, r_Phi, r_Gamma, m_bRoom
########################################################################################################################

# Define one training step as a TensorFlow function to increase speed of training
@tf.function
def train_step_Phi(Phi,Gamma, optim, X_b, X_s, X_c):
    # Compute current loss and gradient w.r.t. parameters
    loss_Phi, grad_theta_Phi, r_Phi, r_Gamma, m_bRoom = get_grad(Phi,Gamma, X_b, X_s, X_c, 'Phi')
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta_Phi, Phi.trainable_variables))
    return loss_Phi, r_Phi, r_Gamma, m_bRoom

@tf.function
def train_step_Gamma(Phi,Gamma, optim, X_b, X_s, X_c):
    # Compute current loss and gradient w.r.t. parameters
    loss_Gamma, grad_theta_Gamma, r_Phi, r_Gamma, m_bRoom = get_grad(Phi,Gamma, X_b, X_s, X_c, 'Gamma')
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta_Gamma, Gamma.trainable_variables))
    return loss_Gamma, r_Phi, r_Gamma, m_bRoom


#######################################################################################################################
# WARMSTART SECTION

def warmstart_loss(f, X0):
    f_pred = f(X0)
    loss = tf.norm(f_pred - tf.sqrt(m0))
    return loss

def grad_warmstart(f, X0):
    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with
        # respect to trainable variables
        theta = f.trainable_variables
        tape.watch(theta)
        loss = warmstart_loss(f, X0)

    g = tape.gradient(loss, theta)
    del tape
    return loss, g


def train_warmstart(f, optim, X0):
    # Compute current loss and gradient w.r.t. parameters
    loss, grad_theta = grad_warmstart(f, X0)
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, f.trainable_variables))
    return loss

def warmstart(Phi_theta,Gamma_theta,verbose):

    #learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([100, 3000], [5e-1, 1e-1, 5e-3])
    optimizer = tf.optimizers.Adam()

    #  Train network
    # for each sampling stage
    sampling_stages = 1
    steps_per_sample = 2001
    
    hist = []
    if verbose > 0:
        print('------ preprocessing for Gamma_theta ------')
    # preprocessing for u
    for i in range(sampling_stages):

        # sample from the required regions
        X_b, X_s, X_c = sample_room(0)
        X0 = tf.concat([X_b, X_s, X_c ],0)
        loss = warmstart_loss(Gamma_theta, X0)
        hist.append(loss.numpy())

        # for a given sample, take the required number of SGD steps
        for j in range(steps_per_sample):
            loss = train_warmstart(Gamma_theta, optimizer,X0)
            if verbose > 0:
                if j % 50 == 0:
                    print('Gamma_theta It {:05d}: loss = {:10.8e}'.format(j, loss))
                    
    hist = []
    #learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [5e-1, 1e-1, 5e-2])
    optimizer = tf.optimizers.Adam()
    if verbose > 0:
        print('------ preprocessing  for Phi_theta------')
    # preprocessing for m
    for i in range(sampling_stages):

        # sample from the required regions
        X_b, X_s, X_c = sample_room(0)
        X0 = tf.concat([X_b, X_s, X_c ],0)
        loss = warmstart_loss(Phi_theta, X0)
        hist.append(loss.numpy())

        # for a given sample, take the required number of training steps
        for j in range(steps_per_sample):
            loss = train_warmstart(Phi_theta, optimizer, X0)
            if verbose > 0:
                if j % 50 == 0:
                    print('Phi_theta It {:05d}: loss = {:10.8e}'.format(j, loss))
    return Phi_theta, Gamma_theta


#######################################################################################################################
# WARMSTART SECTION / VS SOLUTION

def warmstart_loss_sol(f, f_true, X0):
    f_pred = f(X0)
    loss = tf.norm(f_pred - f_true)
    return loss

def grad_warmstart_sol(f, f_true, X0):
    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with
        # respect to trainable variables
        theta = f.trainable_variables
        tape.watch(theta)
        loss = warmstart_loss_sol(f,f_true, X0)

    g = tape.gradient(loss, theta)
    del tape
    return loss, g


def train_warmstart_sol(f, f_true, optim, X0):
    # Compute current loss and gradient w.r.t. parameters
    loss, grad_theta = grad_warmstart_sol(f, f_true, X0)
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, f.trainable_variables))
    return loss

def warmstart_sol(Phi_theta,Gamma_theta,phi,gamma,verbose):
    Nx = 150
    Ny = 150
    Lx = 6
    Ly = 6
    #Define grid 
    x = np.linspace(-Lx,Lx,Nx)
    y = np.linspace(-Ly,Ly,Ny)
    X,Y = np.meshgrid(x,y)
    x = X.reshape((Nx*Ny,1))
    y = Y.reshape((Nx*Ny,1))
    X0 = pd.DataFrame(np.concatenate([x,y],1))    
    X0 = X0.astype(dtype = DTYPE)
    phi = phi.reshape((Nx*Ny,1))
    gamma = gamma.reshape((Nx*Ny,1))
    
    #learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([100, 3000], [1e-1, 5e-2, 1e-2])
    optimizer = tf.optimizers.Adam()

    #  Train network
    # for each sampling stage
    sampling_stages = 1
    steps_per_sample = 5001
    
    hist = []
    if verbose > 0:
        print('------ preprocessing for Gamma_theta ------')
    # preprocessing for u
    for i in range(sampling_stages):
        loss = warmstart_loss_sol(Gamma_theta, gamma, X0)
        hist.append(loss.numpy())

        # for a given sample, take the required number of SGD steps
        for j in range(steps_per_sample):
            loss = train_warmstart_sol(Gamma_theta, gamma, optimizer,X0)
            if verbose > 0:
                if j % 100 == 0:
                    print('Gamma_theta It {:05d}: loss = {:10.8e}'.format(j, loss))
                    
    hist = []
    #learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [5e-1, 1e-1, 5e-2])
    optimizer = tf.optimizers.Adam()
    if verbose > 0:
        print('------ preprocessing  for Phi_theta------')
    # preprocessing for m
    for i in range(sampling_stages):
        loss = warmstart_loss_sol(Phi_theta, phi, X0)
        hist.append(loss.numpy())

        # for a given sample, take the required number of training steps
        for j in range(steps_per_sample):
            loss = train_warmstart_sol(Phi_theta, phi, optimizer, X0)
            if verbose > 0:
                if j % 100 == 0:
                    print('Phi_theta It {:05d}: loss = {:10.8e}'.format(j, loss))        
    return Phi_theta, Gamma_theta, X0

