########################################################################################################################
##############################                                                            ##############################
##############################              Theory of Functional Connections              ##############################
##############################                                                            ##############################
########################################################################################################################


# This section has not been updated/debugged on 28 march 2023, as we are not diving deeper in this method atm. 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import numpy as np
import random
import os


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
V_const = 10e2

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
# TFC
# potential
def V_polar(m, r):
    U0 = tf.zeros(shape = r.shape,dtype=DTYPE)
    U0 = tf.unstack(U0)
    # for i in range(r.shape[0]):
    #     if (tf.equal(r[i],R)):
    #         U0[i] = tf.constant(10,dtype=DTYPE)
    U0 = tf.stack(U0)
    return g * m + U0


########################################################################################################################

def sample_room_polar(verbose):
    N_s = 20000

    r_min     = tf.constant(R+10**(-1), dtype = DTYPE)
    r_max     = tf.constant(B, dtype = DTYPE)
    theta_min = tf.constant(0, dtype = DTYPE)
    theta_max = tf.constant(2*np.pi, dtype = DTYPE)

    # Draw uniform sample points for data in the domain
    rho   = tf.random.uniform((N_s, 1), r_min, r_max, dtype = DTYPE)
    theta = tf.random.uniform((N_s, 1), theta_min, theta_max, dtype = DTYPE)
    X_s   = tf.concat([rho, theta], axis=1)
    X0    = pd.DataFrame(X_s.numpy())


    m_s = (m0/(B))*rho
    u_s = (v0/(B))*rho
    
    F0m = pd.DataFrame(m_s)
    F0u = pd.DataFrame(u_s)
    
    uno = np.ones(rho.shape)
    dC = tf.concat([uno*R,theta],axis=1)
    dB = tf.concat([uno*B,theta],axis=1)
    
    dC = pd.DataFrame(dC.numpy())
    dB = pd.DataFrame(dB.numpy())
    #rho = pd.DataFrame(rho.numpy())
    #theta = pd.DataFrame(theta.numpy())
    
    # if verbose>1:
    #     x = rho.values*np.cos(theta.values)
    #     y = rho.values*np.sin(theta.values)
    #     plt.figure(figsize=(6, 6))
    #     plt.scatter(x, y, marker='.', alpha=0.5)
    
    return X0, dC, dB, rho, theta, F0m, F0u

########################################################################################################################

def warmstart_TFC(f_theta,g_theta, verbose):
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [2e-1, 5e-2, 1e-2])
    optimizer_g = tf.optimizers.RMSprop(learning_rate)
    optimizer_g.build(g_theta.trainable_variables)
    optimizer_f = tf.optimizers.RMSprop(learning_rate)
    optimizer_f.build(f_theta.trainable_variables)

    steps_per_sample = 200
    hist = []
    if verbose > 0:
        print('------ preprocessing ------')
    # preprocessing for g
        # sample from the required regions
        X0, dC, dB, rho, theta, F0m, F0u = sample_room_polar(0)
        loss = warmstart_loss_f(g_theta, X0, dC, dB, rho, F0u)
        hist.append(loss.numpy())

        # for a given sample, take the required number of SGD steps
        for j in range(steps_per_sample):
            loss = train_warmstart_g(g_theta, X0, dC, dB, rho, F0m, optimizer_g)
            loss = train_warmstart_f(f_theta, X0, dC, dB, rho, F0u, optimizer_f)
            if verbose > 0:
                if j % 50 == 0:
                    print('warm-up It {:05d}: loss = {:10.8e}'.format(j, loss))
    return g_theta, f_theta

def warmstart_loss_g(g_theta, X0, dC, dB, rho, F0m):
    m = g_theta(X0) +  (1/(B-R))*((rho-B)*g_theta(dC)  + (rho-R)*(m0 - g_theta(dB)))
    loss = tf.reduce_mean(tf.square(m - F0m))
    return loss


def warmstart_loss_f(f_theta, X0, dC, dB, rho, F0u):
    v = f_theta(X0) +  (1/(B-R))*((rho-B)*f_theta(dC)  + (rho-R)*(v0 - f_theta(dB)))
    loss = tf.reduce_mean(tf.square(v - F0u))
    return loss

def grad_warmstart_f(f_theta, X0, dC, dB, rho, F0u):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(f_theta.trainable_variables)
        loss = warmstart_loss_f(f_theta, X0, dC, dB, rho, F0u)
    f = tape.gradient(loss, f_theta.trainable_variables)
    del tape
    return loss, f

def grad_warmstart_g(g_theta, X0, dC, dB, rho, F0m):
    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with
        # respect to trainable variables
        tape.watch(g_theta.trainable_variables)
        loss = warmstart_loss_g(g_theta, X0, dC, dB, rho, F0m)
    g = tape.gradient(loss, g_theta.trainable_variables)
    del tape
    return loss, g


def train_warmstart_f(f_theta, X0, dC, dB, rho, F0u, optim):
    # Compute current loss and gradient w.r.t. parameters
    loss, grad_theta = grad_warmstart_f(f_theta, X0, dC, dB, rho, F0u)
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, f_theta.trainable_variables))
    return loss

def train_warmstart_g(g_theta, X0, dC, dB, rho, F0m, optim):
    # Compute current loss and gradient w.r.t. parameters
    loss, grad_theta = grad_warmstart_g(g_theta, X0, dC, dB, rho, F0m)
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, g_theta.trainable_variables))
    return loss

########################################################################################################################

def train_TFC(f_theta,g_theta, verbose):
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [5e-2, 1e-2, 5e-3])
    optimizer_g = tf.optimizers.RMSprop()
    optimizer_g.build(g_theta.trainable_variables)
    optimizer_f = tf.optimizers.RMSprop()
    optimizer_f.build(f_theta.trainable_variables)
    # Train network
    # for each sampling stage
    sampling_stages = 1
    steps_per_sample = 10001
    hist = []
    print('round-it           loss')
    for i in range(sampling_stages):
    
            # sample from the required regions
            X0, dC, dB, rho, theta, F0m, F0u = sample_room_polar(0)
            print('-----------------------------------------')
            # for a given sample, take the required number of training steps
            for j in range(steps_per_sample):
                loss = train_step_f(f_theta,g_theta, optimizer_f, X0, dC, dB, rho, theta)
                loss = train_step_g(f_theta,g_theta, optimizer_g, X0, dC, dB, rho, theta)
                hist.append(loss.numpy())
                if verbose > 0:
                    if j % 50 == 0:
                        print(' {:01d}-{:04d}          {:10.4e}'.format(i+1,j,loss))
    
    if verbose > 1:
        plt.figure(figsize=(10,10))
        plt.plot(range(len(hist[50:-1])), hist[50:-1],'k-')
        plt.xlabel('$n_{epoch}$')
        plt.title('Loss: residuals of the PDEs')
        plt.show()
    if verbose > 2:
        v = f_theta(X0) +  (1/(B-R))*((rho-B)*f_theta(dC)  + (rho-R)*(v0 - f_theta(dB)))
        m = g_theta(X0) +  (1/(B-R))*((rho-B)*g_theta(dC)  + (rho-R)*(m0 - g_theta(dB)))
        u = -tf.math.log(v)
        x = rho.numpy()*np.cos(theta)
        y = rho.numpy()*np.sin(theta)

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

        fig = plt.figure(figsize=(7,6))
        plt.scatter(x, y, c=u, cmap=cmap1, marker='.', alpha=0.3)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.colorbar()
        plt.show()
    return f_theta, g_theta

########################################################################################################################

def get_r_TFC(f_theta, g_theta, X0, dC, dB, rho, theta):
    x = X0
    with tf.GradientTape(persistent=True) as tape1:
        x_unstacked = tf.unstack(x, axis=1)
        tape1.watch(x_unstacked)

        # Using nested GradientTape for calculating higher order derivatives
        with tf.GradientTape(persistent=True) as tape2:
            # Re-stack x before passing it into f
            x_stacked = tf.stack(x_unstacked, axis=1)  # shape = (k,n)
            tape2.watch(x_stacked)
            g   = g_theta(x_stacked)
            f   = f_theta(x_stacked)
            # for the boundary evaluation of the gradient ...
            f_C = f_theta(dC)
            f_B = f_theta(dB)
            g_C = g_theta(dC)
            g_B = g_theta(dB)
            
        # Calculate gradient of m with respect to the input variables
        dg   = tape2.batch_jacobian(g, x_stacked)  # shape = (k,n)
        
        # Turn grad_m into a list of n tensors of shape (k,)
        dg_unstacked = tf.unstack(dg, axis=1)
        dg_rho   = tf.transpose(tf.gather(dg_unstacked,0,axis=2))
        dg_theta = tf.transpose(tf.gather(dg_unstacked,1,axis=2))
                
        # Same with u
        df = tape2.batch_jacobian(f, x_stacked)          
        df_unstacked = tf.unstack(df, axis=1)
        df_rho   = tf.transpose(tf.gather(df_unstacked, 0, axis=2))
        df_theta =  tf.transpose(tf.gather(df_unstacked, 1, axis=2))
        
        # ... still for the boundary evaluation of the gradient
        dg_theta_dC = tf.squeeze(tape2.batch_jacobian(g_C, x_stacked))
        dg_theta_dB = tf.squeeze(tape2.batch_jacobian(g_B, x_stacked))
        df_theta_dC = tf.squeeze(tape2.batch_jacobian(f_C, x_stacked)) 
        df_theta_dB = tf.squeeze(tape2.batch_jacobian(f_B, x_stacked)) 

    ddg_dC = tape1.batch_jacobian(dg_theta_dC, x_stacked) 
    ddg_theta_dC = tf.gather(ddg_dC,1,axis=2)
    ddg_dB = tape1.batch_jacobian(dg_theta_dB, x_stacked) 
    ddg_theta_dB = tf.gather(ddg_dB,1,axis=2)
    ddf_dC = tape1.batch_jacobian(df_theta_dC, x_stacked) 
    ddf_theta_dC = tf.gather(ddf_dC,1,axis=2)
    ddf_dB = tape1.batch_jacobian(df_theta_dB, x_stacked) 
    ddf_theta_dB = tf.gather(ddf_dB,1,axis=2)
    
    
    
    # Take 2nd derivative of each dimension separately for the laplacian
    ddg_rho = tape1.batch_jacobian(dg_rho, x_stacked) 
    ddg_rho = tf.gather(ddg_rho,0,axis=2)
    ddg_theta = tape1.batch_jacobian(dg_theta, x_stacked) 
    ddg_theta = tf.gather(ddg_theta,1,axis=2)
    ddf_rho = tape1.batch_jacobian(df_rho, x_stacked) 
    ddf_rho = tf.gather(ddf_rho,0,axis=2)
    ddf_theta = tape1.batch_jacobian(df_theta, x_stacked) 
    ddf_theta = tf.gather(ddf_theta,1,axis=2)
    

    g_dC = g_theta(dC)
    g_dB = g_theta(dB)
    
    f_dC = f_theta(dC)
    f_dB = f_theta(dB)
    
    dg_theta_dC =  tf.reshape(tf.gather(dg_theta_dC,1,axis=1),shape=ddg_rho.shape)
    dg_theta_dB =  tf.reshape(tf.gather(dg_theta_dB,1,axis=1),shape=ddg_rho.shape)
    df_theta_dC =  tf.reshape(tf.gather(df_theta_dC,1,axis=1),shape=ddg_rho.shape)
    df_theta_dB =  tf.reshape(tf.gather(df_theta_dB,1,axis=1),shape=ddg_rho.shape)

    
    resHJB = res_HJB_TFC(rho,theta,g,dg_rho,dg_theta,ddg_rho,ddg_theta,f,df_rho,df_theta,ddf_rho,ddf_theta,g_dC,g_dB,f_dC,f_dB,dg_theta_dC,dg_theta_dB,df_theta_dC,df_theta_dB,ddg_theta_dC,ddg_theta_dB,ddf_theta_dC,ddf_theta_dB)
    resFP = res_FP_TFC(rho,theta,g,dg_rho,dg_theta,ddg_rho,ddg_theta,f,df_rho,df_theta,ddf_rho,ddf_theta,g_dC,g_dB,f_dC,f_dB,dg_theta_dC,dg_theta_dB,df_theta_dC,df_theta_dB,ddg_theta_dC,ddg_theta_dB,ddf_theta_dC,ddf_theta_dB)
    return resHJB, resFP


########################################################################################################################

# residual of the Fokker Plankl
def res_FP_TFC(rho,theta,g,dg_rho,dg_theta,ddg_rho,ddg_theta,f,df_rho,df_theta,ddf_rho,ddf_theta,g_dC,g_dB,f_dC,f_dB,dg_theta_dC,dg_theta_dB,df_theta_dC,df_theta_dB,ddg_theta_dC,ddg_theta_dB,ddf_theta_dC,ddf_theta_dB):
    
    # constrained expressions for the density and the value function
    v = f +  (1/(B-R))*((rho-B)*f_dC  + (rho-R)*(v0 - f_dB))
    m = g +  (1/(B-R))*((rho-B)*g_dC  + (rho-R)*(m0 - g_dB))

    
    # derivatives for the constrained density
    dm_rho    = dg_rho    + (1/(B-R))*(g_dC + m0 - g_dB)
    dm_theta  = dg_theta  + (1/(B-R))*((rho-B)*dg_theta_dC  - (rho-R)*dg_theta_dB)
    ddm_rho   = ddg_rho
    ddm_theta = ddg_theta + (1/(B-R))*((rho-B)*ddg_theta_dC - (rho-R)*ddg_theta_dB)
    laplacian_m = ddm_rho + tf.math.reciprocal(rho)*dm_rho + tf.math.reciprocal(tf.square(rho))*ddm_theta
    
    grad_m = tf.concat([tf.math.cos(theta)*dm_rho - tf.math.reciprocal(rho)*tf.math.sin(theta)*dm_theta,
                         tf.math.sin(theta)*dm_rho + tf.math.reciprocal(rho)*tf.math.cos(theta)*dm_theta],1)
    
    # derivatives for the constrained value function
    du_rho    = tf.math.reciprocal(v)*(df_rho + (1/(B-R))*(f_dC + v0 - f_dB))
    du_theta  = tf.math.reciprocal(v)*(df_theta  + (1/(B-R))*((rho-B)*df_theta_dC  - (rho-R)*df_theta_dB))
    ddu_rho   = tf.math.reciprocal(tf.square(v))*(df_rho + (1/(B-R))*(f_dC + v0 - f_dB)) - tf.math.reciprocal(v)*ddf_rho
    ddu_theta = tf.math.reciprocal(tf.square(v))*(df_theta  + (1/(B-R))*((rho-B)*df_theta_dC  - (rho-R)*df_theta_dB)) - tf.math.reciprocal(v)*(ddf_theta + (1/(B-R))*((rho-B)*ddf_theta_dC - (rho-R)*ddf_theta_dB))
    laplacian_u = ddu_rho + tf.math.reciprocal(rho)*du_rho + tf.math.reciprocal(tf.square(rho))*ddu_theta
    
    resFP = (sigma**2)/2 * laplacian_m + tf.math.reduce_sum(s*grad_m) + tf.math.reciprocal(mu*tf.square(1+alpha*m))*(dm_rho*du_rho + tf.math.reciprocal(tf.square(rho))*dm_theta*du_theta) + m* tf.math.reciprocal(mu*(1+alpha*m))*laplacian_u
            
    
    return tf.reduce_mean(tf.square(resFP))

def res_HJB_TFC(rho,theta,g,dg_rho,dg_theta,ddg_rho,ddg_theta,f,df_rho,df_theta,ddf_rho,ddf_theta,g_dC,g_dB,f_dC,f_dB,dg_theta_dC,dg_theta_dB,df_theta_dC,df_theta_dB,ddg_theta_dC,ddg_theta_dB,ddf_theta_dC,ddf_theta_dB):
    
    # constrained expressions for the density and the value function
    v = f +  (1/(B-R))*((rho-B)*f_dC  + (rho-R)*(v0 - f_dB))
    m = g +  (1/(B-R))*((rho-B)*g_dC  + (rho-R)*(m0 - g_dB))
    u = -tf.math.log(v)
    
    
    # derivatives for the constrained value function
    du_rho    = tf.math.reciprocal(v)*(df_rho + (1/(B-R))*(f_dC + v0 - f_dB))
    du_theta  = tf.math.reciprocal(v)*(df_theta  + (1/(B-R))*((rho-B)*df_theta_dC  - (rho-R)*df_theta_dB))    
    ddu_rho   = tf.math.reciprocal(tf.square(v))*(df_rho + (1/(B-R))*(f_dC + v0 - f_dB)) - tf.math.reciprocal(v)*ddf_rho
    
    ddu_theta = tf.math.reciprocal(tf.square(v))*(df_theta  + (1/(B-R))*((rho-B)*df_theta_dC  - (rho-R)*df_theta_dB)) - tf.math.reciprocal(v)*(ddf_theta + (1/(B-R))*((rho-B)*ddf_theta_dC - (rho-R)*ddf_theta_dB))
    laplacian_u = ddu_rho + tf.math.reciprocal(rho)*du_rho + tf.math.reciprocal(tf.square(rho))*ddu_theta
    
    grad_u = tf.concat([tf.math.cos(theta)*du_rho - tf.math.reciprocal(rho)*tf.math.sin(theta)*du_theta,
                         tf.math.sin(theta)*du_rho + tf.math.reciprocal(rho)*tf.math.cos(theta)*du_theta], 1)
    
    resHJB = (sigma**2)/2 * laplacian_u - tf.math.reciprocal(2*mu*(1+alpha*m))*(tf.square(du_rho)+tf.math.reciprocal(tf.square(rho))*tf.square(du_theta)) - l - tf.math.reduce_sum(s*grad_u) - gamma*u - V_polar(m, rho)
    
    
    return tf.reduce_mean(tf.square(resHJB))
########################################################################################################################

def compute_loss_TFC(f_theta, g_theta, X0, dC, dB, rho, theta):

    r_u, r_m = get_r_TFC(f_theta, g_theta, X0, dC, dB, rho, theta)
    loss_m = r_m 
    loss_u = r_u 
    return loss_u+loss_m


def get_grad_TFC(f_theta, g_theta, X0, dC, dB, rho, theta, target):
    if target == 'FP':
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(g_theta.trainable_variables)
            loss = compute_loss_TFC(f_theta, g_theta, X0, dC, dB, rho, theta)

        g = tape.gradient(loss, g_theta.trainable_variables)
        del tape
        return loss, g
    elif target == 'HJB':
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(f_theta.trainable_variables)
            loss = compute_loss_TFC(f_theta, g_theta, X0, dC, dB, rho, theta)

        g = tape.gradient(loss, f_theta.trainable_variables)
        del tape
        return loss, g

# Define one training step as a TensorFlow function to increase speed of training
@tf.function
def train_step_f(f,g, optim, X0, dC, dB, rho, theta):
    # Compute current loss and gradient w.r.t. parameters
    loss_f, grad_theta_f = get_grad_TFC(f, g, X0, dC, dB, rho, theta, 'HJB')
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta_f, f.trainable_variables))
    return loss_f

@tf.function
def train_step_g(f,g, optim, X0, dC, dB, rho, theta):
    # Compute current loss and gradient w.r.t. parameters
    loss_g, grad_theta_g = get_grad_TFC(f, g, X0, dC, dB, rho, theta, 'FP')
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta_g, g.trainable_variables))
    return loss_g

