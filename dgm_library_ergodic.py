# authors: Sara Bicego, Matteo Butano
# emails: s.bicego21@imperial.ac.uk, matteo.butano@universite-paris-saclay.fr

from DGM import  DGMNet
import pandas as pd
import tensorflow as tf
from keras import backend as K
import numpy as np
import random
import os
import json


class neural_mfg:
    
    def __init__(self):
        
        with open('dgm_config.json') as f:
            var = json.loads(f.read())
        
        self.self.DTYPE = 'float32'
       
        # MFG parameters 
        self.xi   = var['xi']
        self.c_s  = var['c_s']
        self.mu   = var['mu']
        self. m_0 = var['m_0']
        self.g     = -(2*self.xi**2)/self.m_0
        self.sigma = np.sqrt(2*self.xi*self.c_s)
        self.l     = -self.g*self.m_0
        self.u_b   = -self.mu*self.sigma**2*np.log(np.sqrt(self.m_0))
        self.R     = 0.37
        self.s     =  tf.constant([0, -var['s']],  dtype=self.DTYPE, shape=(1, 2))
        self.V = var['V']
        
        # NN parameters
        self.training_steps = var['training_steps']
        
        
        # Room definition 
        self.lx = var['lx']
        self.ly = var['ly']
        self.N_b = var['Nb']
        self.N_in = var['Ns']
        
        
        # Seed value
        self.seed_value = 0
        os.environ['PYTHONHASHSEED'] = str(self.seed_value)
    
        # Initialize random variables for repoducibility
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        tf.random.set_seed(self.seed_value)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=0,
                                   inter_op_parallelism_threads=0)
    
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        K.set_session(sess)
        
        # Set data type
        tf.keras.backend.set_floatx(self.DTYPE)
        
        self.phi_theta = DGMNet(RNN_layers = var['RNN_layers'], FNN_layers = var['FNN_layers'], 
                             nodes_per_layer = var['nodes_per_layer'], activation = var['tanh'])
        self.gamma_theta = DGMNet(RNN_layers = 0, FNN_layers=1, nodes_per_layer=10,activation="tanh")
        
        self.X_b, self.X_in, self.X_out = self.sample_room()
    
        return
    
    
    
    # Potential V entering the HJB equation. The value V_const is a parameter defined above
    def V(self,phi,gamma,points):
        
        U0 = np.zeros(shape = (points.shape[0],1))
        
        for i in range(points.shape[0]):
        
            if tf.less_equal(tf.norm(points[i],'euclidean'),self.R): # for points in the cilinder
                U0[i] = self.V_const   # we have higher cost
        
        U0 = tf.convert_to_tensor(U0, dtype=self.DTYPE)
        
        return self.g * tf.multiply(phi,gamma) + U0 # formula for the potential from reference  
  
    # Here we sample the points on which we will train the NN  
    def sample_room(self):
        
        # Lower bounds
        lb = tf.constant([-self.lx, -self.ly], dtype=self.DTYPE)
        # Upper bounds
        ub = tf.constant([self.lx, self.ly], dtype=self.DTYPE)
    
        # Draw uniform sample points for data in the domain
        x_room = tf.random.uniform((self.Ns, 1), lb[0], ub[0], dtype=self.DTYPE)
        y_room = tf.random.uniform((self.Ns, 1), lb[1], ub[1], dtype=self.DTYPE)
        X_room = tf.concat([x_room, y_room], axis=1)
        
        # Divide between points inside and outside the cylinder
        points_in = tf.where(tf.norm(X_room, axis=1) <= self.R)
        points_out = tf.where(tf.norm(X_room, axis=1) > self.R)
        X_in = tf.gather(X_room, points_in)
        X_out = tf.gather(X_room, points_out)
        X_in = tf.squeeze(X_in)
        X_out = tf.squeeze(X_out)
    
        # Boundary data (square - outside walls)
        x_b1 = lb[0] + (ub[0] - lb[0]) * tf.keras.backend.random_bernoulli((self.Nb, 1), 0.5, dtype=self.DTYPE)
        y_b1 = tf.random.uniform((self.Nb, 1), lb[1], ub[1], dtype=self.DTYPE)
        y_b2 = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((self.Nb, 1), 0.5, dtype=self.DTYPE)
        x_b2 = tf.random.uniform((self.Nb, 1), lb[0], ub[0], dtype=self.DTYPE)
        x_b = tf.concat([x_b1, x_b2], axis=0)
        y_b = tf.concat([y_b1, y_b2], axis=0)
        X_b = tf.concat([x_b, y_b], axis=1)

        
        return pd.DataFrame(X_b.numpy()), pd.DataFrame(X_in.numpy()), pd.DataFrame(X_out.numpy())
    
    
    def train(self):
        
        opt_Phi = tf.optimizers.Adam()
        opt_Gamma = tf.optimizers.Adam()
    
        # for a given sample, take the prescribed number of training steps
        for step in range(self.steps_per_sample):
            
            loss_Phi, r_Phi, r_Gamma, m_bRoom, m_Cyl  = self.train_step_Phi(self.phi_theta,self.gamma_theta, opt_Phi, self.X_b, self.X_s, self.X_c)
            loss_Gamma, r_Phi, r_Gamma, m_bRoom, m_Cyl = self.train_step_Gamma(self.phi_theta,self.gamma_theta, opt_Gamma, self.X_b, self.X_s, self.X_c)
            
            if step % 50 == 0:
                print('Step {:04d} Loss Phi = {:10.4e}'.format(step,loss_Phi))
                print('Step {:04d} Loss Gamma = {:10.4e}'.format(step,loss_Gamma))
                print('r_Phi=' + str(r_Phi.numpy()) + ' r_Gamma=' + str(r_Gamma.numpy()) + 'boundary=' + str(m_bRoom.numpy()))

        return Phi_theta,Gamma_theta
    
    
    ########################################################################################################################
    def get_derivatives(self,f_theta, x): # function that computes the derivatives using automatic differentiation
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
            df_unstacked = tf.unstack(grad_f, axis=2)
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
        term1 = tf.math.scalar_mul(mu*sigma**2,tf.reduce_sum(tf.multiply(s, Gamma_x),1))
        term2 = tf.math.scalar_mul(((mu*sigma**4)/2),Gamma_xx)
        term_pot = tf.multiply(V(Gamma,Phi,points),Gamma)
        term_log = 0#gamma*mu*sigma**2*tf.multiply(Gamma,tf.math.log(Phi))
        
        resFP = tf.math.scalar_mul(l,Gamma) + term1  +term2 +term_pot + term_log
        #print('calcolo residuo Gamma')
        return tf.norm(resFP)
    
    # residual of the HJB
    def residual_Phi(points, Gamma, Gamma_x, Gamma_xx, Phi, Phi_x, Phi_xx):
        term1 = -tf.math.scalar_mul(mu*sigma**2,tf.reduce_sum(tf.multiply(s, Phi_x),1))
        term2 = tf.math.scalar_mul(((mu*sigma**4)/2),Phi_xx)
        term_pot = tf.multiply(V(Gamma,Phi,points),Phi)
        term_log = 0#-gamma*mu*sigma**2*tf.multiply(Phi,tf.math.log(Phi))
        
        resHJB = tf.math.scalar_mul(l,Phi) + term1  +term2 +term_pot + term_log
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
        return r_Phi + r_Gamma + m_bRoom + m_Cyl, r_Phi, r_Gamma, m_bRoom, m_Cyl# + mass_conservation
    
    
    ########################################################################################################################
    
    # gradient of the loss function with respect to the unknown variables in the model, also called `trainable variables`
    def get_grad(Phi,Gamma, X_b, X_s, X_c, target):
        if target == 'Gamma':
            with tf.GradientTape(persistent=True) as tape:
                # This tape is for derivatives with
                # respect to trainable variables
                tape.watch(Gamma.trainable_variables)
                loss, r_Phi, r_Gamma, m_bRoom, m_Cyl = compute_loss(Phi,Gamma, X_b, X_s, X_c)
    
            g = tape.gradient(loss, Gamma.trainable_variables)
            del tape
            return loss, g, r_Phi, r_Gamma, m_bRoom, m_Cyl
        elif target == 'Phi':
            with tf.GradientTape(persistent=True) as tape:
                # This tape is for derivatives with
                # respect to trainable variables
                tape.watch(Phi.trainable_variables)
                loss, r_Phi, r_Gamma, m_bRoom, m_Cyl = compute_loss(Phi,Gamma, X_b, X_s, X_c)
    
            g = tape.gradient(loss, Phi.trainable_variables)
            del tape
            return loss, g, r_Phi, r_Gamma, m_bRoom, m_Cyl
    ########################################################################################################################
    
    # Define one training step as a TensorFlow function to increase speed of training
    @tf.function
    def train_step_Phi(Phi,Gamma, optim, X_b, X_s, X_c):
        # Compute current loss and gradient w.r.t. parameters
        loss_Phi, grad_theta_Phi, r_Phi, r_Gamma, m_bRoom, m_Cyl = get_grad(Phi,Gamma, X_b, X_s, X_c, 'Phi')
        # Perform gradient descent step
        optim.apply_gradients(zip(grad_theta_Phi, Phi.trainable_variables))
        return loss_Phi,r_Phi, r_Gamma, m_bRoom, m_Cyl
    
    @tf.function
    def train_step_Gamma(Phi,Gamma, optim, X_b, X_s, X_c):
        # Compute current loss and gradient w.r.t. parameters
        loss_Gamma, grad_theta_Gamma, r_Phi, r_Gamma, m_bRoom, m_Cyl = get_grad(Phi,Gamma, X_b, X_s, X_c, 'Gamma')
        # Perform gradient descent step
        optim.apply_gradients(zip(grad_theta_Gamma, Gamma.trainable_variables))
        return loss_Gamma,r_Phi, r_Gamma, m_bRoom, m_Cyl
    
 
    def warmstart(self,phi_IC,gamma_IC,IC_points):
        
        optimizer = tf.optimizers.Adam()
        points = IC_points

        for step in range(self.training_steps):
            
            # Compute loss for phi and gamma
            phi_pred = self.phi_theta(points)
            gamma_pred = self.gamma_theta(points)
            phi_loss = tf.norm(phi_pred - phi_IC)
            gamma_loss = tf.norm(gamma_pred - gamma_IC)
            
            # Compute gradient wrt variables for phi and gamma
            
            with tf.GradientTape(persistent=True) as phi_tape:
                phi_vars = self.phi_theta.trainable_variables
                phi_tape.watch(phi_vars)
        
            phi_grad = phi_tape.gradient(phi_loss,phi_vars)
            del phi_tape
            
            with tf.GradientTape(persistent=True) as gamma_tape:
                gamma_vars = self.gamma_theta.trainable_variables
                gamma_tape.watch(gamma_vars)
        
            gamma_grad = gamma_tape.gradient(gamma_loss,gamma_vars)
            del gamma_tape
            
            
            optimizer.apply_gradients(zip(phi_grad, self.phi_theta.trainable_variables))
            optimizer.apply_gradients(zip(gamma_grad, self.gamma_theta.trainable_variables))
                                      
            if step % 50 == 0:
                print('Warmstart phi: step {:04d}, loss = {:10.8e}'.format(step, phi_loss))
                print('Warmstart gamma: step {:04d}, loss = {:10.8e}'.format(step, gamma_loss))
       