# authors: Sara Bicego, Matteo Butano
# emails: s.bicego21@imperial.ac.uk, matteo.butano@universite-paris-saclay.fr

from DGM import DGMNet
import pandas as pd
import tensorflow as tf
from keras import backend as K
import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt


class dgm_net:
    
    def __init__(self):
        '''
        Initialise the two NNs for Phi and Theta, creates points.

        Returns
        -------
        None.

        '''
        
        with open('config.json') as f:
            var = json.loads(f.read())
        
        self.DTYPE = 'float32'
       
        # MFG parameters 
        self.xi    = var['mfg_params']['xi']
        self.c_s   = var['mfg_params']['c_s']
        self.mu    = var['mfg_params']['mu']
        self.m_0  = var['room']['m_0']
        self.g     = -(2*self.xi**2)/self.m_0
        self.sigma = np.sqrt(2*self.xi*self.c_s)
        self.l     = -self.g*self.m_0
        self.R     = 0.37
        self.s     =  tf.constant([0, -var['room']['s']],  dtype=self.DTYPE, shape=(1, 2))
        self.pot     = var['mfg_params']['V']
        
        # NN parameters
        self.training_steps = var['dgm_params']['training_steps']
        self.learning_rate = var['dgm_params']['learning_rate']
        
        # Room definition 
        self.lx   = var['room']['lx']
        self.ly   = var['room']['ly']
        self.N_b  = int(var['room']['N_b'])
        self.N_in = int(var['room']['N_in'])
        
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
        
        self.phi_theta   = DGMNet(var['dgm_params']['nodes_per_layer'],
                                var['dgm_params']['RNN_layers'],
                                var['dgm_params']['FNN_layers'], 2,
                                var['dgm_params']['activation'])
        
        self.gamma_theta = DGMNet(var['dgm_params']['nodes_per_layer'],
                                var['dgm_params']['RNN_layers'],
                                var['dgm_params']['FNN_layers'], 2,
                                var['dgm_params']['activation'])
        
        self.X_b, self.X_in, self.X_out = self.sample_room()
    
        return
    
    def V(self,phi,gamma):
        '''
        Describes the potential of the cost-functional

        Parameters
        ----------
        phi : tensorflow.tensor
            Values of phi_theta over training points.
        gamma : tensorflow.tensor
            Values of gamma_theta over training points.

        Returns
        -------
        tensorflow.tensor
            Values of the potential over the training points.

        '''
        
        all_pts = tf.concat([self.X_out,self.X_in,self.X_b],axis = 0)
        
        U0 = np.zeros(shape = (self.N_b + self.N_in,1))
        
        for i in range(U0.shape[0]):
        
            if tf.less_equal(tf.norm(all_pts[i],'euclidean'),self.R): # for points in the cilinder
                U0[i] = self.pot   # we have higher cost
        
        U0 = tf.convert_to_tensor(U0, dtype=self.DTYPE)
        
        m = tf.multiply(phi,gamma)
        
        return tf.reduce_sum(tf.math.scalar_mul(self.g,m) + U0,axis = 1) # formula for the potential from reference  
      
    def sample_room(self):
        '''
        Samples the training points, randomly over simulation area.

        Returns
        -------
        pandas.DataFrame, pandas.DataFrame, pandas.DataFrame
            Three dataframes containing boundary points, points inside and outside the cylinder.

        '''
        
        # Lower bounds
        lb = tf.constant([-self.lx, -self.ly], dtype=self.DTYPE)
        # Upper bounds
        ub = tf.constant([self.lx, self.ly], dtype=self.DTYPE)
    
        # Draw uniform sample points for data in the domain
        x_room = tf.random.uniform((self.N_in, 1), lb[0], ub[0], dtype=self.DTYPE)
        y_room = tf.random.uniform((self.N_in, 1), lb[1], ub[1], dtype=self.DTYPE)
        X_room = tf.concat([x_room, y_room], axis=1)
        
        # Divide between points inside and outside the cylinder
        points_in = tf.where(tf.norm(X_room, axis=1) <= self.R)
        points_out = tf.where(tf.norm(X_room, axis=1) > self.R)
        X_in = tf.gather(X_room, points_in)
        X_out = tf.gather(X_room, points_out)
        X_in = tf.squeeze(X_in)
        X_out = tf.squeeze(X_out)
    
        # Boundary data (square - outside walls)
        x_b1 = lb[0] + (ub[0] - lb[0]) * tf.keras.backend.random_bernoulli((int(self.N_b/2), 1), 0.5, dtype=self.DTYPE)
        y_b1 = tf.random.uniform((int(self.N_b/2), 1), lb[1], ub[1], dtype=self.DTYPE)
        y_b2 = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((int(self.N_b/2), 1), 0.5, dtype=self.DTYPE)
        x_b2 = tf.random.uniform((int(self.N_b/2), 1), lb[0], ub[0], dtype=self.DTYPE)
        x_b = tf.concat([x_b1, x_b2], axis=0)
        y_b = tf.concat([y_b1, y_b2], axis=0)
        X_b = tf.concat([x_b, y_b], axis=1)

        return pd.DataFrame(X_b.numpy()), pd.DataFrame(X_in.numpy()), pd.DataFrame(X_out.numpy())
 
    def get_loss(self):
        '''
        Computes loss function by calculating the residuals.

        Returns
        -------
        tensorflow.tensor
            Loss function sum of the different residuals.

        '''
        
        all_pts = tf.Variable(tf.concat([self.X_out,self.X_in,self.X_b],axis = 0))
        
        # Compute gradient and laplacian for Phi
        
        with tf.GradientTape() as phi_tape_1:
            with tf.GradientTape() as phi_tape_2:
            
                phi = self.phi_theta(all_pts)
            
            grad_phi = phi_tape_2.gradient(phi,all_pts)
        
        jac_phi = phi_tape_1.gradient(grad_phi,all_pts)
        lap_phi = tf.math.reduce_sum(jac_phi,axis = 1)
        
        # Compute gradient and laplacian for Gamma
        
        with tf.GradientTape() as gamma_tape_1:
            with tf.GradientTape() as gamma_tape_2:
            
                gamma = self.gamma_theta(all_pts)
            
            grad_gamma = gamma_tape_2.gradient(gamma,all_pts)
        
        jac_gamma = gamma_tape_1.gradient(grad_gamma,all_pts)
        lap_gamma = tf.math.reduce_sum(jac_gamma,axis = 1)
        
        term_vel_HJB = tf.math.scalar_mul(self.mu*self.sigma**2,tf.reduce_sum(tf.multiply(self.s, grad_phi),1))
        term_lap_HJB = tf.math.scalar_mul((self.mu*self.sigma**4)/2,lap_phi)
        term_pot_HJB = tf.multiply(self.V(phi,gamma),phi[:,0])
        
        res_HJB = tf.reduce_mean((tf.math.scalar_mul(self.l,phi[:,0]) + term_pot_HJB +  term_lap_HJB - term_vel_HJB)**2)
        
        term_vel_KFP = tf.math.scalar_mul(self.mu*self.sigma**2,tf.reduce_sum(tf.multiply(self.s, grad_gamma),1))
        term_lap_KFP = tf.math.scalar_mul((self.mu*self.sigma**4)/2,lap_gamma)
        term_pot_KFP = tf.multiply(self.V(phi,gamma),gamma[:,0])
       
        res_KFP = tf.reduce_mean((tf.math.scalar_mul(self.l,gamma[:,0]) + term_pot_KFP +  term_lap_KFP + term_vel_KFP)**2)
        
        res_b_phi = tf.reduce_mean((tf.sqrt(self.m_0) - self.phi_theta(self.X_b))**2)
        res_b_gamma = tf.reduce_mean((tf.sqrt(self.m_0) - self.gamma_theta(self.X_b))**2)
        
        res_obstacle = tf.reduce_mean((self.phi_theta(self.X_in))**2) + tf.reduce_mean((self.gamma_theta(self.X_in))**2)
       
        print('res_HJB={:10.3e}, res_KFP={:10.3e}, res_b_phi={:10.3e}, res_b_gamma={:10.3e}, res_obstacle={:10.3e}'.format(res_HJB,res_KFP,res_b_phi,res_b_gamma,res_obstacle))
    
        return res_HJB + res_KFP + res_b_gamma + res_b_phi + res_obstacle
      
    def train_step(self,f_theta):
        '''
        Applies one training to the NN in input

        Parameters
        ----------
        f_theta : DGMNet
            Neural network created using the DGM package.

        Returns
        -------
        f_loss : tensorflow.tensor
            Value of the loss function.

        '''
        
        optimizer = tf.optimizers.Adam(learning_rate = self.learning_rate)
        
        with tf.GradientTape() as f_tape:
            
            f_vars = f_theta.trainable_weights
            f_tape.watch(f_vars)
            f_loss = self.get_loss()
            f_grad = f_tape.gradient(f_loss,f_vars)
        
        optimizer.apply_gradients(zip(f_grad, f_vars))
        
        return f_loss
    
    def train(self):
        '''
        Repetetively applies self.training_steps. 

        Returns
        -------
        None.

        '''
        
        for step in range(self.training_steps + 1):
            
            # Compute loss for phi and gamma
            
            phi_loss = self.train_step(self.phi_theta)
            gamma_loss = self.train_step(self.gamma_theta)
           
            if step % 1 == 0:
                print('Training step {}, loss phi={:10.3e}, loss gamma={:10.3e}'.format(step, phi_loss,gamma_loss))
     
    
    def warmstart_step(self,f_theta,f_IC,points_IC):
        '''
        Applies one step of warmstart. 

        Parameters
        ----------
        f_theta : DGMNet
            Either Phi_theta or Gamma_theta.
        f_IC : numpy.array
            Values of the exact solution for Phi or Gamma reshaped as (nx*ny,1).
        points_IC : numpy.array
            Coordinates of the exact solution's grid reshaped as (nx*ny,1).

        Returns
        -------
        f_loss : tensorflow.tensor
            MSE of the difference between NNs and the exact solution.

        '''
        
        #all_pts = tf.concat([self.X_out,self.X_in,self.X_b],axis = 0)
        
        optimizer = tf.optimizers.Adam(learning_rate = self.learning_rate)
        
        f_IC   = pd.DataFrame(f_IC).astype(dtype = self.DTYPE)
        points_IC   = pd.DataFrame(points_IC).astype(dtype = self.DTYPE)
        
        # Compute gradient wrt variables for phi and gamma
        
        with tf.GradientTape() as f_tape:
            
            f_vars = f_theta.trainable_weights
            f_tape.watch(f_vars)
            f_prediction = f_theta(points_IC)
            f_loss = tf.reduce_mean((f_prediction - f_IC)**2)
            f_grad = f_tape.gradient(f_loss,f_vars)
            
        optimizer.apply_gradients(zip(f_grad, f_vars))
        
        return f_loss
 
    def warmstart(self,phi_IC,gamma_IC,points_IC):
        '''
        Applies self.training_steps of warmstart

        Parameters
        ----------
        phi_IC : numpy.array
            Values of the exact solution for Phi reshaped as (nx*ny,1).
        gamma_IC : numpy.array
            Values of the exact solution for Gamma reshaped as (nx*ny,1)..
        points_IC : numpy.array
            Coordinates of the exact solution's grid reshaped as (nx*ny,1).

        Returns
        -------
        None.

        '''
        
        for step in range(self.training_steps + 1):
            
            # Compute loss for phi and gamma
            
            phi_loss = self.warmstart_step(self.phi_theta,phi_IC,points_IC)
            gamma_loss = self.warmstart_step(self.gamma_theta,gamma_IC,points_IC)
           
            if step % 100 == 0:
                print('WS step {}, loss phi={:10.3e}, loss gamma={:10.3e}'.format(step, phi_loss,gamma_loss))
     
    def draw(self):
        '''
        Draw the scatter plot of the density of pedestrians.

        Returns
        -------
        None.

        '''
        all_pts = tf.concat([self.X_out,self.X_in,self.X_b],axis = 0)
        
        m = self.gamma_theta(all_pts)*self.phi_theta(all_pts)
         
        plt.figure(figsize=(10,10))
        plt.scatter(all_pts.numpy()[:,0], all_pts.numpy()[:,1], c=m, cmap='hot_r')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.colorbar()
        plt.show()     
