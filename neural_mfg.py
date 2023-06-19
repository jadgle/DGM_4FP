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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime


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
        self.var = var
       
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
        
        # Initial total mass
        self.total_mass = tf.constant(self.m_0*(2*self.lx)*(2*self.ly),dtype=self.DTYPE)
        
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
        
        self.all_pts = tf.constant(tf.concat([self.X_out,self.X_in,self.X_b],axis = 0))
        
        self.history = []
    
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
        
        U0[np.sqrt(all_pts[:,0]**2 + all_pts[:,1]**2) < self.R] = self.pot
        
        U0 = tf.constant(U0, dtype=self.DTYPE)
        
        mean_field = tf.math.scalar_mul(self.g,tf.multiply(phi,gamma))
        
        return mean_field + U0 # formula for the potential from reference  
      
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
 
    def get_loss(self,verbose):
        '''
        Computes loss function by calculating the residuals.

        Returns
        -------
        tensorflow.tensor
            Loss function sum of the different residuals. The default is True.

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
        
        res_HJB = tf.reduce_mean(self.l*phi +self.V(phi,gamma)*phi + 0.5*self.mu*self.sigma**4*lap_phi - self.mu*self.sigma**2*tf.reduce_sum(self.s*grad_phi,axis = 1))**2
    
        res_KFP = tf.reduce_mean(self.l*gamma +self.V(phi,gamma)*gamma + 0.5*self.mu*self.sigma**4*lap_gamma + self.mu*self.sigma**2*tf.reduce_sum(self.s*grad_gamma,axis = 1))**2
        
        res_b_phi = tf.reduce_mean((tf.sqrt(self.m_0) - self.phi_theta(self.X_b))**2)
        res_b_gamma = tf.reduce_mean((tf.sqrt(self.m_0) - self.gamma_theta(self.X_b))**2)
        
        res_obstacle = tf.reduce_mean(self.phi_theta(self.X_in)**2) + tf.reduce_mean(self.gamma_theta(self.X_in)**2)
       
        res_total_mass = (tf.reduce_mean(phi*gamma)*(2*self.lx)*(2*self.ly)-self.total_mass)**2
       
        if verbose: 
            print('      {:10.3e}       {:10.3e}       {:10.3e}       {:10.3e}       {:10.3e}       {:10.3e}'.format(res_HJB,res_KFP,res_b_phi,res_b_gamma,res_obstacle,res_total_mass))
        
        self.history.append([res_HJB.numpy(),res_KFP.numpy(),res_b_phi.numpy(),res_b_gamma.numpy(),res_obstacle.numpy(),res_total_mass.numpy()])
        
        return res_HJB + res_KFP + res_b_gamma + res_b_phi + res_obstacle + res_total_mass
      
    def train_step(self,f_theta,verbose):
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
            f_loss = self.get_loss(verbose)
            f_grad = f_tape.gradient(f_loss,f_vars)
        
        optimizer.apply_gradients(zip(f_grad, f_vars))
        
        return f_loss
    
    def train(self,verbose = True):
        '''
        Applies self.training_steps. 
        
        Parameters
        ----------
        verbose : Bool
            When true prints the value of each residual at each iteration.

        Returns
        -------
        None.

        '''
        
        if verbose:
            print(' #iter       res_HJB          res_KFP          res_b_phi        res_b_gamma      res_obstacle     res_total_mass')
            print('-----------------------------------------------------------------------------------------------------------------')
        
        for step in range(1,self.training_steps + 1):
            
            if verbose:
                print('{:6d}'.format(step),end="")
            
            # Train phi 
            self.train_step(self.phi_theta,verbose)
            
            if verbose:
                print('      ',end="")
            
            # Compute loss for phi and gamma
            
            self.train_step(self.gamma_theta,verbose)
    
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
    
    def warmstart_step_simple(self,f_theta):
        '''
        One step of warmstart with simple IC

        Parameters
        ----------
        f_theta : dgmnet
            The net to which apply one step of warmstart.

        Returns
        -------
        f_loss : tf.tensor
            The value of the loss after one step of ws.

        '''
        
        all_pts = tf.concat([self.X_out,self.X_in,self.X_b],axis = 0)
        
        optimizer = tf.optimizers.Adam(learning_rate = self.learning_rate)
       
        # Compute gradient wrt variables for phi and gamma
        
        with tf.GradientTape() as f_tape:
            
            f_vars = f_theta.trainable_weights
            f_tape.watch(f_vars)
            f_prediction = f_theta(all_pts)
            f_loss = tf.reduce_mean((f_prediction - tf.sqrt(self.m_0))**2)
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
        phi_loss = 1
        gamma_loss = 1
        step = 0
       
        while np.maximum(phi_loss,gamma_loss) > 10e-3:
            
            # Compute loss for phi and gamma
            
            phi_loss = self.warmstart_step(self.phi_theta,phi_IC,points_IC)
            gamma_loss = self.warmstart_step(self.gamma_theta,gamma_IC,points_IC)
           
            if step % 100 == 0:
                print('WS step {:5d}, loss phi={:10.3e}, loss gamma={:10.3e}'.format(step, phi_loss,gamma_loss))
                
            step +=1
    
    def warmstart_simple(self,verbose=True):
        '''
        Simple warmstart towards sqrt(m_0) condition.        

        Parameters
        ----------
        verbose : Bool, optional
            Shows info bout the simple warmstart. The default is True.

        Returns
        -------
        None.

        '''
        
        phi_loss = 1
        gamma_loss = 1
        step = 0
       
        while np.maximum(phi_loss,gamma_loss) > 10e-3:
            
            # Compute loss for phi and gamma
            
            phi_loss = self.warmstart_step_simple(self.phi_theta)
            gamma_loss = self.warmstart_step_simple(self.gamma_theta)
           
            if verbose:
                print('WS step {:5d}, loss phi={:10.3e}, loss gamma={:10.3e}'.format(step, phi_loss,gamma_loss))
            
            step +=1
        
     
    def draw(self):
        '''
        Draw the scatter plot of the density of pedestrians.

        Returns
        -------
        None.

        '''
        all_pts = tf.concat([self.X_out,self.X_in,self.X_b],axis = 0)
        
        m = self.gamma_theta(all_pts)*self.phi_theta(all_pts)
         
        plt.figure(figsize=(8,8))
        plt.scatter(all_pts.numpy()[:,0], all_pts.numpy()[:,1], c=m, cmap='hot_r')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.colorbar()
        plt.clim(vmin = 0)
        plt.show()     
        
    def save(self):
        '''
        Save into 'trainings' directory some info about the training. 

        Returns
        -------
        None.

        '''
        
        if not os.path.exists('./trainings'):
            os.mkdir('./trainings')
            
        current_time = datetime.datetime.now()
        dirname = current_time.strftime("%B %d, %Y %H-%M-%S")
        
        if not os.path.exists('./trainings/' + dirname):
            os.mkdir('./trainings/' + dirname)
        
        labels = ['res_HJB', 'res_KFP', 'res_b_phi', 'res_b_gamma', 'res_obstacle', 'res_total_mass']
        history = np.array(self.history)
        fig, ax = plt.subplots(nrows = 2,ncols=3,figsize = (15,10))

        for col in range(history.shape[1]):
    
            ax[col//3,col%3].plot(history[:,col])
            ax[col//3,col%3].set_title(labels[col])
    
        plt.savefig('./trainings/' + dirname + '/residuals')
        
        training = {}

        training['config'] = self.var
        training['phi_theta'] = self.phi_theta(self.all_pts).numpy().tolist()
        training['gamma_theta'] = self.gamma_theta(self.all_pts).numpy().tolist()
        training['points'] = self.all_pts.numpy().tolist()
        
        # Serializing json
        json_object = json.dumps(training, indent=4)
         
        # Writing to sample.json
        with open("./trainings/" + dirname + "/net.json", "w") as outfile:
            outfile.write(json_object)
            
        all_pts = tf.concat([self.X_out,self.X_in,self.X_b],axis = 0)
        
        m = self.gamma_theta(all_pts)*self.phi_theta(all_pts)
         
        plt.figure(figsize=(8,8))
        plt.scatter(all_pts.numpy()[:,0], all_pts.numpy()[:,1], c=m, cmap='hot_r')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.colorbar()
        plt.clim(vmin = 0)
        
        plt.savefig('./trainings/' + dirname + '/density')
        
