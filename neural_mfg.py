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
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime


class dgm_net:
    
    def __init__(self,directory=""):
        '''
        Initialise the two NNs for Phi and Theta, creates points.
        By specifying a different directory it is possible to load models with saved configurations
        Returns
        -------
        None.

        '''
        
        with open("./" + directory + "/config.json") as f:
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
        self.R     = 0.37
        self.s     =  tf.constant([0, -var['room']['s']],  dtype=self.DTYPE, shape=(1, 2))
        self.pot   = var['mfg_params']['V']
        
        self.gamma = var['mfg_params']['gamma']; # discount factor
        self.alpha = var['mfg_params']['alpha']; # conjestion operator
        if self.gamma > 0:
            self.l     = -self.g*self.m_0
        else:
            self.l = 0
        
        # NN parameters
        self.training_steps  = var['dgm_params']['training_steps']
        self.learning_rate   = var['dgm_params']['learning_rate']
        self.M               = var['dgm_params']['M']
        self.resampling_step = var['dgm_params']['resampling_step']
        self.weight_HJB      = 1 # loss weight hyperparameter (can be changed on the main script if 
                                 # the HJB residual is stuck during training). Neutralized in default 

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
        
        self.all_pts   = tf.constant(tf.concat([self.X_out,self.X_in,self.X_b],axis = 0))
        self.points_IC = np.loadtxt("IC/points.txt")
        
        self.history = []
    
        return
    
    def V(self,phi,gamma,all_pts):
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
        
        U0 = np.zeros(shape = (all_pts.shape[0],1))
        
        U0[np.sqrt(all_pts[:,0]**2 + all_pts[:,1]**2) < self.R] = self.pot
        
        U0 = tf.constant(U0, dtype=self.DTYPE)
        
        mean_field = tf.math.scalar_mul(self.g,tf.multiply(phi,gamma))
        
        return mean_field + U0 # formula for the potential from reference paper 
      
    def sample_room(self, resampling = False, N_res = None, Nb_res = None):
        '''
        Samples the training points, randomly over simulation area.

        Returns
        -------
        pandas.DataFrame, pandas.DataFrame, pandas.DataFrame
            Three dataframes containing boundary points, points inside and outside the cylinder.

        '''
        
        if resampling:
            N = N_res
            Nb = Nb_res
        else: 
            N = self.N_in
            Nb = self.N_b
        
        # Lower bounds
        lb = tf.constant([-self.lx, -self.ly], dtype=self.DTYPE)
        # Upper bounds
        ub = tf.constant([self.lx, self.ly], dtype=self.DTYPE)
    
        # Draw uniform sample points for data in the domain
        x_room = tf.random.uniform((N, 1), lb[0], ub[0], dtype=self.DTYPE)
        y_room = tf.random.uniform((N, 1), lb[1], ub[1], dtype=self.DTYPE)
        X_room = tf.concat([x_room, y_room], axis=1)
        
        # Divide between points inside and outside the cylinder
        points_in = tf.where(tf.norm(X_room, axis=1) <= self.R)
        points_out = tf.where(tf.norm(X_room, axis=1) > self.R)
        X_in = tf.gather(X_room, points_in)
        X_out = tf.gather(X_room, points_out)
        X_in = tf.squeeze(X_in)
        X_out = tf.squeeze(X_out)
    
        # Boundary data (square - outside walls)
        x_b1 = lb[0] + (ub[0] - lb[0]) * tf.keras.backend.random_bernoulli((int(Nb/2), 1), 0.5, dtype=self.DTYPE)
        y_b1 = tf.random.uniform((int(Nb/2), 1), lb[1], ub[1], dtype=self.DTYPE)
        y_b2 = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((int(Nb/2), 1), 0.5, dtype=self.DTYPE)
        x_b2 = tf.random.uniform((int(Nb/2), 1), lb[0], ub[0], dtype=self.DTYPE)
        x_b = tf.concat([x_b1, x_b2], axis=0)
        y_b = tf.concat([y_b1, y_b2], axis=0)
        X_b = tf.concat([x_b, y_b], axis=1)

        return pd.DataFrame(X_b.numpy()), pd.DataFrame(X_in.numpy()), pd.DataFrame(X_out.numpy())
    
    
    
    def resample(self):
        '''
        Residual-based adaptive refinement method with greed (RAR-G).
        
        Returns
        ------- 
        void - it changes the attributes X_b, X_in, X_out of self by appending new points.
        Such new points are selected to be the M ones performing the worst in terms of PDE residual

        '''
        X_b, X_in, X_out = self.sample_room(True,10000,200)

        n_in = X_in.shape[0]
        n_out = X_out.shape[0]
        
        res_HJB, res_KFP, _, _ = self.get_loss_terms(X_out,X_in,X_b)
        
        PDEs_residual = pd.DataFrame(res_HJB.numpy() + res_KFP.numpy())
        PDEs_residual.sort_values(by=PDEs_residual.columns[0], ascending=False)
        
        X_b_new     = [] 
        X_in_new    = []
        X_out_new   = []
        
        for i in range(self.M):
            ind = PDEs_residual.index.values[i]
            if  ind < n_out:
                X_out_new.append(X_out.iloc[ind])
            elif ind < n_in:
                X_in_new.append(X_in.iloc[ind-n_out])
            else:
                X_b_new.append(X_b.iloc[ind-n_out-n_in])
                
        X_out_new = pd.DataFrame(X_out_new)
        X_in_new = pd.DataFrame(X_in_new)
        X_b_new = pd.DataFrame(X_b_new)
        
        self.X_out = pd.concat([self.X_out, X_out_new], ignore_index=True)
        self.X_in  = pd.concat([self.X_in, X_in_new], ignore_index=True)
        self.X_b   = pd.concat([self.X_b, X_b_new], ignore_index=True)
        
        self.N_in  = self.X_in.shape[0] + self.X_out.shape[0]
        self.N_b   = self.X_b.shape[0]
        self.all_pts   = tf.constant(tf.concat([self.X_out,self.X_in,self.X_b],axis = 0))
        return
    
    def get_L2_loss(self,verbose):
        '''
        Computes L2 loss function by calculating the L2 norm of the residuals.

        Returns
        -------
        tensorflow.tensor
            Loss function sum of the different residuals. The default is True.

        '''
        res_HJB, res_KFP, res_b,res_total_mass = self.get_loss_terms(self.X_out,self.X_in,self.X_b)
        
        L2_HJB = tf.reduce_mean(res_HJB)  
        L2_KFP = tf.reduce_mean(res_KFP)
        
        L2_b = tf.reduce_mean(res_b)
        
        L_tot = self.weight_HJB*L2_HJB + L2_KFP + L2_b + res_total_mass
        if verbose: 
            print('        {:10.3e}       {:10.3e}       {:10.3e}        {:10.3e}       |  {:10.3e}'.format(L2_HJB,L2_KFP,L2_b,res_total_mass, L_tot))
        
        
        self.history.append([L_tot.numpy(),L2_HJB.numpy(),L2_KFP.numpy(),L2_b.numpy(),res_total_mass.numpy()])
        
        return L_tot
    
    
    def get_loss_terms(self,X_out,X_in,X_b):
        '''
        Computes the terms of loss function by calculating the residuals at each point.

        Returns
        -------
        tensorflow.tensors
            Loss function terms of the different residuals. The default is True.

        '''
        points = tf.Variable(tf.concat([X_out,X_in,X_b],axis = 0))
        
        # Compute gradient and laplacian for Phi
        
        with tf.GradientTape() as phi_tape_1:
            with tf.GradientTape() as phi_tape_2:
            
                phi = self.phi_theta(points)
            
            grad_phi = phi_tape_2.gradient(phi,points)
        
        jac_phi = phi_tape_1.gradient(grad_phi,points)
        lap_phi = tf.math.reduce_sum(jac_phi,axis = 1,keepdims=True)
        
        # Compute gradient and laplacian for Gamma
        
        with tf.GradientTape() as gamma_tape_1:
            with tf.GradientTape() as gamma_tape_2:
            
                gamma = self.gamma_theta(points)
            
            grad_gamma = gamma_tape_2.gradient(gamma,points)
        
        jac_gamma = gamma_tape_1.gradient(grad_gamma,points)
        lap_gamma = tf.math.reduce_sum(jac_gamma,axis = 1,keepdims=True)
        
        # Compute terms of the loss function
        
        res_HJB = (self.l*phi +self.V(phi,gamma,points)*phi + 0.5*self.mu*self.sigma**4*lap_phi - self.mu*self.sigma**2*tf.reduce_sum(self.s*grad_phi,axis = 1,keepdims=True))**2
    
        res_KFP = (self.l*gamma +self.V(phi,gamma,points)*gamma + 0.5*self.mu*self.sigma**4*lap_gamma + self.mu*self.sigma**2*tf.reduce_sum(self.s*grad_gamma,axis = 1,keepdims=True))**2
        
        res_b = (self.m_0 - self.phi_theta(X_b)*self.gamma_theta(X_b))**2
        
        res_total_mass = (tf.reduce_mean(phi*gamma)*(2*self.lx)*(2*self.ly)-self.total_mass)**2
        
        return res_HJB, res_KFP, res_b, res_total_mass
      
    def train_step(self,verbose):
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
        
        gamma_theta = self.gamma_theta
        phi_theta   = self.phi_theta
        optimizer = tf.optimizers.Adam(learning_rate = self.learning_rate)
        
        with tf.GradientTape() as f_tape:
            
            f_vars = gamma_theta.trainable_weights + gamma_theta.trainable_weights
            f_tape.watch(f_vars)
            f_loss = self.get_L2_loss(verbose)
            f_grad = f_tape.gradient(f_loss,f_vars)
        
        optimizer.apply_gradients(zip(f_grad, f_vars))
        
        return f_loss
    
    def train(self, resampling = False):
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

        print('    #iter          res_HJB          res_KFP          res_b           total_mass      |   Loss_total')
        print('----------------------------------------------------------------------------------------------------')
        print('    ',end="")
            
        # standard training (without resampling)
        if resampling:
            for step in range(1,self.training_steps + 1):

                if step % 10 == 0:
                    print('{:6d}'.format(step),end="")
                    self.train_step(True)
                    print('    ',end="")
                else:
                    self.train_step(False)
                # every m=resampling_step, we refine the dataset by RAR-G with M new points
                if step % self.resampling_step == 0:
                    self.resample()
            
            # training with resampling
        else:
            for step in range(1,self.training_steps + 1):

                if step % 10 == 0:
                    print('{:6d}'.format(step),end="")
                    # Train phi 
                    self.train_step(True)
                    print('    ',end="")
                else:
                    self.train_step(False)

                
            
    
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
    
    
    def warmstart_step_simple(self,f_theta,all_pts):
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
        
        
        optimizer = tf.optimizers.Adam(learning_rate = self.learning_rate)
        
        # Compute gradient wrt variables for phi and gamma together
        with tf.GradientTape() as f_tape:
            
            f_vars = f_theta.trainable_weights
            f_tape.watch(f_vars)
            f_prediction = f_theta(all_pts)
            f_loss = tf.reduce_mean((f_prediction - tf.sqrt(self.m_0))**2)
            f_grad = f_tape.gradient(f_loss,f_vars)
            
        optimizer.apply_gradients(zip(f_grad, f_vars))
        
        return f_loss
    
    
    def warmstart(self,phi_IC,gamma_IC):
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
            
            phi_loss = self.warmstart_step(self.phi_theta,phi_IC,self.points_IC)
            gamma_loss = self.warmstart_step(self.gamma_theta,gamma_IC,self.points_IC)

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

        while np.maximum(phi_loss,gamma_loss) > 10e-4:
            
            # Compute loss for phi and gamma
            phi_loss = self.warmstart_step_simple(self.phi_theta,self.all_pts)
            gamma_loss = self.warmstart_step_simple(self.gamma_theta,self.all_pts)

            if step % 100 == 0:
                print('WS step {:5d}, loss phi={:10.3e}, loss gamma={:10.3e}'.format(step, phi_loss, gamma_loss))
            
            step +=1
        print('WS step {:5d}, loss phi={:10.3e}, loss gamma={:10.3e}'.format(step, phi_loss,gamma_loss))
            
    def draw(self, IC_points=False, saving=False, directory=None):
        '''
        Draw the scatter plot of the density of pedestrians. Two different possible resolutions:
        - IC_points == False (default) draws the mass at the training points
        - IC_points == True draws the mass at the points of the reference finite difference solution
        Option to save in desired directory (default is just plotting)

        Returns
        -------
        None.

        '''
        plt.figure(figsize=(10,10))
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.yticks(np.arange(-2, 2.1, 2),fontsize=30)
        plt.xticks(np.arange(-2, 2.1, 2),fontsize=30)
        plt.box(False)
            
        if not IC_points:
            m = self.gamma_theta(self.all_pts)*self.phi_theta(self.all_pts)           
            plt.scatter(self.all_pts.numpy()[:,0], self.all_pts.numpy()[:,1], c=m, cmap='hot_r')
            
        else: 
            m = self.gamma_theta(self.points_IC)*self.phi_theta(self.points_IC)            
            plt.scatter(self.points_IC[:,0], self.points_IC[:,1], s=13, c=m, cmap='hot_r')
        
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=30) 
        plt.clim(vmin = 0)    
        
        plt.show()   
        if saving:
                plt.savefig('./trainings/' + directory + '/solution_m')
                
                
        
    def error_plots(self, saving=False, directory=None):
        '''
        Draw the error plots, option to save in desired directory (default is just plotting)

        "residuals"      -> plot of the different components of the loss function over training time
        "loss_in_room"   -> plot of the L2 loss in the space domain
        "dgm_vs_ref"     -> plot of the L2 distance from the finite diff. reference solution in space
        
        Returns
        -------
        None.

        '''
            
        # "residuals"         
        labels = ['res_HJB', 'res_KFP', 'res_b', 'res_total_mass']
        history = np.array(self.history)
        fig, ax = plt.subplots(nrows = 2,ncols=2,figsize = (15,10))

        for cols in range(1,history.shape[1]):
            col = cols-1
            ax[col//2,col%2].plot(history[:,cols])
            ax[col//2,col%2].set_title(labels[col])        
        plt.show()
        if saving:
            plt.savefig('./trainings/' + directory + '/residuals')
            
        # "loss_in_room"
        plt.figure(figsize=(10,10))
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.yticks(np.arange(-2, 2.1, 2),fontsize=30)
        plt.xticks(np.arange(-2, 2.1, 2),fontsize=30)
        plt.box(False)
        
        X_b, X_in, X_out = self.sample_room(True,1,1)
        X_out = self.points_IC
        points = tf.constant(tf.concat([X_out,X_in,X_b],axis = 0))
        res_HJB, res_KFP,  _, _ = self.get_loss_terms(X_b, X_in, X_out)
        loss = res_HJB + res_KFP
        plt.scatter(points.numpy()[:,0], points.numpy()[:,1], c=loss, cmap='magma_r')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=30) 
        plt.title("L2 loss in space")
        plt.show()
        if saving:
            plt.savefig('./trainings/' + directory + '/loss_in_room')
            
            
        # "dgm_vs_ref"
        plt.figure(figsize=(10,10))
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.yticks(np.arange(-2, 2.1, 2),fontsize=30)
        plt.xticks(np.arange(-2, 2.1, 2),fontsize=30)
        plt.box(False)
            
        gamma_ref = np.loadtxt("IC/gamma.txt",ndmin=2)
        phi_ref = np.loadtxt("IC/phi.txt",ndmin=2)
        m_ref = gamma_ref*phi_ref
        m_dgm = self.gamma_theta(self.points_IC)*self.phi_theta(self.points_IC) 
        discrepancy = np.absolute(m_ref-m_dgm)
        plt.scatter(self.points_IC[:,0], self.points_IC[:,1], s=13, c=discrepancy, cmap='magma_r')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=30) 
        plt.title("Discrepancy vs Finite Difference solution")
        plt.show() 
        if saving:
            plt.savefig('./trainings/' + directory + '/dgm_vs_ref')
        
        
        
    def save(self):
        '''
        Save into 'trainings' directory some info about the training. 

        Returns
        -------
        None.

        '''
        
        # creating a folder for saving the info (if doesn't exist already) depending on current time and day
        if not os.path.exists('./trainings'):
            os.mkdir('./trainings')
            
        current_time = datetime.datetime.now()
        dirname = current_time.strftime("%B %d, %Y %H-%M-%S")
        
        if not os.path.exists('./trainings/' + dirname):
            os.mkdir('./trainings/' + dirname)
        
        # plotting/saving the solution and the errors
        self.error_plots(saving=True, directory=dirname)
        self.draw(IC_points=True, saving=True, directory=dirname)
        
        # saving the network weights
        self.phi_theta.save_weights("./trainings/" + dirname +"/phi.h5")
        self.gamma_theta.save_weights("./trainings/" + dirname +"/gamma.h5")
        
        # saving the network model
        with open("./trainings/" + dirname +"/config.json", "w") as outfile:
            json.dump(self.var, outfile)
            
        # saving the loss history
        loss_history = pd.DataFrame(self.history)
        loss_history.to_csv("./trainings/" + dirname +"/history.csv")
            
            
    def load(self,directory_name):
        '''
        Load model weights from previously saved trainings. Before running this, please run
        nn = dgm_net(directory_name)
        nn.load(directory_name)

        Returns
        -------
        None.

        '''
        
        # initialize weights
        self.warmstart_step_simple(self.phi_theta,self.all_pts)
        self.warmstart_step_simple(self.gamma_theta,self.all_pts)
        # load saved weights
        self.phi_theta.load_weights("./" + directory_name + "/phi.h5")
        self.gamma_theta.load_weights("./" + directory_name + "/gamma.h5")
        
        loss_history = pd.read_csv(directory_name + "/history.csv",header='infer',index_col=0)
        self.history = loss_history.values.tolist()
        
            
        
        
        
        
