from numba import jit
import scipy
import numpy as np
import csv


import GPy
import GPyOpt

from GPyOpt.acquisitions.base import AcquisitionBase
from . import ep_minmax
from GPyOpt.models.gpmodel import GPModel

import ray

def sample_from_GP(model,sample_size,bounds,x1_dim,slices,x1_t=[None,None]):
    x2_dim=len(slices)
    if any(np.array(x1_t)==None):
        x1_t = np.linspace(bounds[0][0], bounds[0][1], x1_dim)
    if not bounds[1][0] is None:
        x2_t = slices
    X1_t, X2_t = np.meshgrid(x1_t, x2_t)
    X_t = np.hstack((X1_t.reshape(x2_dim*x1_dim,1),X2_t.reshape(x2_dim*x1_dim,1)))
    
    sim=model.posterior_samples_f(X_t,size=sample_size)
    return sim,X_t,x1_t,x2_t

ray.init()

@ray.remote(num_return_vals=4)
def core_calc(mu, var, nr_of_sim_parameters, proposed_worstcase, idx2coords):
    logP_t, dlogPdMu_t, dlogPdSigma_t, logPdMudMu_t = ep_minmax.joint_min_max(mu, var, nr_of_sim_parameters=nr_of_sim_parameters, proposed_worstcase=proposed_worstcase, idx2coords=idx2coords,with_derivatives=True)
    return logP_t, dlogPdMu_t, dlogPdSigma_t, logPdMudMu_t


     
    
    

class AcquisitionEntropySearch_MinMax(AcquisitionBase):
        def __init__(self, model, space, slices, sampler, optimizer=None, cost_withGradients=None,num_samples=22, num_representer_points=50,proposal_function=None, burn_in_steps=50,sample_size=20,ignor_previous_best_candidat=False):
            """
            This class is based on AcquisitionEntropySearch from gpyOpt
            In a nutshell entropy search approximates the
            distribution of the global minimum and tries to decrease its
            entropy. See this paper for more details:
                   Hennig and C. J. Schuler
                   Entropy search for information-efficient global optimization
                   Journal of Machine Learning Research, 13, 2012

            Current implementation does not provide analytical gradients, thus
            DIRECT optimizer is preferred over gradient descent for this acquisition

            Parameters
            ----------
            :param model: GPyOpt class of model
            :param space: GPyOpt class of Design_space
            :param slices: experiment parameters
            :param sampler: mcmc sampler for representer points, an instance of util.McmcSampler
            :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
            :param cost_withGradients: function
            :param num_samples: integer determining how many samples to draw for each candidate input
            :param num_representer_points: integer determining how many representer points to sample
            :param proposal_function: Function that defines an unnormalized log proposal measure from which to sample the representer. The default is expected improvement.
            :param burn_in_steps: integer that defines the number of burn-in steps when sampling the representer points
            """
            self.ignor_previous_best_candidat=ignor_previous_best_candidat

            self.minmax_x1=[]
            
            self.sample_size=sample_size

            if not isinstance(model, GPModel):
                raise RuntimeError("The current entropy search implementation supports only GPModel as model")

            self.optimizer = optimizer
            self.analytical_gradient_prediction = False
            AcquisitionBase.__init__(self, model, space, optimizer, cost_withGradients=cost_withGradients)
            
            self.input_dim = self.space.input_dim()

            if not self.input_dim>1:
                raise RuntimeError("The model has to be at least 2 dimensional")

            self.num_repr_points = num_representer_points
            self.burn_in_steps = burn_in_steps
            self.sampler = sampler

            self.slices=slices
            self.nr_of_exp=len(slices)

            self.bounds=space.get_bounds()

            # (unnormalized) density from which to sample representer points
            self.proposal_function = proposal_function
            if self.proposal_function is None:
                bounds = space.get_bounds()
                mi = np.zeros(len(bounds))
                ma = np.zeros(len(bounds))
                for d in range(len(bounds)):
                    mi[d] = bounds[d][0]
                    ma[d] = bounds[d][1]

            #We simulate 10 functions and use the minimum over all worst case functions as density
            
            self.sample_size_for_prop=50
            self.x1_dim_for_prop=500
            self.x2_dim_for_prop=self.nr_of_exp
            
            if not self.model.model is None:
                self.sims_for_prop,self.X_t_for_prop,self.x1_t_for_prop,self.x2_t_for_prop=sample_from_GP(
                    model=self.model.model,
                    sample_size=self.sample_size_for_prop,
                    bounds=self.bounds,
                    x1_dim=self.x1_dim_for_prop,
                    slices=self.slices)
            else:
                print('ERROR mim_mac_entropy_acquisition.py prop_func defined without model!')
                self.sims_for_prop=np.zeros(self.sample_size_for_prop)
            

            def prop_func(x):
                if len(x.shape) != 1:
                    raise ValueError("Expected a vector, received a matrix of shape {}".format(x.shape))
                if np.all(np.all(mi <= x)) and np.all(np.all(x <= ma)):
                    
                    f_t_max_s=[]
                    for i in range(self.sample_size_for_prop):
                        Y_t=self.sims_for_prop[:,0,i]
                        f_t=Y_t.reshape((self.x2_dim_for_prop,self.x1_dim_for_prop))
                        f_t_max_s.append(np.max(f_t,axis=0))

                    f_min=np.min(np.array(f_t_max_s),axis=0)

                    #we map x to the closest value in x1_t
                    idx_for_x=np.argmin(np.abs(self.x1_t_for_prop-x[0]))

                    
                    
                    return -1*(f_min[idx_for_x]-np.min(f_min))**2
                else:
                    return np.NINF
            
            self.proposal_function = prop_func

            # This is used later to calculate derivative of the stochastic part for the loss function
            # Derived following Ito's Lemma, see for example https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma
            self.W = scipy.stats.norm.ppf(np.linspace(1. / (num_samples + 1),
                                                      1 - 1. / (num_samples + 1),
                                                      num_samples))[np.newaxis, :]
            # Initialize parameters to lazily compute them once needed
            self.repr_points = None
            self.repr_points_log = None
            self.logP = None

            #To increas the chance of keeping the current best candidat for the minmax point in the self.repr_points
            self.best_candidat_for_minmax_from_previous_run=None

        def _update(self):
            if not (self.repr_points is None):
                self.best_candidat_for_minmax_from_previous_run=self.repr_points[np.argmax(self.logP)]
            self.logP = None

        def plot_logP(self,filename=None):
            
            if not filename is None:
                plot_logP(filename=filename,logP=self.logP,repr_points=self.rer_points_sim_par,bounds=self.bounds)

        #@jit(nopython=True)
        def _update_parameters(self):
            """
            Update parameters of the acquisition required to evaluate the function. In particular:
            * Sample representer points repr_points
            * Compute their log values repr_points_log
            * Compute belief locations logP
            """
            
            #representative points for the simulation parameters
            self.sims_for_prop,self.X_t_for_prop,self.x1_t_for_prop,self.x2_t_for_prop=sample_from_GP(
                model=self.model.model,
                sample_size=self.sample_size_for_prop,
                bounds=self.bounds,
                x1_dim=self.x1_dim_for_prop,
                slices=self.slices)
            

            attempt=0
            dist_to_last_best=100
            if self.ignor_previous_best_candidat:
                self.repr_points, self.repr_points_log = self.sampler.get_samples(self.num_repr_points, self.proposal_function, self.burn_in_steps)
            else:
                while dist_to_last_best>0.1 and attempt<100:
                    self.repr_points, self.repr_points_log = self.sampler.get_samples(self.num_repr_points, self.proposal_function, self.burn_in_steps)
                    attempt=attempt+1
                    if self.best_candidat_for_minmax_from_previous_run is None:
                        attempt=500
                    else:
                        x1_length=np.abs(self.bounds[0][1]-self.bounds[0][0])
                        dist_to_last_best=np.min(
                            np.abs(self.repr_points-self.best_candidat_for_minmax_from_previous_run))/x1_length
                

            if np.any(np.isnan(self.repr_points_log)) or np.any(np.isposinf(self.repr_points_log)):
                raise RuntimeError("Sampler generated representer points with invalid log values: {}".format(self.repr_points_log))
                
            # Removing representer points that have 0 probability of being the minimum (corresponding to log probability being minus infinity)
            idx_to_remove = np.where(np.isneginf(self.repr_points_log))[0]
            if len(idx_to_remove) > 0:
                idx = list(set(range(self.num_repr_points)) - set(idx_to_remove))
                self.repr_points = self.repr_points[idx, :]
                self.repr_points_log = self.repr_points_log[idx]
                
            # Add experiments to the representative points and determine argmax proposals
            x1_dim=len(self.repr_points)
            
            sample_size=self.sample_size
            #self.repr_points contains 2D points, we care only about the simulation paramters
            self.rer_points_sim_par,x2_c=list(zip(*self.repr_points.tolist()))
            
            sim,self.repr_points_and_exps,x1_repr,x2_repr=sample_from_GP(
                self.model.model,
                sample_size=sample_size,
                bounds=self.bounds,
                x1_dim=x1_dim,
                slices=self.slices,
                x1_t=self.rer_points_sim_par)

            #self.repr_points_log is extended to the experiments
            self.repr_points_log_per_sim=self.repr_points_log
            X1,X2=np.meshgrid(self.repr_points_log,np.ones(self.nr_of_exp)/self.nr_of_exp)
            self.repr_points_log=np.array([x*y for x,y in np.hstack((X1.reshape(-1,1),X2.reshape(-1,1)))])
            
            arg_max_proposals=[]
            for i in range(sample_size):
                Y_t=sim[:,0,i]
                f_t=Y_t.reshape((self.nr_of_exp,len(self.rer_points_sim_par)))
                f_t_arg_max_t=np.argmax(f_t,axis=0)#np.array(x2_dim*[np.argmax(f_t,axis=0)])
                f_t_arg_max=np.zeros_like(f_t)
                for z,worst_case_idx in enumerate(f_t_arg_max_t):
                    f_t_arg_max[worst_case_idx,z]=1
                    
                arg_max_proposals.append(f_t_arg_max)
            
            # We predict with the noise as we need to make sure that var is indeed positive definite.
            mu, _ = self.model.predict(self.repr_points_and_exps)
            # we need a vector
            mu = np.ndarray.flatten(mu)
            var = self.model.predict_covariance(self.repr_points_and_exps)
            
            logP_s=[]
            dlogPdMu_s=[]
            dlogPdSigma_s=[]
            dlogPdMudMu_s=[]
            
            for h in arg_max_proposals:
                #Not parallel: logP_t, dlogPdMu_t, dlogPdSigma_t, logPdMudMu_t = ep_minmax.joint_min_max(mu, var, nr_of_sim_parameters=x1_dim, proposed_worstcase=np.ndarray.flatten(h), idx2coords=np.ndarray.flatten(self.repr_points_and_exps),with_derivatives=True)
                logP_t, dlogPdMu_t, dlogPdSigma_t, logPdMudMu_t= core_calc.remote(mu,
                                 var,
                                 x1_dim,
                                 np.ndarray.flatten(h),
                                 np.ndarray.flatten(self.repr_points_and_exps))
                
                logP_s.append(logP_t)
                dlogPdMu_s.append(dlogPdMu_t)
                dlogPdSigma_s.append(dlogPdSigma_t)
                dlogPdMudMu_s.append(logPdMudMu_t)

            #Block until the results have finished and get the results.
            logP_s=ray.get(logP_s)
            dlogPdMu_s=ray.get(dlogPdMu_s)
            dlogPdSigma_s=ray.get(dlogPdSigma_s)
            dlogPdMudMu_s=ray.get(dlogPdMudMu_s)
            
            
            self.logP=np.log(np.mean(np.exp(logP_s),axis=0))
            self.dlogPdMu=np.mean(dlogPdMu_s,axis=0)
            self.dlogPdSigma =np.mean(dlogPdSigma_s,axis=0)
            self.dlogPdMudMu =np.mean(dlogPdMudMu_s,axis=0)
                
            # add a second dimension to the array
            self.logP = np.reshape(self.logP, (self.logP.shape[0], 1))

            if not self.rer_points_sim_par is None:
                self.minmax_x1.append(self.rer_points_sim_par[np.argmax(self.logP)])
                                                                                                                                                                                    
        def _required_parameters_initialized(self):
            """
            Checks if all required parameters are initialized.
            """
            return not (self.repr_points is None or self.repr_points_log is None or self.logP is None)
            
        @staticmethod
        def fromConfig(model, space, optimizer, cost_withGradients, config):
            raise NotImplementedError("Not implemented")

        def _compute_acq(self, x):
            # Naming of local variables here follows that in the paper
            
            if x.shape[1] != self.input_dim:
                message = "Dimensionality mismatch: x should be of size {}, but is of size {}".format(self.input_dim, x.shape[1])
                raise ValueError(message)
            
            if not self._required_parameters_initialized():
                self._update_parameters()

            
            if x.shape[0] > 1:
                results = np.zeros([x.shape[0], 1])
                for j in range(x.shape[0]):
                    results[j] = self._compute_acq(x[[j], :])
                return results
                
            # Number of belief locations
            N = self.repr_points_and_exps.shape[0]#self.logP.size
            
            # Evaluate innovation, these are gradients of mean and variance of the repr points wrt x
            # see method for more details
            dMdx, dVdx = self._innovations(x)
            
            # The transpose operator is there to make the array indexing equivalent to matlab's
            dVdx = dVdx[np.triu(np.ones((N, N))).T.astype(bool), np.newaxis]
            
            dMdx_squared = dMdx.dot(dMdx.T)
            trace_term = np.sum(np.sum(np.multiply(self.dlogPdMudMu, np.reshape(dMdx_squared, (1, dMdx_squared.shape[0], dMdx_squared.shape[1]))), 2), 1)[:, np.newaxis]
                
            # Deterministic part of change:
            deterministic_change = self.dlogPdSigma.dot(dVdx) + 0.5 * trace_term
            # Stochastic part of change:
            stochastic_change = (self.dlogPdMu.dot(dMdx)).dot(self.W)
            # Predicted new logP:
            predicted_logP = np.add(self.logP + deterministic_change, stochastic_change)
            max_predicted_logP = np.amax(predicted_logP, axis=0)
            
            # normalize predictions
            max_diff = max_predicted_logP + np.log(np.sum(np.exp(predicted_logP - max_predicted_logP), axis=0))
            lselP = max_predicted_logP if np.any(np.isinf(max_diff)) else max_diff
            predicted_logP = np.subtract(predicted_logP, lselP)

            
            
            # We maximize the information gain
            dHp = np.sum(np.multiply(np.exp(predicted_logP), np.add(predicted_logP, self.repr_points_log_per_sim)), axis=0)# predicted_logP-> dim0=simulation parameters, dim1= runs from W
            
            
            dH = np.mean(dHp)
            
            return dH # there is another minus in the public function

        def _compute_acq_withGradients(self, x):
            raise NotImplementedError("Analytic derivatives are not supported.")

        def _innovations(self, x):
            """
            Computes the expected change in mean and variance at the representer 
            points (cf. Section 2.4 in the paper). 
            
            
            :param x: candidate for which to compute the expected change in the GP
            :type x: np.array(1, input_dim)
            
            :return: innovation of mean (without samples) and variance at the representer points
            :rtype: (np.array(num_repr_points, 1), np.array(num_repr_points, num_repr_points))
            
            """
            
            
            '''
            The notation differs from the paper. The representer points
            play the role of x*, the test input x is X. The Omega term is applied
            in the calling function _compute_acq. Another difference is that we
            ignore the noise as in the original Matlab implementation:
            https://github.com/ProbabilisticNumerics/entropy-search/blob/master/matlab/GP_innovation_local.m
            '''
            
            # Get the standard deviation at x without noise
            _, stdev_x = self.model.predict(x, with_noise=False)
            
            # Compute the variance between the test point x and the representer points
            sigma_x_rep = self.model.get_covariance_between_points(self.repr_points_and_exps, x)#update: replace self.repr_points by self.repr_points_and_exps
            dm_rep = sigma_x_rep / stdev_x
            
            # Compute the deterministic innovation for the variance
            dv_rep = -dm_rep.dot(dm_rep.T)
            return dm_rep, dv_rep

        def _write_csv(self, filename, data):
            with open(filename, 'w') as csv_file:
                writer = csv.writer(csv_file, delimiter='\t')
                writer.writerows(data)

        def save_report(self, report_file= None):
            """
            Saves a report with the main results of the optimization.
            
            :param report_file: name of the file in which the results of the optimization are saved.
            """
            results = [self.minmax_x1]
            header= ['minmax_locations']

            data=[header] + results
            
            self._write_csv(report_file, data)
                

import numpy as np
from pylab import grid
import matplotlib.pyplot as plt
from pylab import savefig
import pylab

def plot_logP(filename,logP,repr_points,bounds=None):
    plt.close()
    plt.figure(figsize=(15,5))
    if not bounds is None:
        plt.xlim(bounds[0][0], bounds[0][1])

    if not (repr_points is None):
        best_candidat_for_minmax=repr_points[np.argmax(logP)]
    else:
        best_candidat_for_minmax='None'
                
    plt.title('{0} len(logP: {2}  best_candidat_for_minmax: {1} sum P: {3} \n P: {4}'.format(str(filename),best_candidat_for_minmax,len(repr_points), np.sum(np.exp(logP)),0 ))#, sum([' {:.2f}'.format(v) for v in np.exp(logP)]) )
    #lq=np.quantile(logP,0.0)
    #uq=np.quantile(logP,1.0)
    #plt.plot(repr_points, np.exp(np.clip(logP,lq,uq)), 'r.', markersize=10)
    plt.plot(repr_points, np.exp(logP), 'r.', markersize=10)
    savefig(filename)
    plt.clf()
    
