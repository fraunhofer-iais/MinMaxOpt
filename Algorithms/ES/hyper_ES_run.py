# Use entropy search to minimize the worst-case-error
import numpy as np
import GPy
import GPyOpt
import torch#, botorch, gpytorch

#import matplotlib.pyplot as plt
import pandas as pd
from EP.utils_dorina import *
from sklearn.preprocessing import StandardScaler

from GPyOpt.models.gpmodel import GPModel
from GPyOpt.core.task.space import Design_space, bounds_to_space
from GPyOpt.util.mcmc_sampler import AffineInvariantEnsembleSampler
from GPyOpt.acquisitions.ES import AcquisitionEntropySearch

from EP.utils.mim_mac_entropy_acquisition import AcquisitionEntropySearch_MinMax



def min_max_optimization():
    np.random.seed(42)
    torch.manual_seed(32)
    for problem in ['eggholder','branin','camel' ]:

        # Problem wird definiert durch:
        # Funktion, Bounds, Slices, Hyperparameter
        testfunction, slices, scalers, hyperparameters = get_test(problem)
        gp_noise,signal_var,lengthscale=hyperparameters
        # convert from torch to numpy:
        
        def func_on_normalized(X):
            #Assumption X is in normed domain
            #Functions returns values in normed domain of Y
            X_scaled = torch.tensor(scalers[0].inverse_transform(X), dtype=torch.float32)#Now X_scaled lives in the original domain
            return scalers[1].transform(testfunction(torch.tensor(X_scaled, dtype=torch.float32)).reshape(-1, 1)) #scalers[1].transform(testfunction(torch.tensor(X_scaled, dtype=torch.float32)))
        

        objective = GPyOpt.core.task.SingleObjective(func_on_normalized)
        bounds=list()
        bounds.append(list())
        bounds.append(list())
        
        bounds[0].append(0)
        bounds[0].append(1)
        bounds[1].append(0)
        bounds[1].append(1)
        
        space_exp = GPyOpt.Design_space(space =[
            {'name': 'var_1', 'type': 'continuous', 'domain': (0,1)},
            {'name': 'var_2', 'type': 'discrete', 'domain': slices}])

        # initialization: LHC # TO-DO: change to random design laded from array, just now on 0 1 (already scaled space)
        n_init = 5
        jj = 0
        # load data
        df_read = pd.read_csv('initialization_'+problem+'.csv')

        sample_size=15
        ignor_previous_best_candidat=True
        num_representer_points=15

        
        for initialization in range(int(df_read.shape[0]/n_init)):
            # Initialize the Bo setting
            acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space=space_exp,
                                                                             optimizer='Grid',
                                                                             slices=slices)
            kern=GPy.kern.RBF(input_dim=2,ARD=True,lengthscale = lengthscale,variance=signal_var)
            es_model = GPyOpt.models.GPModel(kernel=kern,noise_var=gp_noise,max_iters=0,verbose=False)
            sampler=AffineInvariantEnsembleSampler(space_exp)
            
            es=AcquisitionEntropySearch_MinMax(model=es_model,
                                               space=space_exp,
                                               slices=slices,
                                               sampler=sampler,
                                               optimizer=acquisition_optimizer,
                                               num_representer_points=num_representer_points,
                                               proposal_function=None,
                                               burn_in_steps=20,
                                               sample_size=sample_size,
                                               ignor_previous_best_candidat=ignor_previous_best_candidat)
            
            es_evaluator = GPyOpt.core.evaluators.Sequential(es)
        
            X = np.array(df_read.iloc[(initialization*n_init):(initialization*n_init+n_init),0:2])
            z = np.array(df_read.iloc[(initialization*n_init):(initialization*n_init+n_init),2]).reshape(-1, 1)

            
            X_scaled = torch.tensor(scalers[0].transform(X), dtype=torch.float32)
            
            initial_design = X_scaled.numpy()

            
            
        
            bo_es = GPyOpt.methods.ModularBayesianOptimization(model=es_model,
                                                                    space=space_exp,
                                                                    objective=objective,
                                                                    acquisition=es,
                                                                    evaluator=es_evaluator,
                                                                    X_init=initial_design)
                      
            iterations = 25
            max_iter = iterations

            end_fix='run_{0}_for_problem_{1}_nr_argmaxes_{2}_rep_{3}_iterations_{4}.tex'.format(initialization,problem,sample_size,num_representer_points,iterations)
            if ignor_previous_best_candidat:
                end_fix='igr_'+end_fix
            bo_es.run_optimization(max_iter = max_iter,
                                   verbosity=False,
                                   evaluations_file = './eval_'+end_fix,
                                   models_file='./model_'+end_fix,
                                   slices=slices)

            print('+++++++          +++++++++++++++      Finished       +++++++++++++++           +')
            print('./eval_'+end_fix)
            print('+++++++          +++++++++++++++      Finished       +++++++++++++++           +')

            es.save_report('./report_'+end_fix)
            
            
            del bo_es
            del es

            if initialization>200:
                break

            print('finished the optimization')


if __name__=='__main__':
    torch.multiprocessing.freeze_support()
    min_max_optimization()
