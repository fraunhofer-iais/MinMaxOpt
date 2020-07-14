# Use the Benchmark of Wabersich and Touissant for the Min Max Optimization

import numpy as np
import torch, botorch, gpytorch
import pandas as pd
from utils import *

def min_max_optimization():
    np.random.seed(42)
    torch.manual_seed(32)
    iterations_list = [20, 20, 100]
    for problem_idx, problem in enumerate(['branin', 'camel', 'eggholder']):

        # Problem definition
        testfunction, slices, scalers, hyperparameters = get_test(problem)

        # initialization: loaded from csv data
        n_init = 5
        jj = 0
        # load data
        df_read = pd.read_csv('initialization_'+problem+'.csv')
        for initialization in range(int(df_read.shape[0]/n_init)):
            X = np.array(df_read.iloc[(initialization*n_init):(initialization*n_init+n_init),0:2])
            z = np.array(df_read.iloc[(initialization*n_init):(initialization*n_init+n_init),2]).reshape(-1, 1)

            # scale
            X_scaled = torch.tensor(scalers[0].transform(X), dtype=torch.float32)
            z_scaled = torch.tensor(scalers[1].transform(z), dtype=torch.float32)

            # run the optimization
            iterations = iterations_list[problem_idx]

            results = torch.zeros((1, 7))
            options = []
            option = 'inner'
            i_inner = 1
            i_outer = 1

            # initialize model
            model = build_models(X_scaled, z_scaled, hyperparameters)
            model = model.eval()

            for i in range(iterations):
                # select an initialization randomly
                options.append(option)

                # calculations for stopping criterion:
                epsilon = calculate_epsilon(model)

                new_candidate, min_max_location, current_min_max = min_max_wabersich(model, slices, option, i_inner, i_outer, hyperparameters)

                current_min_max_unscaled = torch.tensor(scalers[1].inverse_transform(current_min_max.detach().numpy().reshape(1, 1)))
                min_max_location_unscaled = torch.tensor(scalers[0].inverse_transform(min_max_location.detach().numpy()))
                new_candidate_unscaled = torch.tensor(scalers[0].inverse_transform(new_candidate.detach().numpy()))

                new_function_value = testfunction(new_candidate_unscaled.reshape(1, -1))

                # update the model
                model = model.condition_on_observations(new_candidate, torch.tensor(
                    scalers[1].transform(new_function_value.numpy().reshape(-1, 1))))

                print('new candidate:', new_candidate_unscaled)
                print('min max location:', min_max_location_unscaled)
                print('current min max:', current_min_max_unscaled)

                print('iteration ', i)
                results[0, 0] = i
                results[0, 1:3] = new_candidate_unscaled
                results[0, 3:5] = min_max_location_unscaled
                results[0, 5] = current_min_max_unscaled
                results[0, 6] = initialization

                df = pd.DataFrame(results.detach().numpy(), columns=['i', 'x_cand0', 'x_cand1', 'min_max0', 'min_max1', 'min_max_val','init'])
                df['problem'] = problem
                if jj == 0:
                    df.to_csv(problem + '_results_Benchmark.csv', index=False)
                else:
                    df.to_csv(problem + '_results_Benchmark.csv', mode='a', header=False, index=False)
                jj += 1

                # prepare next iteration:
                if option == 'inner':
                    i_inner += 1
                    if (abs(calculate_epsilon(model) - epsilon) < 0.1)\
                            | (new_candidate[-1, 1] == model.train_inputs[0][-2, 1]):
                        option = 'outer'
                    # evaluate point and write to file
                else:
                    i_outer += 1
                    option = 'inner'
                    i_inner = 1

            print('finished the optimization')


if __name__=='__main__':
    min_max_optimization()
