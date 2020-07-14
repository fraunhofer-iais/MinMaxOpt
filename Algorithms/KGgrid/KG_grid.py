# Use the knowledge gradient to minimize the worst-case-error
import numpy as np
import torch, botorch, gpytorch
import pandas as pd
import ray
from grid_utils import get_grid, get_KG_grid, build_models, get_test


@ray.remote
def get_KG(x, sampler, X_for_min_max, current_min_max, xx, X_train, y_train, hyperparameters):
    model = build_models(X_train, y_train, hyperparameters)
  
    single_fantasy = model.fantasize(x.reshape(1, -1), sampler)  # will give an updated model

    new_min_max = torch.mean(
        torch.min(torch.max(single_fantasy(X_for_min_max).mean.view(100, xx.shape[0], xx.shape[1]), axis=2).values,
                  axis=1).values)
    return current_min_max - new_min_max

def min_max_optimization():
    np.random.seed(42)
    torch.manual_seed(32)
    iterations_list = [20, 20, 100]
    for problem_idx, problem in enumerate(['branin', 'camel', 'eggholder']):
        print(problem)

        # Problem wird definiert durch:
        # Funktion, Bounds, Slices, Hyperparameter
        testfunction, slices, scalers, hyperparameters = get_test(problem)

        # initialization
        n_init = 5
        jj = 0
        # load data
        df_read = pd.read_csv('initialization_'+problem+'.csv')
        print("read data")
        for initialization in range(int(df_read.shape[0]/n_init)):
            X = np.array(df_read.iloc[(initialization*n_init):(initialization*n_init+n_init),0:2])
            z = np.array(df_read.iloc[(initialization*n_init):(initialization*n_init+n_init),2]).reshape(-1, 1)

            # scale
            X_scaled = torch.tensor(scalers[0].transform(X), dtype=torch.float32)
            z_scaled = torch.tensor(scalers[1].transform(z), dtype=torch.float32)

            # run the optimization
            iterations = iterations_list[problem_idx]
            model = build_models(X_scaled, z_scaled, hyperparameters)
            model = model.eval()

            results = torch.zeros((1, 7))
            print("started optimization")
            for i in range(iterations):
                resolution = 100
                # create an evaluation grid
                grid_vals = get_grid(model, slices, resolution)

                xx, yy = torch.meshgrid([grid_vals, slices.flatten()])
                X_for_min_max = torch.stack([torch.flatten(xx), torch.flatten(yy)]).T

                # find minimum of worst case function
                prediction = model(X_for_min_max).mean
                current_min_max = torch.min(torch.max(prediction.reshape(xx.shape), axis=1).values)

                # the min max location:
                min_max_location = X_for_min_max[prediction == current_min_max]
                if (min_max_location.ndim > 1) & (min_max_location.shape[0] > 1):
                    min_max_location = min_max_location[
                        torch.multinomial(torch.arange(min_max_location.shape[0], dtype=float), 1)]
                min_max_location = min_max_location.reshape(1, -1)

                KG_grid_vals = get_KG_grid(model, slices, resolution)
                xx_KG, yy_KG = torch.meshgrid([KG_grid_vals, slices.flatten()])
                X_for_KG = torch.stack([torch.flatten(xx_KG), torch.flatten(yy_KG)]).T

                # create fantasy models: take 100 samples on every different X
                sampler = botorch.sampling.samplers.IIDNormalSampler(100, resample=False, seed=42,
                                                                     collapse_batch_dims=True)

                result_ids = []
                X_train = model.train_inputs[0]
                y_train = model.train_targets.reshape(-1, 1)
                
                for x in X_for_KG:
                    result_ids.append(get_KG.remote(x, sampler, X_for_min_max, current_min_max, xx, X_train, y_train, hyperparameters))
                KG = torch.tensor(ray.get(result_ids))
                
                max_idx_KG = torch.argmax(KG)
                max_value_KG = KG[max_idx_KG]
                print('maximum knowledge gradient of', max_value_KG, 'at ', X_for_KG[max_idx_KG])

                new_candidate = X_for_KG[max_idx_KG].reshape(1, -1)
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
                    df.to_csv(problem + '_results_KGgrid.csv', index=False)
                else:
                    df.to_csv(problem + '_results_KGgrid.csv', mode='a', header=False, index=False)
                jj += 1

            print('finished the optimization')


if __name__=='__main__':
    ray.init()
    min_max_optimization()
