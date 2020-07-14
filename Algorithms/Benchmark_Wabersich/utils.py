import torch, botorch, gpytorch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def build_models(X, z, hyperparameters):
    # learn the gpytorch model
    # kernel: squared exponential
    # set fixed kernel hyperparameters
    model = botorch.models.gp_regression.SingleTaskGP(X, z,
                                                      likelihood=gpytorch.likelihoods.GaussianLikelihood(
                                                          noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
                                                      ),
                                                      covar_module=gpytorch.kernels.ScaleKernel(
                                                          gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1])),
                                                      )

    hypers = {
        'likelihood.noise_covar.noise': hyperparameters[0],
        'covar_module.base_kernel.lengthscale': hyperparameters[2],
        'covar_module.outputscale': hyperparameters[1],
    }
    model.initialize(**hypers)

    return model

def get_test(tag):
    if tag == 'branin':
        def testfunction(X):
            tfunction = botorch.test_functions.Branin()
            f_val = tfunction.evaluate_true(X)
            return -f_val
        slices = np.array([[0., 4., 8., 12.]]).T

        # scalers: input domain of [-5, 10] [0, 15] to 0,1
        X_scale = np.array([[-5, 0], [10, 15]])

        # hyperparameters
        hypers = [0.001, 1, torch.tensor([[0.2, 0.4]])]
    elif tag == 'camel':
        def testfunction(X):
            tfunction = botorch.test_functions.SixHumpCamel()
            f_val = torch.log(tfunction.evaluate_true(X)+2)
            return f_val
        slices = np.array([[-0.9, 0., 1.]]).T

        # scalers: input domain of [(-3.0, 3.0), (-2.0, 2.0)]
        X_scale = np.array([[-3, -2], [3, 2]])

        # hyperparameters
        hypers = [0.001, 0.5, torch.tensor([[0.1, 0.1]])]
    elif tag == 'eggholder':
        def testfunction(X):
            tfunction = botorch.test_functions.EggHolder()
            f_val = tfunction.evaluate_true(X)
            return f_val
        slices = np.array([[-512., 0., 185.]]).T

        # scalers: input domain of [-512, 512] [-512, 512] to 0,1
        X_scale = np.array([[-512, -512], [512, 512]])


        # hyperparameters
        hypers = [0.001, 1., torch.tensor([[0.09, 0.09]])]
    else: raise('Testfunction not implemented.')

    scaler1 = MinMaxScaler()
    scaler1.fit(X_scale)

    slices = torch.tensor(scaler1.transform(np.concatenate((np.zeros_like(slices), slices), axis=1)), dtype=torch.float)[:, 1]

    # y-scale: monte-carlo integration, as scipy dblquad fails
    X = torch.tensor(scaler1.inverse_transform(np.random.rand(10 ** 7, 2)))
    y = testfunction(X)
    scaler2 = StandardScaler()
    scaler2.fit(y.numpy().reshape(-1,1))

    return testfunction, slices, [scaler1, scaler2], hypers

def min_max_wabersich(model, slices, option, i_inner, i_outer, hyperparameters):
    # construct a model for the inner loop
    # option inner loop (find the maximum):
    if option == 'inner':
        # calculate the current beta-value
        beta = 2*np.log(len(slices)*i_inner**2 * np.pi**2 / 30)
        GP_UCB = botorch.acquisition.analytic.UpperConfidenceBound(model, beta, objective=None, maximize=True)

        # bounds = torch.tensor([[0., 0.], [1., 1.]])
        # evaluate GP_UCB on the slices (for the current X) and return the maximizing one
        XX = torch.cat((model.train_inputs[0][-1, 0].repeat(len(slices)).reshape(-1, 1), slices.reshape(-1, 1)), 1)
        ucb = GP_UCB(XX.view(len(slices), 1, 2))
        candidate = XX[torch.argmax(ucb)].reshape(1, -1)

    elif option =='outer':
        beta = 2*np.log(i_outer**2 * 2* np.pi**2 / 15) + 2 * 1*np.log(i_outer**2 * 1 * np.sqrt(np.log(20*1)))

        # problem: botorch ucb optimization is not made for minimization
        model_min = build_models(model.train_inputs[0], -model.train_targets.reshape(-1, 1), hyperparameters)
        GP_UCB = botorch.acquisition.analytic.UpperConfidenceBound(model_min, beta, objective=None, maximize=True)

        bounds = torch.tensor([[0., 0.], [1., 1.]])
        candidate, acqf = botorch.optim.optimize.optimize_acqf(
            GP_UCB,
            bounds,
            q=1,
            num_restarts = 5,
            raw_samples=100,
            fixed_features={1: model.train_inputs[0][-1, 1]})

    return candidate, candidate, model(candidate).mean


# stopping criterion (only for the inner loop = maximization)
def calculate_epsilon(model):
    # calculations for stopping criterion:
    UCB = botorch.acquisition.analytic.UpperConfidenceBound(model, beta=1., objective=None, maximize=True)
    PM = botorch.acquisition.analytic.PosteriorMean(model, objective=None)
    bounds = torch.tensor([[0., 0.], [1., 1.]])
    _, UCB_val = botorch.optim.optimize.optimize_acqf(
        UCB,
        bounds,
        q=1,
        num_restarts=5,
        raw_samples=100,
        fixed_features={0: model.train_inputs[0][-1, 0]})
    _, PM_val = botorch.optim.optimize.optimize_acqf(
        PM,
        bounds,
        q=1,
        num_restarts=5,
        raw_samples=100,
        fixed_features={0: model.train_inputs[0][-1, 0]})
    return UCB_val-PM_val