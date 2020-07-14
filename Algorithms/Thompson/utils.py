import torch, botorch, gpytorch
import numpy as np
from copy import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



def build_models(X, z, hyperparameters):
    # learn the gpytorch model
    # kernel: squared exponential
    # set kernel hyperparameters
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


def get_grid(model, slices, resolution):
    xx, yy = torch.meshgrid([torch.linspace(0., 1., int(resolution/10)), slices])
    X = torch.stack([torch.flatten(xx), torch.flatten(yy)]).T

    # learn about the probability distribution
    model = model.eval()
    if model.train_targets.ndim == 1:
        prediction = model(X).mean
    else: prediction = model(X).mean.mean(axis=0) # use the mean values for a lot of samples
    f_vals = -torch.max(prediction.reshape(xx.shape), axis=1).values

    # map to probability values
    ps = (f_vals-torch.min(f_vals))/(torch.max(f_vals)-torch.min(f_vals))
    ps = ps/ps.sum()

    # sample with probability according to f_values, but lower f_values have higher probability
    pdf = torch.distributions.categorical.Categorical(ps)
    samples = pdf.sample_n(resolution)

    # map back to grid
    # draw resolution samples of the uniform distribution between 0 and 1
    noise = (torch.rand(resolution)-0.5)*(1/(resolution))
    grid_vals = torch.linspace(0, 1, resolution).repeat(resolution, 1)[:, samples*10][0] + noise

    # append the grid_vals with the coarse grid
    grid_vals = torch.sort(torch.cat([grid_vals, torch.linspace(0,1,int(resolution/10))]))[0]

    # suppress points out of bounds
    grid_vals = grid_vals[(grid_vals>=0.) & (grid_vals<=1.)]

    return grid_vals


def thompson_sampling_acquisition(model, slices):
    # idea: draw a sample from the model and find its' min max
    grid_vals = get_grid(model, slices, resolution=1000)

    xx, yy = torch.meshgrid([grid_vals, slices.flatten()])
    X = torch.stack([torch.flatten(xx), torch.flatten(yy)]).T

    predictions = model(X)

    # stddeviation might become nan due to ill-conditioned covariance matrix
    # catch: set stddev of this entry back to 1
    a = copy(predictions.stddev.detach())
    a[torch.isnan(a)] = 1.
    px = torch.distributions.Normal(predictions.mean, a)

    posterior_sample = px.sample()
    # search for the min max value
    current_min_max = torch.min(torch.max(posterior_sample.reshape(xx.shape), axis=1).values)
    # find the corresponding x value
    candidate = X[posterior_sample == current_min_max]
    if (candidate.ndim > 1) & (candidate.shape[0] > 1):
        candidate = candidate[torch.multinomial(torch.arange(candidate.shape[0], dtype=float), 1)]
    candidate = candidate.reshape(1, -1)
    return candidate, candidate, current_min_max


