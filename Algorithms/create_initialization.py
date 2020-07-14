# create initializations for the three different test problems
# for every problem: create 100*5 random X values, distributed on the given slices
# for every problem: create corresponding responses
# save the data to be used in the torch and GPy implementation
from utils import get_test
import numpy as np
import torch
import pandas as pd

torch.manual_seed(42)
np.random.seed(42)

problems = ['branin', 'eggholder', 'camel']
for problem in problems:
    testfunction, slices, scalers, hyperparameters = get_test(problem)

    # create 500 points on [0, 1]
    X1 = np.random.rand(500, 1)
    X2 = np.random.choice(slices.numpy(), (500, 1)) # slices are already scaled

    X = scalers[0].inverse_transform(torch.tensor(np.concatenate((X1, X2), axis=1))) # scale back for evaluation

    y = testfunction(torch.tensor(X))

    # write to csv
    df = pd.DataFrame(np.concatenate((X, y.reshape(-1, 1)), axis=1), columns=['X1', 'X2', 'y'])
    df.to_csv('initialization_'+problem+'.csv', index=False)
