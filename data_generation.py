"""
Main file contaning functions to generate data for our experiments. 

Creation date: 01/02/2024
Last modificaiton: 01/02/2024
By: Mehdi EL KANSOULI 
"""

import numpy as np 
import torch 
import torch.nn as nn 

def generate_data(n_points, noise_scale=.25, deterministic=False, freq=1.5):
    """
    Function to generate data from a cosinus with noise added. 
    X is chosen uniformly between 0 and 1

    :params n_points: int
        Number of data points to generate. 
    :params noise_scale: float, default=.25
        Std of gaussian noise

    :return tuple of arrays
        X and y. 
    """
    # generate X points
    if deterministic:
        X = torch.tensor(np.linspace(0, 1., n_points).reshape(n_points, 1), dtype=torch.float32)
    else: 
        X = torch.tensor(np.random.rand(n_points, 1), dtype=torch.float32)

    # generate correpsonding y (with added noise)
    noise = np.random.normal(loc=0., scale=noise_scale, size=(n_points, 1))
    y = torch.tensor(np.cos(X * 2 * freq * np.pi) + noise, dtype=torch.float32)
    
    return X, y 