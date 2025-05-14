import torch 
import torch.nn as nn
import math
from tqdm import tqdm
import numpy as np
from .model import Model
from utils import UniformValueSampler





class BlackScholes_ANN(Model):
    """
        A neural network-based model for pricing options using the Black-Scholes framework.
        The model can compute an approximation of the solution to the Black-Scholes PDE using the Deep Kolmogorov method,
        which turns the PDE into a stochastic optimization problem.
    """
    def __init__(self, 
                 neurons,  # list specifying the number of neurons in each layer of the network
                 space_bounds,  # the bounds for the input space
                 T,  # time to maturity for the option
                 c,  # cost of carry
                 r,  # risk-free interest rate
                 K,  # strike price
                 sigma,  # volatility
                 dev,  # device
                 activation='GELU',  # activation function to use between layers
                 final_u=None,  # a function to provide the true solution for the option, if available
                 mc_samples=1024,  # number of Monte Carlo samples to use for approximation
                 test_size=10000,  # size of the test data for evaluation
                 mc_rounds=1000  # number of rounds for Monte Carlo simulation when computing the reference solution
                 ):  
        super().__init__()
        
        # Initialize model parameters
        self.id = "blackscholes"
        self.plotname = "Black Scholes Model"
        self.activation = activation
        
        self.sampler = UniformValueSampler(neurons[0], space_bounds, dev)
        self.test_data = (space_bounds[1] - space_bounds[0]) * torch.rand(test_size, neurons[0], device=dev) + space_bounds[0]
        # custom attributes
        self.dev = dev 
        self.mu = r - c
        self.K = K
        self.r = r
        self.sigma = sigma.to(dev)
        self.mc_samples = mc_samples
        self.T = T
        self.space_bounds = space_bounds
        self.final_u = final_u

        # Construct the network layers
        depth = len(neurons) - 1
        self.layers.append(nn.BatchNorm1d(neurons[0]).to(dev))
        for i in range(depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]).to(dev))
            self.layers.append(self.act_dict[self.activation])
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]).to(dev))

        self.done_steps = 0
        self.losses = []
        self.lr_list = []

        # Reference solution computation via Monte Carlo simulation if no final_u is provided
        if self.final_u is None:  
            print(f'Computing reference sol by MC with {mc_rounds} rounds and {mc_samples} samples.')
            u_ref = torch.zeros([test_size, neurons[-1]],device = self.dev)
            for i in tqdm(range(mc_rounds)):
                #print(f"Round {i}/{mc_rounds}", end = "\r")
                x = torch.stack([self.test_data for _ in range(self.mc_samples)])
                w = torch.randn_like(x, device = self.dev)
                u = self.phi(self.test_data * torch.exp((self.mu - 0.5 * self.sigma ** 2) * self.T + self.sigma * torch.tensor(math.sqrt(self.T),device = self.dev) * w))
                u = torch.mean(u, dim=0)
                u_ref += u
            self.u_test = u_ref / mc_rounds
            print(f'Reference sol computed, shape: {self.u_test.shape}.')
        else:
            self.u_test = self.final_u(self.test_data)

    def phi(self,x):  
        """Initial value function for the Black-Scholes option pricing."""
        return torch.tensor(np.exp(-self.r * self.T)) * torch.maximum(torch.max(x, dim=-1, keepdim=True)[0] - self.K, torch.tensor(0.))
    
    def data_sampler(self,batch_size):
        """Samples a batch of data from the sampler."""
        return self.sampler.sample(batch_size)


    def loss(self, data):
        """Computes the loss between the model's prediction and a noisy version of the solution."""
        W = torch.randn_like(data, device=self.dev)
        X = data * torch.exp((self.mu - 0.5 * self.sigma ** 2) * self.T + self.sigma * math.sqrt(self.T) * W)
        return (self.phi(X).to(self.dev) - self.forward(data).to(self.dev)).square().mean().sqrt()/self.phi(X).to(self.dev).square().mean().sqrt()

    def test_loss(self, data):
        """Computes the test loss based on the high quality MC reference solution."""
        output = self.forward(data)
        return ((self.u_test - output) ).square().mean().sqrt()/ self.u_test.square().mean().sqrt()


