import torch 
import torch.nn as nn
import math
from .model import Model
from utils import UniformValueSampler


class Heat_PDE_ANN(Model):
    """
        This model approximates the solution to the Heat Partial Differential Equation
        using a neural network. The model is trained to learn the temperature 
        distribution over time and space based on given initial and boundary conditions.
    """
    def __init__(self, 
                 neurons,  # list specifying the number of neurons in each layer of the network
                 space_bounds,  # the bounds for the input space
                 T,  # time 
                 rho,   # Thermal diffusivity parameter in the heat equation
                 test_size,   # size of the test data for evaluation
                 dev=None,  # device
                 activation='GELU'   # activation function to use between layers
                 ): 

        super().__init__()
        
        # Initialize model parameters
        self.id = "heat"
        self.plotname = "Heat Model"
        self.neurons = neurons
        self.space_bounds = space_bounds
        self.sampler = UniformValueSampler(neurons[0], space_bounds, dev)
        self.test_data = (space_bounds[1] - space_bounds[0]) * torch.rand(test_size, neurons[0], device=dev) + space_bounds[0]
        self.rho = rho
        self.T = T
        
        # Construct the network layers
        depth = len(neurons) - 1
        for i in range(depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]).to(dev))
            self.layers.append(self.act_dict[activation])
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]).to(dev))
        
        
       
    def data_sampler(self,batch_size):
        """Samples a batch of data from the sampler."""
        return self.sampler.sample(batch_size)
    
    def phi(self, x):
        """Initial condition of the heat PDE."""
        return x.square().sum(dim=1, keepdim=True) 
    
    def u_T(self,x):  
        """Solution of the heat PDE at time T."""
        return self.phi(x) + 2. * self.rho * self.T * self.neurons[0]
    
    def loss(self, data):
        """Computes the loss between the model's prediction and a noisy version of the solution."""
        W = torch.randn_like(data)
        return (self.phi(math.sqrt(2 * self.rho * self.T) * W + data) - self.forward(data)).square().mean().sqrt()/self.phi(math.sqrt(2 * self.rho * self.T) * W + data).square().mean().sqrt() 
    

    def test_loss(self, data):
        """Compute the test loss (relative L2-error) using the real solution."""
        output = self.forward(data)
        u_test =self.u_T(data)
        return ((u_test - output)).square().mean().sqrt() / u_test.square().mean().sqrt()
