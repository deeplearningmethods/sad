import torch
import torch.nn as nn
from .model import Model
from utils import ShuffleSampler

class Polynomial_ANN(Model):
    """
        A simple feedforward neural network for approximating polynomials.
    """
    def __init__(self,
                 neurons, # list specifying the number of neurons in each layer of the network
                 space_bounds,  # the bounds for the input space
                 dev,  # device
                 train_size,  # size of the train data
                 test_size,   # size of the test data for evaluation
                 activation, # activation function to use between layers
                 ):
        super().__init__()
        
        # Initialize model parameters
        self.id = "polynomial"
        self.plotname = "Polynomial Model"
        self.activation = activation
        
        # Construct the network layers
        depth = len(neurons) - 1
        for i in range(depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]).to(dev))
            self.layers.append(self.act_dict[self.activation])
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]).to(dev))
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        self.dev = dev
        
        self.sampler = ShuffleSampler(train_size, neurons[0], space_bounds, dev)
        self.test_data = torch.cartesian_prod(*[torch.linspace(space_bounds[0], space_bounds[1], test_size, device=dev)]*neurons[0])
        if neurons[0] == 1:
            self.test_data = self.test_data.unsqueeze(1)
        
        # assign the target function depending on the input dimension
        if neurons[0]==1:
            self.target = self.target_onedim
        elif neurons[0]==2:
            self.target = self.target_twodim
        elif neurons[0]==4:
            self.target =  self.target_fourdim
        else:
            raise ValueError("Input dimension not supported")
    
    def target_onedim(self,x):
        result = torch.pow(x, 10) - 2 * torch.pow(x, 8) + 2 * torch.pow(x, 5) + 3 * torch.pow(x, 3)\
            - 2 * torch.pow(x, 2) + 5
        return result
    
    def target_twodim(self,x):
        result =  torch.pow(x[:, 1], 5) \
             - torch.pow(x[:, 0], 3) * torch.pow(x[:, 1], 2)  \
             - 4 * torch.pow(x[:, 0], 2) * torch.pow(x[:, 1], 1) + 3 * torch.pow(x[:, 0], 3) \
             - torch.pow(x[:, 1], 2) + x[:, 0] + 2
        return result.unsqueeze(1)
    
    def target_fourdim(self,x):
        result = torch.pow(x[:, 0], 6) * torch.pow(x[:, 3], 5) + torch.pow(x[:, 1], 6) \
            - torch.pow(x[:, 0], 3) * torch.pow(x[:, 1], 2) * torch.pow(x[:, 2], 1) + torch.pow(x[:, 3], 2) \
            - 4 * torch.pow(x[:, 2], 4) * torch.pow(x[:, 1], 4) + 3 * torch.pow(x[:, 3], 3) * torch.pow(x[:, 1], 3) \
            - torch.pow(x[:, 2], 2)*x[:, 0] +x[:, 2] + 3
        return result.unsqueeze(1)


    def data_sampler(self,batch_size):
        """Samples a batch of data from the sampler."""
        return self.sampler.sample(batch_size)
    
    
    def loss(self, data):
        """Loss function for training the model."""        
        return self.loss_fn(self.forward(data.to(self.dev)), self.target(data).float().to(self.dev)) #+ 0.0001 * reg_loss

    
    def test_loss(self,data):
        """Computes the test loss based on the full test dataset."""
        return self.loss(data)
    
    
