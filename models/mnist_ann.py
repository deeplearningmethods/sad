import torch.nn as nn
import torchvision
import torch
from torchvision import transforms
from .model import Model


class MNIST_ANN(Model):
    """
        A simple feedforward neural network for classifying MNIST digits.
    """
    def __init__(self,
                 neurons, # list specifying the number of neurons in each layer of the network
                 batch_size,  # batch size for training
                 dev,  # device
                 activation='GELU', # activation function to use between layers
                 ):
        super().__init__()
        
        # Initialize model parameters
        self.id = "mnist"
        self.plotname = "MNIST Model"
        self.activation = activation
        # Construct the network layers
        depth = len(neurons) - 1
        self.layers.append(nn.Flatten(start_dim=1))
        for i in range(depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]).to(dev))
            self.layers.append(self.act_dict[self.activation])
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]).to(dev))
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        self.dev = dev
        
        # Dataset initialization
        transform = transforms.Compose([transforms.ToTensor()])      # transformations
        self.train_dataset = torchvision.datasets.MNIST('classifier_data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST('classifier_data', train=False, download=True, transform=transform)
        # Load full test data into one batch
        self.test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(self.dev)
        self.test_data_label = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))]).to(self.dev)

    def data_sampler(self,batch_size):
        """Randomly samples a batch of data from the training set."""
        perm = torch.randperm(len(self.train_dataset))[:batch_size]
        images = torch.stack([self.train_dataset[i][0] for i in perm])
        labels = torch.tensor([self.train_dataset[i][1] for i in perm])
        return images.to(self.dev), labels.to(self.dev)

    
    
    def loss(self, data):
        """Loss function for training the model."""
        image_batch, label_batch = data 
        return self.loss_fn(self.forward(image_batch), label_batch)

    
    def test_loss(self,data):
        """Computes the test loss based on the full test dataset."""
        return self.loss([self.test_data, self.test_data_label])
    
            
