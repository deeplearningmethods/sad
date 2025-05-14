import torch 
import pickle
import io
import yaml
import matplotlib.pyplot as plt 
import os
import numpy as np
import string


def load_config(config_path):
    """Load the configuration in the yaml file"""
    if config_path=='':
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def override_config(config, args):
    """Overwrite the configuration in the yaml file with arguments passed in the command line"""
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config

def initial_values_sampler_uniform(batch_size, space_dim, space_bounds,dev):
    """Sample initial values uniformly within the specified bounds."""
    a, b = space_bounds
    return (b - a) * torch.rand([batch_size, space_dim], device = dev) + a

class UniformValueSampler:
    """
        This class samples from a uniform distribution by calling self.sample(batch_size)
    """
    def __init__(self, space_dim, space_bounds, dev):
        self.space_dim = space_dim
        assert len(space_bounds) == 2 # Ensure space_bounds has two elements (min, max)
        self.space_bounds = space_bounds
        self.dev = dev

    def sample(self, batch_size):
        values = initial_values_sampler_uniform(batch_size, self.space_dim, self.space_bounds, self.dev)
        return values

class ShuffleSampler:
    """
        This class samples from a dataset by calling self.sample(batch_size)
    """
    def __init__(self, size_data, space_dim, space_bounds, dev):
        
        assert len(space_bounds) == 2 # Ensure space_bounds has two elements (min, max)
        self.data = (space_bounds[1] - space_bounds[0]) * torch.rand(size_data, space_dim, device=dev) + space_bounds[0]
        

    def sample(self, batch_size):
        # Randomly sample b indices from the data tensor
        indices = torch.randint(0, self.data.size(0), (batch_size,))
        # Sample the batch using the generated indices
        batch = self.data[indices]
        return batch


class CPU_Unpickler(pickle.Unpickler):
    """
    Class needed to load class when only cpu is available and model is trained with gpu 
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
        
        
def load_experiment(folder_name):
    """
    Load a saved experiment from the specified folder.
    
    """
    with open(f"{folder_name}/trainer.pkl", "rb") as f:
        model = CPU_Unpickler(f).load()
    
    print(f"Experiment loaded from '{folder_name}'")
    return model


def ema_smooth(data, decay=0.9):
   smoothed = np.zeros_like(data)
   smoothed[0] = data[0]
   for t in range(1, len(data)):
       smoothed[t] = decay * smoothed[t-1] + (1 - decay) * data[t]
   return smoothed


def plot_mean_norm_and_loss(folder_paths, scale = "linear", folder_name="", legend = []):
     """
     Plot the mean of 'norm' and 'loss' from multiple saved models in the same figure.
     """
     
     columns = len(folder_paths)
     fig_size = (5*columns, 7)
     fig, axes = plt.subplots(2, columns, figsize=fig_size)  # Create subplots
     
     # Ensure axes is a 2D array even if columns=1
     if columns == 1:
         axes = np.expand_dims(axes, axis=1)
         
     for column in range(columns):
         steps = []
         for i in range(len(folder_paths[column])):
             # Load the model and its training data
             model = load_experiment(folder_paths[column][i])
             
             # Extract norms and test loss from the 'runs' dictionary
             all_norms = [run["norms"] for run in model.runs]
             all_loss = [run["train_loss"] for run in model.runs]
             if 'pde' in folder_paths[column][i]:
                 all_loss = [run["test_loss"] for run in model.runs]
             # Convert lists to numpy arrays
             norm_matrix = np.array(all_norms)
             loss_matrix = np.array(all_loss)
             
             # Compute mean at each step
             norm_mean = np.mean(norm_matrix, axis=0)
             loss_mean = np.mean(loss_matrix, axis=0)
             
             # Common steps (assuming all runs share the same step sequence)
             common_steps = model.runs[0]["steps"]
             steps.append(max(common_steps))
             
             # Plot mean norm
             axes[0,column].plot([0] + common_steps, norm_mean, label=legend[column if len(legend)>column else 0][i])
             
             smoothed_loss = ema_smooth(loss_mean, decay=0.95)
             axes[1,column].plot(common_steps[::2], smoothed_loss[::2], label=legend[column if len(legend)>column else 0][i])
             
         # Customize Norm Plot
         axes[0,column].set_xlim(0, max(steps))
         axes[0,column].set_ylim(0)
     
         # Customize Loss Plot
         axes[1,column].set_xlim(0, max(steps))
         axes[1,column].set_xlabel("gradient steps")
         axes[1,column].set_yscale(scale)  
         label = f"({string.ascii_lowercase[column]})"
         axes[1,column].text(0.5, -0.3, label, transform=axes[1,column].transAxes,ha='center', va='center', fontsize=12)
         
         
     axes[0,0].set_ylabel("parameter norm")
     axes[1,0].set_ylabel("loss")
     for l in range(len(legend)):
         axes[0,l].legend()
     
     # Save the plot if required
     if folder_name != "":
         os.makedirs(folder_name, exist_ok=True)
         file_name = f"{folder_name}/mean_norm_loss_plot.pdf"
         plt.savefig(file_name, bbox_inches="tight")
    
     # Show the plot
     plt.tight_layout()
     plt.show()
     
         