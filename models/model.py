import torch 
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 
import os
import pickle


class Model(nn.Module):
    """
    An abstract base class for Artificial Neural Networks.
    Every model should have:
        - a loss function (for training),
        - a sampler (for sampling the data),
        - a test_loss function (for validation).
    """
    def __init__(self) -> None:
        super().__init__()
        
        self.id = None  # Unique identifier for the model (to be generated)
        self.plotname = None  # Name used for plots
        self.layers = nn.ModuleList()  # Stores the layers of the model
        self.test_data = None  # Sampler for the test data
        # Lists for tracking training process
        self.runs = []  # Stores multiple training runs with train loss, test loss, weights norm, steps.
        
        self.act_dict = {
            "softplus": nn.Softplus(),
            "GELU": nn.GELU(),
            "tanh": nn.Tanh(),
            "logistic": nn.Sigmoid(),
            "mish": nn.Mish(),
            "swish": nn.SiLU(),
            'ELU': nn.ELU(),
            "softsign": nn.Softsign()
        }

    def forward(self, data: torch.Tensor) -> torch.Tensor: 
        """Forward pass through the network."""
        for fc in self.layers:
            data = fc(data)
        return data
        
    def loss(self, data: torch.Tensor) -> torch.Tensor:
        """Compute the loss. To be defined in subclasses."""
        pass

    def test_loss(self, data: torch.Tensor) -> torch.Tensor:
        """Compute the test loss. To be defined in subclasses."""
        pass
                    
                    
    def initialize_weights(self) -> None:
        """Reinitialize model weights using Xavier initialization."""
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def data_sampler(self,batch_size) -> torch.Tensor:
        """Return a batch of data for training. To be defined in subclasses."""
        pass
    
    
    
    def train_model(self, optimizer, nr_steps, batch_size, lr, eval_steps=1000)-> None:
        """
        Train the model for a specified number of steps.

        Args:
            optimizer (string): The optimizer used for training (e.g., Adam, SGD).
            nr_steps (int): The total number of training steps.
            batch_size (int): The number of samples per training step.
            eval_steps (int): Interval at which to evaluate the model on the test set.
        """
         
        self.train() # Set the model to training mode
        
        optimizer = {
            "Adam": torch.optim.Adam(self.parameters(), lr=lr),
            "SGD": torch.optim.SGD(self.parameters(), lr=lr),
            "RMSprop": torch.optim.RMSprop(self.parameters(), lr=lr),
            "Adagrad": torch.optim.Adagrad(self.parameters(), lr=lr),
            "AdamW": torch.optim.AdamW(self.parameters(), lr=lr),
            }[optimizer]
        
        
        norm_list = []
        step_list = []
        test_loss_list = []
        train_loss_list = []
        
        norm_list.append(torch.linalg.vector_norm(torch.cat([p.flatten() for p in self.parameters()])).cpu().detach().numpy().item()) # Record initial parameter norm
        steps = 0

        for n in tqdm(range(nr_steps)):
            self.train()  # Ensure model is in training mode
            optimizer.zero_grad()
            data = self.data_sampler(batch_size)  # Sample training data
            loss = self.loss(data)
            loss.backward()
            optimizer.step()
            steps += 1

            # Evaluate the model every eval_steps
            if (n + 1) % eval_steps == 0:
                self.eval()  # Set model to evaluation mode
                with torch.no_grad():  # Disable gradients for efficiency
                    step_list.append(steps)
                    train_loss_list.append(loss.cpu().detach().numpy().item())
                    test_loss_list.append(self.test_loss(self.test_data).cpu().detach().numpy().item())
                    norm_list.append(torch.linalg.vector_norm(torch.cat([p.flatten() for p in self.parameters()])).cpu().detach().numpy().item())
        
        # Get final output on test data
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            last_output = self.forward(self.test_data).cpu().detach().numpy()  # Forward pass on test data
            
        return {"steps": step_list, "norms": norm_list, "test_loss": test_loss_list, "train_loss": train_loss_list, "last_output": last_output}
    
    def train_multiple_runs(self, optimizer, nr_steps, batch_size, lr, eval_steps=1000, num_runs=5):
        """
        Train the model multiple times and store the results.

        Args:
            optimizer (string): The optimizer used for training (e.g., Adam, SGD).
            nr_steps (int): The total number of training steps.
            batch_size (int): The number of samples per training step.
            eval_steps (int): Interval at which to evaluate the model on the test set.
            num_runs (int): Number of times to train the model.

        """
        self.runs = []  # Reset runs before training

        for run in range(num_runs):
            print(f"Training Run {run + 1}/{num_runs}")
            self.initialize_weights()  # Reset model weights before each run
            run_data = self.train_model(optimizer, nr_steps, batch_size, lr, eval_steps)
            run_data["label"] = f"Run {run + 1}"
            self.runs.append(run_data)


    def save_experiment(self, folder_name=""):
            """Save the model, trainer state, and parameters in a dedicated folder."""
            
            if not folder_name:
                folder_name = "{},id_{}".format(
                    self.plotname,
                    self.id
                )
    
            # Create a folder for the experiment
            os.makedirs(folder_name, exist_ok=True)
            
            # Save model state dictionary
            torch.save(self.state_dict(), os.path.join(folder_name, "model.pt"))
            
            # Save the entire trainer object using pickle
            with open(os.path.join(folder_name, "trainer.pkl"), "wb") as f:
                pickle.dump(self, f)
            
            print(f"Experiment saved in '{folder_name}'")


    def plot_graph(self, scale="linear", save=True, ylim_value=None, folder_name=None, realization = False):
        """
        Plot the training process including weight norm and loss.
    
        Args:
            scale (str): 'linear' or 'log' scale for loss.
            save (bool): Whether to save the plot.
            ylim_value (float or None): y-axis limit for loss.
            folder_name (str or None): Folder name to save the figure.
            realization (bool): Whether to plot realization (only when approximating polynomials)
        """

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Create subplots
        #props = dict(boxstyle="round", facecolor="white", alpha=0.25)
        
        ## First subplot - Norm of model parameters
        ax1 = axes[0]
        for run in self.runs:
            ax1.plot([0] + run["steps"], run["norms"],)
        
        ax1.set_xlim(0, max(self.runs[0]["steps"]))
        ax1.set_xlabel("Gradient steps")
        ax1.set_ylabel("Parameter Norm")
        #ax1.text(0.1, 0.85, r"$\|\Theta\|$", transform=ax1.transAxes, bbox=props)
    
        ## Second subplot - Loss evolution
        ax2 = axes[1]
        for run in self.runs:
            ax2.plot(run["steps"][::4], run["test_loss"][::4],)
       
        ax2.set_xlabel("Gradient steps")
        ax2.set_ylabel("Test loss")
        ax2.set_yscale(scale)
        if ylim_value is not None:
            ax2.set_ylim(top = ylim_value)
        #ax2.text(0.8, 0.85, r"$\mathcal{L}(\Theta)$", transform=ax2.transAxes, bbox=props)
        
        plt.tight_layout()
        
        # Save the first figure (norm & loss)
        if save and folder_name:
            os.makedirs(folder_name, exist_ok=True)  # Ensure folder exists
            file_name = f"{folder_name}/{self.id}_norm_loss.pdf"
            plt.savefig(file_name, bbox_inches="tight")
    
        # Show first figure
        plt.show()
        
        # Third subplot - Realization plot
        if realization:
            if self.test_data.size()[1] == 1:
                fig2, ax3 = plt.subplots(figsize=(8, 6))

                # Extract test data
                x_test = self.test_data.squeeze(1).cpu().detach().numpy()
                y_target = self.target(self.test_data).cpu().detach().squeeze(1).numpy()
        
                # Plot target function
                ax3.plot(x_test, y_target, 'k:', label="Target Function")
                
                # Plot NN approximation
                for m in range(len(self.runs)):
                    ax3.plot(x_test, self.runs[m]['last_output'], label=f"Run {m+1}")
        
                ax3.set_xlabel("x")
                ax3.set_ylabel("Realization function")
                ax3.set_title("NN Approximation vs Target Function (1D)")
                ax3.legend()
        
            elif self.test_data.size()[1] == 2:
                fig2 = plt.figure(figsize=(8, 6))         # Create figure only
                ax3 = fig2.add_subplot(111, projection='3d')  # Now only one axis
                
                ax3 = fig2.add_subplot(111, projection='3d')
        
                # Extract test data for 2D case
                x_test, y_test = self.test_data[:, 0].cpu().detach().numpy(), self.test_data[:, 1].cpu().detach().numpy()
                z_target = self.target(self.test_data).cpu().detach().squeeze(1).numpy()  # True function values
                z_approx = self.runs[0]['last_output'].squeeze(1)  # NN approximation
        
                # Create a grid
                X, Y = np.meshgrid(np.unique(x_test), np.unique(y_test))  
        
                # Reshape Z values
                Z_target = z_target.reshape(X.shape)  
                Z_approx = z_approx.reshape(X.shape)
        
                # Plot the target function as a surface
                ax3.plot_surface(X, Y, Z_target, cmap="viridis", alpha=0.6)
        
                # Overlay the NN approximation as a wireframe
                ax3.plot_wireframe(X, Y, Z_approx, color="red", linewidth=1, alpha=0.8)
        
                # Labels and title
                ax3.set_xlabel('x')
                ax3.set_ylabel('y')
                ax3.set_zlabel('Realization function')
                #ax3.set_title('$\mathcal{N}_{\Theta}(x_1, x_2)$ vs Target')
        
            # Save the second figure (realization plot)
            if save and folder_name:
                file_name = f"{folder_name}/{self.id}_realization.pdf"
                plt.savefig(file_name, bbox_inches="tight")
            # Show realization plot
            fig2.tight_layout()
            plt.show()
            
            
    

    
