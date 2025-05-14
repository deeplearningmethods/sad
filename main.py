import torch
import argparse

from models import Polynomial_ANN, Heat_PDE_ANN, BlackScholes_ANN, MNIST_ANN
from utils import load_config, override_config, load_experiment, plot_mean_norm_and_loss


def run_simulation(config_path, **kwargs):
    
    config = load_config(config_path)
    config = override_config(config, argparse.Namespace(**kwargs))
    # If config is ""  we just plot results
    if config["load_multiple_results"] != "":
        raw_paths = config["load_multiple_results"]
        legend = config["legend"] 
        paths = [group.split(',') for group in raw_paths.split(':')]
        legend = [leg.split(',') for leg in legend.split(':')]
        folder_name = "" if "folder_name" not in config  else config["folder_name"]
        plot_mean_norm_and_loss(
            paths, 
            scale= config["plot_scale"], 
            folder_name= folder_name, 
            legend=legend
        )
        return
    
    if config["load_results"]:
        model = load_experiment(config["folder_name"])
        model.plot_graph(folder_name=config["folder_name"], scale = config["plot_scale"] )
        return model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if "neurons" in config:
        config["neurons"] = [int(n) for n in config["neurons"].split(",")] 
    else:
        raise ValueError("Architecture of the Neural Networks is needed")
    
    
    if config["model_name"] == "polynomial":
        model = Polynomial_ANN(
            neurons=config["neurons"],
            space_bounds=config["space_bounds"],
            train_size=config["train_size"],
            test_size=config["test_size"],
            activation=config["activation"],
            dev=device
        )
    elif config["model_name"] == "heat_pde":
        model = Heat_PDE_ANN(
            neurons=config["neurons"],
            space_bounds=config["space_bounds"],
            T=config["T"],
            rho=config["rho"],
            test_size=config["test_size"],
            activation=config["activation"],
            dev=device
        )
    elif config["model_name"] == "black_scholes":
        model = BlackScholes_ANN(
            neurons=config["neurons"],
            space_bounds=config["space_bounds"],
            T=config["T"],
            r=config["r"],
            c=config["c"],
            K=config["K"],
            sigma=torch.tensor(config["sigma"]),
            mc_samples=config["mc_samples"],
            mc_rounds=config["mc_rounds"],
            test_size=config["test_size"],
            activation=config["activation"],
            dev=device
        )
    elif config["model_name"] == "mnist":
        model = MNIST_ANN(
            neurons=config["neurons"],
            batch_size=config["batch_size"],
            dev=device
        )
    else:
        raise ValueError(f"Unknown model: {config['model_name']}")


    model.train_multiple_runs(
        optimizer=str(config["optimizer"]),
        batch_size = int(config["batch_size"]),
        nr_steps = int(config["nr_steps"]),
        eval_steps = int(config["eval_steps"]),
        lr = float(config["learning_rate"]),
        num_runs = config["num_runs"]
    )
    
    model.plot_graph(scale = config["plot_scale"], folder_name=config["folder_name"], realization = config["realization"] )
    model.save_experiment(config["folder_name"])
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--config", type=str, default = "", help="Path to YAML config file")
    parser.add_argument("--neurons", type=str, help="Comma-separated list of neurons (override)")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--nr_steps", type=int, help="Override training steps")
    parser.add_argument("--activation", type=str, choices=["softplus", "GELU", "ELU", "logistic", "mish", "swish", "softsign", "tanh"], help="Override activation function (e.g., ReLU, GELU, Tanh)")
    parser.add_argument("--train_size", type=int, help="Override train dataset size")
    parser.add_argument("--test_size", type=int, help="Override test dataset size")
    parser.add_argument("--eval_steps", type=int, help="Override evaluation steps")
    parser.add_argument("--optimizer", type=str, choices=['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adamw'], help="Override optimizer (Adam, SGD, RMSprop, Adagrad, Adamw)")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs (default: 5)")
    parser.add_argument("--load_results", type=int, choices=[0, 1], default=0, help="Load and plot results instead of training (1 = True, 0 = False)")
    parser.add_argument("--load_multiple_results", type=str, default="", help="Comma-separated list of folders")
    parser.add_argument("--realization", type=int,choices=[0, 1], default=0, help="Plot realization function in the case of polynomials (1 = True, 0 = False)")
    parser.add_argument("--plot_scale", type=str, choices=["linear", "log"], default="linear", help="Scale for the plot of test loss")
    parser.add_argument("--folder_name", type=str, help="Folder where model and plots are saved")
    parser.add_argument("--legend", type=str, help="Legend when loading multiple results")


    args = parser.parse_args()
    
    override_args = {k: v for k, v in vars(args).items() if v is not None}
    model = run_simulation(config_path=args.config,**override_args)
    
  
    
  
