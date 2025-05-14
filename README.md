# SAD neural networks
This repository is the official implementation of *SAD Neural Networks: Divergent Gradient Flows and Asymptotic Optimality via o-minimal Structures*. 
It provides implementations for showing the divergence phenomena of neural networks parameters during the training of:
- supervised learning with polynomial target function (`polynomial.yaml`) 
- deep Kolmogorov method for Heat PDE (`heat.yaml`)  
- deep Kolmogorov method for Black-Scholes PDE (`black_scholes.yaml`)  
- MNIST Classification (`mnist.yaml`). 

In the simulations involving polynomials, we validate the results established in the paper (see Corollary 3.6). The real-world examples (Heat PDE, Black-Scholes PDE and MNIST classification) confirm the same behavior, suggesting that the phenomena extend beyond the theoretical setting and are relevant in broader practical contexts. 

## Repository Structure
```
SAD/
├── configs/                     # Configuration files
│   ├── __init__.py               
│   ├── black_scholes.yaml       
│   ├── heat.yaml                
│   └── mnist.yaml               
│   └── polynomial.yaml           
├── models/		         # Neural network definition
│   ├── __init__.py                  
│   ├── black_scholes_ann.py     
│   ├── heat_pde_ann.py          
│   ├── mnist_ann.py             
│   ├── model.py                 # Base class for the all the models
│   └── polynomial_ann.py        
├── trained_models/              # Pre-trained models used in the paper 
│   ├── real_world/              # MNIST, heat PDE and Black-Scholes PDE models           
│   └── polynomial/              # Polynomial target function models
├── LICENSE.txt                  # License
├── main.py                      # Main training script
├── requirements.txt             # Project dependencies
└── utils.py                     # Helper functions
```


## Requirements

The necessary python dependencies can be installed by:
```setup
pip install -r requirements.txt
```
This project was developed and tested using:
```setup
Python 3.11.5
```
Make sure to use this version (or a compatible one) to avoid potential compatibility issues.


## Training 
To train a model, run one of the following commands, specifying one of the available .yaml configuration files:
`polynomial.yaml`, `heat.yaml`, `black_scholes.yaml` or `mnist.yaml`. Additional parameters can be overridden via command-line arguments:

```train
python main.py --config configs/heat.yaml --nr_steps 200000 --batch_size 1024 --activation tanh --num_runs 4
python main.py --config configs/black_scholes.yaml
python main.py --config configs/mnist.yaml --nr_steps 30000
python main.py --config configs/polynomial.yaml --activation GELU --neurons 1,10,20,10,1
```

To reproduce the results reported in the paper for the Heat PDE, Black-Scholes PDE, and MNIST classification task, it is sufficient to run the corresponding command with the appropriate configuration file and no additional arguments.
For the polynomial target function, depending on which model is wanted to replicate, specify:
- the activation function (`softplus`, `GELU`, `ELU`, `swish`, `tanh`)
- the optimizer (`SGD`, `Adam`)
- the neural network architecture, given as a comma-separated list of neurons (e.g. `1,10,20,10,1`). 

Below are the exact commands used in the paper to train the models with the GELU activation (the other activations are equivalent). Note that the `test_size` argument determines the number of points per input dimension forming a test grid. For instance, in the 2D case, a grid of `test_size x test_size` points results in `test_size^2` total test data.

```train
# 1D case, SGD
python main.py --config configs/polynomial.yaml --activation GELU --neurons 1,10,20,10,1 --optimizer SGD --num_runs 20
# 1D case, Adam
python main.py --config configs/polynomial.yaml --activation GELU --neurons 1,10,20,10,1 --learning_rate 0.005 --optimizer Adam --num_runs 20
# 2D case, Adam
python main.py --config configs/polynomial.yaml --activation GELU --neurons 2,20,40,20,1 --learning_rate 0.005 --test_size 100 --optimizer Adam --num_runs 20
# 4D case, Adam
python main.py --config configs/polynomial.yaml --activation GELU --neurons 4,20,40,20,1 --learning_rate 0.005 --test_size 15 --optimizer Adam --num_runs 20 
```


## Loading and visualizing model

All trained models and plots are stored in folders named after the corresponding experiment.
To visualize results from multiple models, averaging over the runs as in the figures of the paper, use the `load_multiple_results` argument. 
Model folders should be separated by a comma to include them in the same plot, and by a colon to separate them into different plots within the same figure.
To save the figure in a specific folder, provide the desired path using the optional `folder_name` argument.
The figures from the paper can be reproduced using the commands:

```eval
python main.py \
	--load_multiple_results \
	trained_models/real_world/heat_pde,trained_models/real_world/blackscholes_pde:\
 	trained_models/real_world/mnist\
	--legend "Heat PDE,Black-Scholes PDE:MNIST"\
	--plot_scale log
```

```eval
python main.py \
	--load_multiple_results \
	trained_models/polynomial/swish_SGD,trained_models/polynomial/softplus_SGD,trained_models/polynomial/GELU_SGD,trained_models/polynomial/ELU_SGD,trained_models/polynomial/tanh_SGD:\
	trained_models/polynomial/swish_Adam,trained_models/polynomial/softplus_Adam,trained_models/polynomial/GELU_Adam,trained_models/polynomial/ELU_Adam,trained_models/polynomial/tanh_Adam:\
	trained_models/polynomial/swish_in2_Adam,trained_models/polynomial/softplus_in2_Adam,trained_models/polynomial/GELU_in2_Adam,trained_models/polynomial/ELU_in2_Adam,trained_models/polynomial/tanh_in2_Adam:\
	trained_models/polynomial/swish_in4_Adam,trained_models/polynomial/softplus_in4_Adam,trained_models/polynomial/GELU_in4_Adam,trained_models/polynomial/ELU_in4_Adam,trained_models/polynomial/tanh_in4_Adam \
	--legend "Swish,Softplus,GELU,ELU,Hyperbolic tangent" \
	--plot_scale log
```


To load a previously trained model and visualize the results across the runs, use:

```eval
python main.py --config configs/heat.yaml --load_results 1
```

For polynomial tasks, also specify the folder name. 

```eval
python main.py --config configs/polynomial.yaml --load_results 1 --folder_name trained_models/polynomial/GELU_SGD
```
Each activation has trained models with SGD and Adam for 1D inputs, and Adam for 2D and 4D inputs.



## License

This repository is licensed under the [MIT License](LICENSE.txt).
