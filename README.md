# Scout Optimizer

Scout is a [Bayesian optimizer](https://en.wikipedia.org/wiki/Bayesian_optimization_) designed to be as easy to use as possible by limiting the number of options and tuning parameters through the use of experts and internal searches to determine appropriate configurations. The only parameters needed to run an optimization with scout, is the objective function, the problem design variable bounds, and the evaluation budget. The goal of Scout is to produce the best possible solution given the evaluation budget. The primary use case for Scout is on expensive black box functions where only a limited number of of function evaluations are possible.

### Installation

Scout can be installed directly from Github using pip

```bash
pip install git+https://github.com/jldaniel/ScoutOpt.git
``` 

or by downloading the source code and calling pip on the directory.

```bash
git clone https://github.com/jldaniel/ScoutOpt.git
pip -e install ScoutOpt
```

### Usage

Scout can be used by importing the module, creating a new Scout object, setting the problem and bounds, then simply calling `optimize` and providing an evaluation budget.

```python
import numpy as np
from scout import Scout

# This is the loss function we are going to optimize
def gramacy_lee(x):
    return np.sin(10 * np.pi * x) / (2 * x) + np.power(x - 1., 4.)

# The input variable bounds for the loss function
bounds = np.array([[0.5, 2.5]])

# The total number of function evaluations to perform during the search
evaluation_budget = 30

# Set up the optimizer
opt = Scout(gramacy_lee, bounds, seed=42)

# Run the optimization
x_best, y_best = opt.optimize(evaluation_budget)

# Display the results
print('Best x: ' + repr(x_best))
print('Best y: ' + repr(y_best))
```

### Algorithm

Scout uses the following algorithm

1. Given the evaluation budget, select a portion to use as an initial training set for a surrogate model.
2. A mixed design of experiments is generated combining a regular full-factorial design and a random latin-hypercube design
3. An initial surrogate model is trained using Gaussian process regression and an grid search to determine the best kernel
4. For the remaining evaluation budget a new point is selected via an attainment function which initially favors exploration of the design space then gradually shifts to prefer exploitation of the objective.
5. As each new point is evaluated, the surrogate is updated
