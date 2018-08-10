
import logging

import numpy as np
from scipy.optimize import differential_evolution


from .surrogate import Surrogate
from .doe import mixed_doe


# TODO Feature: Add ability to supply initial training set
# TODO Feature: Add flag for maximization problems
class Scout(object):
    """
    Bayesian optimization
    """

    def __init__(self, fun, bounds, seed=None, logger=None):
        # TODO Add logger
        self.fun = fun
        self.bounds = np.array(bounds)
        self.exploration_factor = 10.0
        self.training_ratio = 0.1
        self.seed = seed

        if not logger:
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())
        else:
            self.logger = logger

    def optimize(self, evaluation_budget):
        """
        Find the best value of the function in the given evaluation budget
        :param evaluation_budget: The number of function evaluations to use in the search
        :return: x_opt, y_opt
        """
        # Get the initial training set
        n_training_points = np.floor(np.cbrt(self.training_ratio*evaluation_budget**3))
        x_train = mixed_doe(n_training_points, self.bounds, seed=self.seed)
        y_train = np.apply_along_axis(self.fun, 1, x_train)

        # Generate the initial surrogate
        surrogate = Surrogate().fit(x_train, y_train)

        # Get the current best design
        best_index = np.argmin(y_train)
        x_best = x_train[best_index]
        y_best = y_train[best_index]

        # Start the main optimization loop
        current_evaluation = len(x_train)

        while current_evaluation <= evaluation_budget:
            # Get the next sample
            exploration_parameter = (-1./np.power(evaluation_budget, 3.))*np.power(current_evaluation, 3.) + 1
            exploration_parameter = self.exploration_factor*exploration_parameter
            sample = self.get_next_sample(surrogate, self.bounds, exploration_parameter)

            y_cur = self.fun(sample)

            if y_cur < y_best:
                x_best = sample
                y_best = y_cur

            # Update the model with the new point
            x_train = np.append(x_train, sample, axis=0)
            y_train = np.append(y_train, y_cur, axis=0)
            surrogate = surrogate.fit(x_train, y_train)

            current_evaluation += 1

        return x_best, y_best

    @staticmethod
    def get_next_sample(surrogate, bounds, exploration_parameter, maximize=False):
        n_dim = bounds.shape[0]

        result = differential_evolution(func=Scout.upper_confidence_bound,
                                        bounds=bounds,
                                        popsize=10*n_dim,
                                        init='random',
                                        maxiter=50,
                                        tol=1e-5,
                                        args=(surrogate, exploration_parameter))

        best_x = result.x
        return np.array([best_x])

    @staticmethod
    def upper_confidence_bound(X, model, exploration_parameter):
        """
        The upper confidence bound attainment function
        :param X: Array like, points to evaluate the attainment function at
        :param model: The gaussian process model to use
        :param exploration_parameter: The upper confidence bound free parameter, larger values favor exploration
        :return: The attainment value at the specified points
        """
        n_dim = model.X_train_.shape[1]
        mean, std_dev = model.predict(X.reshape(-1, n_dim), return_std=True)
        mean = mean.reshape(-1, 1)
        std_dev = std_dev.reshape(-1, 1)

        return mean + exploration_parameter*std_dev




