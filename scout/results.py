

class OptimizationResults(object):
    def __init__(self):
        self.seed = None
        self.x_best = None
        self.y_best = None
        self.surrogate = None
        self.x_train = None
        self.y_train = None
        self.surrogates = []
        self.x_best_history = []
        self.y_best_history = []
        self.exploration_parameter_history = []

    def update_best(self, x_best, y_best):
        self.x_best = x_best
        self.y_best = y_best
        self.x_best_history.append(x_best)
        self.y_best_history.append(y_best)

