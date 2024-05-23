from domain import vec
from optimizers import SGD, Nesterov, RMSProp, Adadelta, Momentum, Adam, Adamax, Nadam, AMSGrad, Adagrad
from presentation import show_results
from test_functions import all_functions

optimizers = [SGD, Momentum, Nesterov, Adagrad, RMSProp, Adadelta, Adam, Adamax, Nadam, AMSGrad]
labels = ['SGD', 'Momentum', 'Nesterov', 'Adagrad', 'RMSProp', 'Adadelta', 'Adam', 'Adamax', 'Nadam', 'AMSGrad']
X = [vec(4, 4), vec(4, 4), vec(3, 2), vec(6, 5), vec(0.6, -1.5)]

params1 = {
    'SGD': {'e': 0.01, 'lr': 0.1},
    'Momentum': {'e': 0.01, 'lr': 0.1, 'gamma': 0.9},
    'Nesterov': {'e': 0.01, 'lr': 0.1, 'gamma': 0.9},
    'Adagrad': {'e': 0.01, 'lr': 1.0, 'eps': 0.0001},
    'RMSProp': {'e': 0.01, 'lr': 1.0, 'gamma': 0.95, 'eps': 0.1},
    'Adadelta': {'e': 0.01, 'lr': 1.0, 'gamma': 0.95, 'eps': 0.1},
    'Adam': {'e': 0.01, 'lr': 1.0, 'b1': 0.95, 'b2': 0.9, 'eps': 0.1},
    'Adamax': {'e': 0.01, 'lr': 1.0, 'b1': 0.95, 'b2': 0.99},
    'Nadam': {'e': 0.01, 'lr': 1.0, 'b1': 0.95, 'b2': 0.9, 'eps': 0.1},
    'AMSGrad': {'e': 0.01, 'lr': 1.0, 'b1': 0.95, 'b2': 0.9, 'eps': 0.1},
}

params2 = {
    'SGD': {'e': 0.01, 'lr': 0.001},
    'Momentum': {'e': 0.01, 'lr': 0.005, 'gamma': 0.9},
    'Nesterov': {'e': 0.01, 'lr': 0.01, 'gamma': 0.9},
    'Adagrad': {'e': 0.01, 'lr': 1.0, 'eps': 0.01},
    'RMSProp': {'e': 0.01, 'lr': 0.01, 'gamma': 0.95, 'eps': 10.0},
    'Adadelta': {'e': 0.01, 'lr': 0.01, 'gamma': 0.95, 'eps': 0.1},
    'Adam': {'e': 0.01, 'lr': 0.1, 'b1': 0.9, 'b2': 0.9, 'eps': 0.1},
    'Adamax': {'e': 0.01, 'lr': 0.1, 'b1': 0.95, 'b2': 0.99},
    'Nadam': {'e': 0.01, 'lr': 0.1, 'b1': 0.95, 'b2': 0.9, 'eps': 0.1},
    'AMSGrad': {'e': 0.01, 'lr': 1.0, 'b1': 0.9, 'b2': 0.999, 'eps': 0.01},
}

params3 = {
    'SGD': {'e': 0.01, 'lr': 0.01},
    'Momentum': {'e': 0.01, 'lr': 0.01, 'gamma': 0.95},
    'Nesterov': {'e': 0.01, 'lr': 0.01, 'gamma': 0.95},
    'Adagrad': {'e': 0.01, 'lr': 1.0, 'eps': 0.01},
    'RMSProp': {'e': 0.01, 'lr': 0.1, 'gamma': 0.95, 'eps': 0.1},
    'Adadelta': {'e': 0.01, 'lr': 0.1, 'gamma': 0.95, 'eps': 0.1},
    'Adam': {'e': 0.01, 'lr': 0.1, 'b1': 0.9, 'b2': 0.9, 'eps': 0.1},
    'Adamax': {'e': 0.01, 'lr': 0.01, 'b1': 0.99, 'b2': 0.99},
    'Nadam': {'e': 0.01, 'lr': 0.1, 'b1': 0.95, 'b2': 0.9, 'eps': 0.1},
    'AMSGrad': {'e': 0.01, 'lr': 0.1, 'b1': 0.95, 'b2': 0.99, 'eps': 0.1},
}

params4 = {
    'SGD': {'e': 0.01, 'lr': 0.01},
    'Momentum': {'e': 0.01, 'lr': 0.01, 'gamma': 0.98},
    'Nesterov': {'e': 0.01, 'lr': 0.1, 'gamma': 0.95},
    'Adagrad': {'e': 0.01, 'lr': 1.0, 'eps': 0.01},
    'RMSProp': {'e': 0.01, 'lr': 0.1, 'gamma': 0.95, 'eps': 0.1},
    'Adadelta': {'e': 0.01, 'lr': 0.1, 'gamma': 0.95, 'eps': 0.1},
    'Adam': {'e': 0.01, 'lr': 0.1, 'b1': 0.97, 'b2': 0.9, 'eps': 0.001},
    'Adamax': {'e': 0.01, 'lr': 0.1, 'b1': 0.9, 'b2': 0.99},
    'Nadam': {'e': 0.01, 'lr': 0.1, 'b1': 0.97, 'b2': 0.9, 'eps': 0.1},
    'AMSGrad': {'e': 0.01, 'lr': 0.1, 'b1': 0.9, 'b2': 0.99, 'eps': 0.1},
}

params5 = {
    'SGD': {'e': 0.01, 'lr': 0.01},
    'Momentum': {'e': 0.01, 'lr': 0.01, 'gamma': 0.9},
    'Nesterov': {'e': 0.01, 'lr': 0.005, 'gamma': 0.97},
    'Adagrad': {'e': 0.01, 'lr': 0.1, 'eps': 0.01},
    'RMSProp': {'e': 0.01, 'lr': 0.01, 'gamma': 0.95, 'eps': 0.1},
    'Adadelta': {'e': 0.01, 'lr': 0.01, 'gamma': 0.95, 'eps': 0.1},
    'Adam': {'e': 0.01, 'lr': 0.01, 'b1': 0.9, 'b2': 0.999, 'eps': 0.1},
    'Adamax': {'e': 0.01, 'lr': 0.1, 'b1': 0.9, 'b2': 0.999},
    'Nadam': {'e': 0.01, 'lr': 0.1, 'b1': 0.9, 'b2': 0.999, 'eps': 0.1},
    'AMSGrad': {'e': 0.01, 'lr': 0.01, 'b1': 0.9, 'b2': 0.999, 'eps': 0.1},
}

show_results([all_functions[4]], optimizers, params5, labels, X[4])
# show_results([all_functions[1]], [AMSGrad], params2, ['AMSGrad'], X[1])