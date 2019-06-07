from builtins import range
from builtins import object
import numpy as np

from layers import *


class FullyConnectedNet(object):
    """
    A fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of [H, ...], and perform classification over C classes.

    The architecure should be like affine - relu - affine - softmax for a one
    hidden layer network, and affine - relu - affine - relu- affine - softmax for
    a two hidden layer network, etc.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim, hidden_dim=[10, 5], num_classes=8,
                 weight_scale=0.1):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: A list of integer giving the sizes of the hidden layers
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        """
        self.params = {}
        self.hidden_dim = hidden_dim

        ############################################################################
        # Initialize the weights and biases of the net. Weights                    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # number of layers
        self.layers = 1 + len(hidden_dim)
        modif_hidden_dims = [input_dim] + hidden_dim + [num_classes]
        for i in range(0, self.layers):
            W = 'W' + str(i + 1)
            b = 'b' + str(i + 1)
            self.params[b] = np.zeros(modif_hidden_dims[i + 1])
            self.params[W] = weight_scale * np.random.normal(size=(modif_hidden_dims[i], modif_hidden_dims[i + 1]))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """

        scores = X
        ############################################################################
        # TODO: Implement the forward pass for the net, computing the              #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        cache = {}
        scores, cache['cA1'] = affine_forward(scores, self.params['W1'], self.params['b1'])
        for i in range(1, self.layers):
            W = 'W' + str(i + 1)
            b = 'b' + str(i + 1)
            cA = 'cA' + str(i + 1)
            cR = 'cR' + str(i + 1)
            scores, cache[cR] = relu_forward(scores)
            scores, cache[cA] = affine_forward(scores, self.params[W], self.params[b])


        if y is None:
            return scores


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        ############################################################################
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k].                                                          #
        ############################################################################
        loss, grads = 0.0, {}
        loss, dx = softmax_loss(scores, y)
        regParam = 10
        dx, dw, db = affine_backward(dx, cache['cA' + str(self.layers)])
        grads['W' + str(self.layers)] = dw + regParam * self.params['W' + str(self.layers)]
        grads['b' + str(self.layers)] = db
        for i in range(self.layers - 1, 0, -1):
            W = 'W' + str(i)
            b = 'b' + str(i)
            cA = 'cA' + str(i)
            cR = 'cR' + str(i + 1)
            dx = relu_backward(dx, cache[cR])
            dx, dw, db = affine_backward(dx, cache[cA])
            grads[W] = dw + regParam*self.params[W]
            grads[b] = db + self.params[b]

            loss += 0.5 * np.sum(np.square(self.params['W' + str(i)]))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################




        return loss, grads

if __name__=="__main__":
    path = # Type your path here
    from data_utils import load_mds189

    # load the dataset
    debug = False  # OPTIONAL: you can change this to True for debugging *only*. Your reported results must be with debug = False
    feat_train, label_train, feat_val, label_val = load_mds189(path, debug)
    from solver import Solver

    data = {
        'X_train': feat_train,
        'y_train': label_train,
        'X_val': feat_val,
        'y_val': label_val}

    hyperparams = {'lr_decay': .5,
                   'num_epochs': 30,
                   'batch_size': 10,

                   'learning_rate': 0.0001
                   }

    hidden_dim = [50, 50]  # this should be a list of units for each hiddent layer

    model = FullyConnectedNet(input_dim=75,
                              hidden_dim=hidden_dim)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                        'learning_rate': hyperparams['learning_rate'],
                    },
                    lr_decay=hyperparams['lr_decay'],
                    num_epochs=hyperparams['num_epochs'],
                    batch_size=hyperparams['batch_size'],
                    print_every=100)
    solver.train()