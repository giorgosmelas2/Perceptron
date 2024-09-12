import numpy as np
import matplotlib.pyplot as plt
import random

class Perceptron:
    def __init__(self, N, alpha=0.1):
        #initialize the weight matrix and store the learning rate
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self ,x):
        #apply the step function
        return 1 if x > 0 else 0
    
    def fit(self, X, y, epochs=10):
        #adds a column of 1 in the end
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):

            #zip comnines X values with targets
            for (x, target) in zip(X, y):

                p = self.step(np.dot(x, self.W))

                if p != target:
                    error = p - target
                    self.W = self.W - self.alpha * error * x

        print("[INFO] Test completed.")
    
    def predict(self, X, addBias=True):

        #ensure the input is a matrix
        X = np.atleast_2d(X)

        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]

        return self.step(np.dot(X, self.W))

