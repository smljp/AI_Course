#   Copyright (C) 2017 All rights reserved.
#
#   filename : Logistic_Regression.py
#   author   : chendian / okcd00@qq.com
#   date     : 2017-11-13
#   desc     : Theano Logistic Regression Tutorial
#   refer    : Vincent.Y & zhiyong_will @ CSDN
#   
# ============================================================================

from __future__ import print_function

import os
import sys
# os.environ["THEANO_FLAGS"] = 'device=cuda3'
os.environ["THEANO_FLAGS"] = 'device=cpu'

import numpy as np
import theano
import theano.tensor as T
from theano import function, printing

import matplotlib.pyplot as plt

# it can be set in file ~/.theanorc
theano.config.floatX = 'float64'


# use data from sklearn package
def load_moons():
    from sklearn.datasets import make_moons
    np.random.seed(0)
    X, y = make_moons(800, noise=0.2)
    print ("dataset shape:", X.shape)
    
    # return train validate test sets 
    # return [(X[0:600,],y[0:600,]), (X[600:800,],y[600:800,])]
    return [(np.concatenate((X[0:600,], np.array([[3,3]]*80)), axis=0),
    np.concatenate((y[0:600,], np.array([0]*80)), axis=0)), (X[600:800,],y[600:800,])]

def load_circles():
    from sklearn.datasets import make_circles
    np.random.seed(0)
    X, y = make_circles(800, noise=0.2, factor=0.5, random_state=2)
    print ("dataset shape:", X.shape)
    
    # return train validate test sets 
    return [(X[0:600,],y[0:600,]), (X[600:800,],y[600:800,])]

def load_linear():
    from sklearn.datasets import make_classification
    np.random.seed(0)
    X, y = make_classification(
        800, n_features=2, n_redundant=0, n_informative=1,
        random_state=1, n_clusters_per_class=1)
    print ("dataset shape:", X.shape)
    
    # return train validate test sets 
    # return [(X[0:600,],y[0:600,]), (X[600:800,],y[600:800,])]
    return [(np.concatenate((X[0:600,], np.array([[3,3]]*30),np.array([[3,2]]*15),np.array([[2,2]]*15)), axis=0),
    np.concatenate((y[0:600,], np.array([0]*60)), axis=0)), (X[600:800,],y[600:800,])]


def load_data(name='moons'):
    _datasets={
        'moons': load_moons,
        'linear': load_linear,
        'circles': load_circles,
    }
    return _datasets[name]()
    
class LogisticRegression():
    def __init__(self, X, n_in, n_out):
        # Important: shared()
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX),
            name='Weight',
            borrow=True
        )

        self.b=theano.shared(
            value=np.zeros(
                (n_out, ),
                dtype=theano.config.floatX),
            name='bias',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(X,self.W)+self.b) 
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)           
        self.params = [self.W, self.b]
        self.X = X

    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])


    def errors(self,y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y',y.type, 'y_pred',self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            return NotImplementedError()

def sgd_optimization(datasets, learning_rate=0.12, n_epochs=300, draw_freq=60):
    train_set_x, train_set_y = datasets[0]
    test_set_x,  test_set_y  = datasets[1]

    index = T.lscalar() # long-Scalar 
    y = T.lvector('y')  # long-Vector /1 dim 
    x = T.matrix('x')   # matrix      /2 dim 
                        # tensor      /3 dim 
    
    classifier = LogisticRegression(X=x, n_in=2, n_out=2)  # Classifier
    cost = classifier.negative_log_likelihood(y)           # Cost or Loss_Function

    g_W = T.grad(cost=cost, wrt=classifier.W) 
    g_b = T.grad(cost=cost, wrt=classifier.b) 

    updates=[(classifier.W, classifier.W - learning_rate * g_W),
             (classifier.b, classifier.b - learning_rate * g_b)]

    train_model=function(
        inputs=[x,y],
        outputs=classifier.errors(y),
        updates=updates)
    
    test_model=function(
        inputs=[x, y],
        outputs=classifier.errors(y)
    )
    
    predict_model=function(
        inputs=[x],
        outputs=classifier.y_pred)
    
    epoch = 0
    while(epoch < n_epochs):
        # draw a figure every 'draw_freq' times
        if epoch % draw_freq == 0:
            plot_decision_boundary(lambda x:predict_model(x), train_set_x, train_set_y)
        # print cost per epoch
        avg_cost = train_model(train_set_x, train_set_y)
        test_cost = test_model(test_set_x, test_set_y)
        print ("epoch is %d,train error %f, test error %f" % (epoch,avg_cost,test_cost))
        epoch += 1
    # draw a figure at last        
    plot_decision_boundary(lambda x:predict_model(x), train_set_x, train_set_y)
        

def plot_decision_boundary(pred_func, train_set_x, train_set_y):
    # Draw figures as Matlab 
    x_min, x_max = train_set_x[:, 0].min() - .5, train_set_x[:, 0].max() + .5
    y_min, y_max = train_set_x[:, 1].min() - .5, train_set_x[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(train_set_x[:, 0], train_set_x[:, 1], c=train_set_y, cmap=plt.cm.Spectral)
    plt.show()


    
if __name__=="__main__":
    data = load_data('linear')
    sgd_optimization(data, n_epochs=300, draw_freq=50)