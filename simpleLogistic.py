import numpy as np
import math 


def sigmoid(z):
    value=np.zeros(z.shape)
    value=1/(1+np.exp(-z))
    return value

def derivSig(z):
    deri=sigmoid(z)*(1-sigmoid(z))
    return deri

def logistic_loss(X, y, W, reg):
        loss=0.0
        grad=np.zeros(W.shape)
        m=y.size
        Hypothesis=X.dot(W)
        H=sigmoid(Hypothesis)
        
        lossI=y*np.log(H)+(1-y)*np.log(1-H)
        loss=lossI.sum()/m
        loss=-1*loss
        regterm=reg*(np.sum(W*W)-np.sum(W[0]*W[0]))/(2*m)
        # regterm=reg*(np.sum(W*W))/(2*m)
        loss+=regterm
        
        some=(X.T).dot((H-y))
       
        grad=(some+ reg*W)/m
        grad[0]=some[0]/m
        
        return loss, grad        

def logistic_train(X, y, W, reg, learning_rate, iterations):
        cost_history=[]
        X=np.hstack((np.ones((y.size,1)), X))
        Weights=np.array(W)
        num_train=y.size
        batch_size=1200
        y=y.reshape(num_train, 1)
        for epoch in range(iterations):
            # indices=np.random.choice(num_train, batch_size, replace=False)
            # X_in=X[indices]
            # y_in=y[indices]
            # loss, grad=logistic_loss(X_in, y_in, Weights, reg)
            loss, grad=logistic_loss(X, y, Weights, reg) 
            cost_history.append(loss)
            Weights=Weights-learning_rate*grad
            
        return cost_history, Weights

def logistic_predict(X, W, threshold):
        X=np.hstack((np.ones((X.shape[0],1)), X))
        class_pred=sigmoid(X.dot(W))
        y_pred=np.zeros(X.shape[0])
        y_pred=(class_pred>=threshold)
        
        return y_pred


def logistic_accuracy(X, y, W, threshold):
        y_pred=logistic_predict(X, W, threshold)
        y_correct=np.mean(y_pred==y)
        return y_correct 