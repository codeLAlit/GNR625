import numpy as np

def svm_loss(X, y, W, reg):
        loss=0.0
        grad=np.zeros(W.shape)
        numClass=W.shape[1]
        numTrain=X.shape[0]

        predict=X.dot(W)
        correct_prediction=predict[np.arange(numTrain), y].reshape(numTrain, 1)
        margin=np.maximum(0, predict-correct_prediction+1)
        margin[np.arange(numTrain), y]=0
        loss=margin.sum()/numTrain
        loss+=reg*np.sum(W*W)

        margin[margin>0]=1
        valid_margins=margin.sum(axis=1)
        margin[np.arange(numTrain), y] -=valid_margins
        grad=((X.T).dot(margin))/numTrain
        grad=grad+2*reg*(W)

        return loss, grad



def svm_train(X, y, W, reg, learning_rate, iterations):
        Weights=np.array(W)
        num_train=y.size
        batch_size=1200
        X=np.hstack((np.ones((X.shape[0],1)), X))
        cost_history=[]
        for epoch in range(iterations):
            indices=np.random.choice(num_train, batch_size, replace=False)
            X_in=X[indices]
            y_in=y[indices]
            loss, grad=svm_loss(X_in, y_in, Weights, reg)
            #loss, grad=svm_loss(X, y, W, reg)
            cost_history.append(loss)
            Weights=Weights-learning_rate*grad
        return cost_history, Weights


def svm_predict(X, W):
        X=np.hstack((np.ones((X.shape[0],1)), X))
        class_pred=X.dot(W)
        y_pred=np.zeros(X.shape[0])
        y_pred=class_pred.argmax(axis=1)
        return y_pred


def svm_accuracy(X, y, W):
        y_pred=svm_predict(X, W)
        y_correct=np.mean(y_pred==y)
        return y_correct

