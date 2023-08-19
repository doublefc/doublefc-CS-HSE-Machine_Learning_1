from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
        depth: int = None
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators
        self.depth = depth
        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)
        
        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):

        model = self.base_model_class()
        indexes = np.random.randint(0, x.shape[0], round(self.subsample*x.shape[0]))

        x_sub = x[indexes]
        y_sub = y[indexes]
        pred_sub = predictions[indexes]
        #print(len(y_sub), len(pred_sub))
        target = self.loss_derivative(y_sub, pred_sub)
        #print(y_sub, pred_sub, target)
        model.fit(x_sub, target)

        new_pred = model.predict(x_sub)
        #print(new_pred, pred_sub)
        self.gammas.append(self.find_optimal_gamma( y_sub, pred_sub, new_pred))
        self.models.append(model)

    def fit(self, x_train, y_train, x_valid, y_valid):

        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        
        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions += self.gammas[-1] *self.learning_rate*self.models[-1].predict(x_train)
            #print(self.gammas[-1] * self.models[-1].predict(x_train), self.gammas[-1], self.models[-1].predict(x_train))
            valid_predictions += self.gammas[-1] *self.learning_rate*self.models[-1].predict(x_valid)
           # print(self.predict_proba(x_train).shape, len(y_train))
            train_score = self.loss_fn( y_train, self.predict_proba(x_train)[:,1])
            #print(train_predictions)
            val_score = self.loss_fn( y_valid, self.predict_proba(x_valid)[:,1])
                           
            self.history['train'].append(train_score)
            self.history['val'].append(val_score) 
            try:
            
                if np.all(self.validation_loss <= val_score):
                    break
                else:
                    self.validation_loss.pop(0)
                    self.validation_loss = np.append(self.validation_loss, val_score)
            except:
                pass


        if self.plot:
            len_ = len( self.history['train'])
            plt.plot(range(len_), self.history['train'], label='train')
            plt.plot(range(len_), self.history['train'], label='val')
            plt.title('Train and val scores during learning')
            plt.ylabel('Scores')
            plt.legend()
            plt.show()
        #print(train_predictions[:10], np.mean(train_predictions))                
    def predict_proba(self, x):
        result = 0
        for gamma, model in zip(self.gammas, self.models):
            result += gamma*self.learning_rate*model.predict(x)
        a = self.sigmoid(result)
        b = 1-self.sigmoid(result)
      
        c = np.array([[a[i], b[i]] for i in range(len(a))])
        #print(c)
        return c

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0.1, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + self.learning_rate*gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        importances = np.zeros_like(self.models[0].feature_importances_)
        for model in self.models:
            importances += model.feature_importances_
        importances /= len(importances)
        return importances/sum(importances)
