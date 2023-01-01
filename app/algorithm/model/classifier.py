
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 



from sklearn.linear_model import LogisticRegression


model_fname = "model.save"

MODEL_NAME = "bin_class_base_logistic_regression_sklearn"

COST_THRESHOLD = float('inf')



class Classifier(): 
    
    def __init__(self, penalty="elasticnet", C = 1.0, l1_ratio=0.5, **kwargs) -> None:
        self.penalty = penalty
        self.C = np.float(C)
        if self.penalty == 'elasticnet':
            self.l1_ratio = l1_ratio if l1_ratio is not None and 0 <= l1_ratio <= 1 else 0.5
        else: 
            self.l1_ratio = None
        self.model = self.build_model()     
        
        
    def build_model(self): 
        model = LogisticRegression(penalty = self.penalty, C = self.C, solver='saga', l1_ratio=self.l1_ratio)
        return model
    
    
    def fit(self, train_X, train_y):        
        self.model.fit(train_X, train_y)            
        
    
    def predict(self, X, verbose=False): 
        preds = self.model.predict(X)
        return preds          
        
    
    def predict_proba(self, X, verbose=False): 
        preds = self.model.predict_proba(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)        

    
    def save(self, model_path): 
        joblib.dump(self, os.path.join(model_path, model_fname))
        


    @classmethod
    def load(cls, model_path):         
        model = joblib.load(os.path.join(model_path, model_fname))
        return model


def save_model(model, model_path):
    model.save(model_path)
    

def load_model(model_path): 
    model = Classifier.load(model_path)   
    return model


