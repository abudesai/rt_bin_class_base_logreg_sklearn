
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 



from sklearn.linear_model import LogisticRegression


model_fname = "model.save"
MODEL_NAME = "LogisticRegression_sklearn"

COST_THRESHOLD = float('inf')


# class InfCostStopCallback(Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         loss_val = logs.get('loss')
#         if(loss_val == COST_THRESHOLD or tf.math.is_nan(loss_val)):
#             print("Cost is inf, so stopping training!!")
#             self.model.stop_training = True


class LogisticRegression_sklearn(): 
    
    def __init__(self, penalty="l2", C = 1.0, max_iter = 100, l1_ratio = None, **kwargs) -> None:
        super(LogisticRegression_sklearn, self).__init__(**kwargs)
        self.penalty = penalty
        self.C = np.float(C)
        self.max_iter = int(max_iter) 
        if self.penalty == 'elasticnet':
            self.l1_ratio = l1_ratio if l1_ratio is not None and 0 <= l1_ratio <= 1 else 0.5
        else: 
            self.l1_ratio = None
        self.model = self.build_model()     
        
        
    def build_model(self): 
        model = LogisticRegression(penalty = self.penalty, C = self.C, max_iter = self.max_iter, solver='saga', l1_ratio=self.l1_ratio)
        return model
    
    
    def fit(self, train_X, train_y):        
        self.model.fit(train_X, train_y)            
        
    
    def predict(self, X, verbose=False): 
        preds = self.model.predict(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)        

    
    def save(self, model_path): 
        joblib.dump(self.model, os.path.join(model_path, model_fname))
        


    @classmethod
    def load(cls, model_path):         
        logisticregression = joblib.load(os.path.join(model_path, model_fname))
        # print("where the load function is getting the model from: "+ os.path.join(model_path, model_fname))        
        return logisticregression


def save_model(model, model_path):
    # print(os.path.join(model_path, model_fname))
    joblib.dump(model, os.path.join(model_path, model_fname)) #this one works
    # print("where the save_model function is saving the model to: " + os.path.join(model_path, model_fname))
    

def load_model(model_path): 
    try: 
        model = joblib.load(os.path.join(model_path, model_fname))   
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


