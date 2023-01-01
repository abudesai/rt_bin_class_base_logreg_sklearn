import numpy as np, pandas as pd
import os
import sys

import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.classifier as classifier


# get model configuration parameters
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema):
        self.model_path = model_path
        self.data_schema = data_schema
        self.id_field_name = self.data_schema["inputDatasets"][
            "binaryClassificationBaseMainInput"
        ]["idField"]
        self.preprocessor = None
        self.model = None

    def _get_preprocessor(self):
        if self.preprocessor is None:
            self.preprocessor = pipeline.load_preprocessor(self.model_path)
        return self.preprocessor

    def _get_model(self):
        if self.model is None:
            self.model = classifier.load_model(self.model_path)
        return self.model

    def _get_predictions(self, data):
        """ Returns the predicted class probilities
        """
        preprocessor = self._get_preprocessor()
        model = self._get_model()

        if preprocessor is None:
            raise Exception("No preprocessor found. Did you train first?")
        if model is None:
            raise Exception("No model found. Did you train first?")

        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data)
        # Grab input features for prediction
        pred_X = proc_data["X"].astype(np.float)
        # make predictions
        preds = model.predict_proba(pred_X)
        return preds


    def predict_proba(self, data):
        """ Returns predicted probabilities of each class """
        preds = self._get_predictions(data)
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)    
        print(preds.shape)   
        print(class_names)   
        preds_df = data[[self.id_field_name]].copy()
        preds_df[class_names] = preds
        return preds_df


    def predict(self, data):        
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)
        preds_df = data[[self.id_field_name]].copy()
        preds_df["prediction"] = pd.DataFrame(
            self.predict_proba(data), columns=class_names
        ).idxmax(axis=1)
        return preds_df


    def predict_to_json(self, data):
        preds_df = self.predict_proba(data)
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)
        preds_df["__label"] = pd.DataFrame(
            preds_df[class_names], columns=class_names
        ).idxmax(axis=1)

        predictions_response = []
        for rec in preds_df.to_dict(orient="records"):
            pred_obj = {}
            pred_obj[self.id_field_name] = rec[self.id_field_name]
            pred_obj["label"] = rec["__label"]
            pred_obj["probabilities"] = {
                str(k): np.round(v, 5)
                for k, v in rec.items()
                if k not in [self.id_field_name, "__label"]
            }
            predictions_response.append(pred_obj)
        
        print("predictions_response", predictions_response)
        return predictions_response
