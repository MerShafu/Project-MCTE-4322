from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from FuzzyLayer import DefuzzLayer, FuzzyLayer, NormLayer, RuleLayer, SummationLayer
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


class PredictorData(object):

    def __init__(self,cholestrol,bps,physical_activity,age,bmi,smoking,diabetes):
        self.cholestrol = cholestrol
        self.bps = bps
        self.physical_activity = physical_activity
        self.age = age
        self.bmi = bmi
        self.smoking = smoking
        self.diabetes = diabetes

        # Load the trained Keras model

class Predictor(object):
    def __init__(self):
        # Load the trained Keras model
        self.model = load_model('ANFIS_model.h5',
            custom_objects={'FuzzyLayer': FuzzyLayer,
                            'RuleLayer': RuleLayer,
                            'NormLayer': NormLayer,
                            'DefuzzLayer': DefuzzLayer,
                            'SummationLayer': SummationLayer})

        self.input_data: PredictorData = None

   # Function to perform prediction
    def predict(self):
        try:
            # Ensure that input_data is not None
            if self.input_data is None:
                raise ValueError("Input data is not loaded.")

            # Preprocess input
            smoking = 1 if self.input_data.smoking == True else 0
            diabetes = 1 if self.input_data.diabetes == True else 0

            data = [self.input_data.cholestrol, 
                    self.input_data.bps,
                    self.input_data.physical_activity, 
                    self.input_data.age,
                    self.input_data.bmi, 
                    self.input_data.smoking, 
                    self.input_data.diabetes]
            
            # scaler = MinMaxScaler()
            # data = scaler.fit_transform(np.array([data]))
            
            print("Input Data:", data)  # Add this line for debugging

            # Make prediction using the loaded model
            prediction = self.model.predict(np.array([data]))

            print("Raw Prediction:", prediction)  # Add this line for debugging

            # Check for NaN in the prediction
            if np.isnan(prediction).any():
                raise ValueError("Prediction contains NaN values.")

            result = prediction[0][0]
            #print("Result:", result)  # Add this line for debugging

            return result

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None
    
    def load_predictor_data(self, predictor_data: PredictorData) -> None:
        # Load the input data
        if isinstance(predictor_data, PredictorData):
            self.input_data = predictor_data
        else:
            print("Invalid input for PredictorData.")

    # Function to preprocess input data
    @staticmethod
    def preprocess(input_text):
        # Placeholder for actual preprocessing
        return input_text