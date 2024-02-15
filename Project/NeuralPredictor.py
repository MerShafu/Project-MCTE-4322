import tensorflow as tf
from keras.models import load_model

class PredictorData:
    def __init__(self, cholestrol, bps, physical_activity, age, bmi, smoking, diabetes):
        self.cholestrol = cholestrol
        self.bps = bps
        self.physical_activity = physical_activity
        self.age = age
        self.bmi = bmi
        self.smoking = smoking
        self.diabetes = diabetes

class HeartAttackPredictor:
    def __init__(self, model_path='NeuralNetwork_model.h5'):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        self.model = load_model(self.model_path)

    def load_predictor_data(self, predictor_data):
        self.predictor_data = predictor_data

    def preprocess_input(self):
        # Perform any necessary preprocessing on input data
        # Example: Convert categorical variables to numerical, scaling, etc.
        pass

    def predict(self):
        if self.model is None:
            self.load_model()

        if not hasattr(self, 'predictor_data'):
            raise ValueError("Please provide predictor data using load_predictor_data method.")

        self.preprocess_input()

        smoking = 1 if self.predictor_data.smoking == True else 0
        diabetes = 1 if self.predictor_data.diabetes == True else 0

        input_data = [[
            self.predictor_data.cholestrol,
            self.predictor_data.bps,
            self.predictor_data.physical_activity,
            self.predictor_data.age,
            self.predictor_data.bmi,
            self.predictor_data.smoking,
            self.predictor_data.diabetes
        ]]

        # prediction = self.model.predict(input_data)
        # return prediction[0][0]

        prediction = self.model.predict(input_data)[0][0]
        print(prediction)

        if prediction < 0.50:
            result = "NOT LIKELY"
        elif 0.5 <= prediction < 0.75:
            result = "POSSIBLY"
        else:
            result = "MOST LIKELY"

        return result, prediction
