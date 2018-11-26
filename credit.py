
"""Service to predict the probability of a person to default a loan.
How to use:
    firefly credit.predict
"""

# Prediction function for the server

import os
import numpy as np
from sklearn.externals import joblib

# read the encoders and the model
le_grade = joblib.load("encGrade.pkl")
le_ownership = joblib.load("encOwn.pkl")
model = joblib.load("model.pkl")

def predict(amount, grade, years, ownership, income, age):
    """
    Given these inputs, return the probability to default
    """
    
    # Encoders work on a vector. Wrap to make it work
    ownership_code = le_ownership.transform([ownership])[0]
    grade_code = le_grade.transform([grade])[0]
    
    # Transform age, income and amount
    age_feature = np.log(age)
    income_feature = np.log(income)
    amount_feature = np.log(amount)
    
    # Create feature array
    features = [amount_feature, grade_code, years, ownership_code, income_feature,
              age_feature]
    
    # Get the probabilities
    p0, p1 = model.predict_proba([features])[0]
    
    return p1
