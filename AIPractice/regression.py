import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

regression_data = {
    'area_sqft': [1000, 1500, 800, 1200, 2000, 1800, 1600, 2200, 1300, 1700],
    'bedrooms': [2, 3, 2, 3, 4, 4, 3, 5, 3, 4],
    'bathrooms': [1, 2, 1, 2, 3, 2, 2, 4, 2, 3],
    'age_years': [10, 5, 15, 7, 3, 4, 6, 2, 8, 5],
    'price': [200000, 300000, 180000, 250000, 400000, 380000, 320000, 500000, 260000, 350000]
}

df = pd.DataFrame(regression_data)

# Define X and y
X = df[['area_sqft', 'bedrooms', 'bathrooms', 'age_years']]
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.3)
LR = LinearRegression()
ModelLR = LR.fit(x_train, y_train)
predictionLR = ModelLR.predict(x_test)

print("predictions", predictionLR)

teachLR = r2_score(y_test, predictionLR)
teachacclr = teachLR * 100