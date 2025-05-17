import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

classification_data = {
    'study_hours': [2, 4, 1, 5, 3, 7, 6, 8, 3, 9],
    'sleep_hours': [5, 7, 4, 6, 5, 8, 7, 9, 6, 9],
    'attendance': [60, 80, 55, 90, 70, 95, 85, 100, 75, 100],
    'passed': [0, 1, 0, 1, 0, 1, 1, 1, 0, 1]  # 1 = passed, 0 = failed
}

df = pd.DataFrame(classification_data)

x = df[['study_hours', 'sleep_hours', 'attendance']]
y= df['passed']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

svm = SVC(kernel='rbf', C=1, gamma='scale')
svm.fit(x_train, y_train)

y_predict = svm.predict(x_test)

print("SVM Accuracy", accuracy_score(y_test, y_pred=y_predict))