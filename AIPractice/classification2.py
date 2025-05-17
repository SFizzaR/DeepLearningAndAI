import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = {
    'age': [22, 25, 47, 52, 46, 56, 55, 60, 48, 33],
    'salary': [15000, 29000, 48000, 60000, 52000, 65000, 70000, 80000, 50000, 35000],
    'gender': ['female', 'male', 'male', 'male', 'female', 'female', 'female', 'male', 'female', 'male'],
    'purchased': [0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  # 1 = bought, 0 = did not buy
}

df = pd.DataFrame(data)

le =LabelEncoder()

df['gender'] = le.fit_transform(df['gender'])

x = df[['age', 'salary', 'gender']]
y = df['purchased']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

svm = SVC(kernel='rbf', C=1, gamma='scale')
svm.fit(x_train, y_train)

y_predict = svm.predict(x_test)

print("Accuracy: ", accuracy_score(y_test, y_predict))