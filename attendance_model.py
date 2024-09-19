import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('/mnt/data/student_performance.csv')

X = df[['AttendanceRate']]
y = df['FinalGrade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

test_score = model.score(X_test, y_test)

attendance = float(input("Enter your attendance rate: "))
prediction = model.predict(pd.DataFrame([[attendance]]))

print(f"Predicted grade: {prediction[0]}")
print(f"Model accuracy: {test_score:.3f}")
