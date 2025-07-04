# 17. Develop a model to predict student performance or final exam scores
# based on attendance, participation, assignment scores, and other academic indicators.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load dataset with student names
df = pd.read_csv("student_performance.csv")

#Define features and target
X = df[['Attendance', 'Participation', 'AssignmentScore']]
y = df['FinalExamScore']
students = df['Student']

#Split into training and test sets (with student names)
X_train, X_test, y_train, y_test, student_train, student_test = train_test_split(
    X, y, students, test_size=0.2, random_state=1
)

#Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Predict final exam scores
y_pred = model.predict(X_test)

#Evaluate model
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

#Plot predictions with student names
plt.figure(figsize=(14, 10))  # Larger figure for readability
plt.scatter(y_test, y_pred, color='blue', label='Predicted Points')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Prediction Line')
plt.xlabel("Actual Final Exam Scores")
plt.ylabel("Predicted Final Exam Scores")
plt.title("Actual vs Predicted Final Exam Scores")

# Annotate student names with alternate offset positions
for i, (actual, predicted, name) in enumerate(zip(y_test, y_pred, student_test)):
    offset = (-10, 5) if i % 2 == 0 else (5, -10)
    plt.annotate(
        name,
        (actual, predicted),
        textcoords="offset points",
        xytext=offset,
        fontsize=8,
        color='blue',
        ha='right' if offset[0] < 0 else 'left'
    )

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

