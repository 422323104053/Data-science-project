
# Student Grade Prediction with Linear Regression and Gradio Interface

from google.colab import files
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import gradio as gr

# Upload the Dataset
uploaded = files.upload()

# Load the Dataset
df = pd.read_csv('student-mat.csv', sep=';')

# Display Data Summary
print("Head:")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nInfo:")
df.info()
print("\nDescribe:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Visualizations
sns.histplot(df['G3'], kde=True)
plt.title('Distribution of Final Grade (G3)')
plt.xlabel('Final Grade')
plt.show()

sns.boxplot(x='studytime', y='G3', data=df)
plt.title('Study Time vs Final Grade')
plt.show()

# Prepare Data
target = 'G3'
features = df.columns.drop(target)
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded.drop('G3', axis=1))
y = df_encoded['G3']

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Prediction Function
def predict_grade(school, sex, age, address, famsize, Pstatus, Medu, Fedu,
                  Mjob, Fjob, reason, guardian, traveltime, studytime,
                  failures, schoolsup, famsup, paid, activities, nursery,
                  higher, internet, romantic, famrel, freetime, goout,
                  Dalc, Walc, health, absences, G1, G2):
    input_data = {
        'school': school, 'sex': sex, 'age': int(age), 'address': address, 'famsize': famsize,
        'Pstatus': Pstatus, 'Medu': int(Medu), 'Fedu': int(Fedu), 'Mjob': Mjob, 'Fjob': Fjob,
        'reason': reason, 'guardian': guardian, 'traveltime': int(traveltime), 'studytime': int(studytime),
        'failures': int(failures), 'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
        'activities': activities, 'nursery': nursery, 'higher': higher, 'internet': internet,
        'romantic': romantic, 'famrel': int(famrel), 'freetime': int(freetime), 'goout': int(goout),
        'Dalc': int(Dalc), 'Walc': int(Walc), 'health': int(health), 'absences': int(absences),
        'G1': int(G1), 'G2': int(G2)
    }
    input_df = pd.DataFrame([input_data])
    df_temp = pd.concat([df.drop('G3', axis=1), input_df], ignore_index=True)
    df_temp_encoded = pd.get_dummies(df_temp, drop_first=True)
    df_temp_encoded = df_temp_encoded.reindex(columns=df_encoded.drop('G3', axis=1).columns, fill_value=0)
    scaled_input = scaler.transform(df_temp_encoded.tail(1))
    prediction = model.predict(scaled_input)
    return round(prediction[0], 2)

# Gradio Interface
inputs = [
    gr.Dropdown(['GP', 'MS'], label="School"),
    gr.Dropdown(['M', 'F'], label="Gender"),
    gr.Number(label="Age"),
    gr.Dropdown(['U', 'R'], label="Residence Area"),
    gr.Dropdown(['LE3', 'GT3'], label="Family Size"),
    gr.Dropdown(['A', 'T'], label="Parent Cohabitation"),
    gr.Number(label="Mother's Education"),
    gr.Number(label="Father's Education"),
    gr.Dropdown(['teacher', 'health', 'services', 'at_home', 'other'], label="Mother's Job"),
    gr.Dropdown(['teacher', 'health', 'services', 'at_home', 'other'], label="Father's Job"),
    gr.Dropdown(['home', 'reputation', 'course', 'other'], label="School Reason"),
    gr.Dropdown(['mother', 'father', 'other'], label="Guardian"),
    gr.Number(label="Travel Time"),
    gr.Number(label="Study Time"),
    gr.Number(label="Failures"),
    gr.Dropdown(['yes', 'no'], label="School Support"),
    gr.Dropdown(['yes', 'no'], label="Family Support"),
    gr.Dropdown(['yes', 'no'], label="Paid Classes"),
    gr.Dropdown(['yes', 'no'], label="Activities"),
    gr.Dropdown(['yes', 'no'], label="Nursery"),
    gr.Dropdown(['yes', 'no'], label="Higher Education"),
    gr.Dropdown(['yes', 'no'], label="Internet"),
    gr.Dropdown(['yes', 'no'], label="Romantic"),
    gr.Number(label="Family Relationship"),
    gr.Number(label="Free Time"),
    gr.Number(label="Go Out"),
    gr.Number(label="Workday Alcohol"),
    gr.Number(label="Weekend Alcohol"),
    gr.Number(label="Health"),
    gr.Number(label="Absences"),
    gr.Number(label="G1"),
    gr.Number(label="G2")
]
output = gr.Number(label="Predicted Final Grade (G3)")
gr.Interface(fn=predict_grade, inputs=inputs, outputs=output, title="Student Grade Predictor").launch(share=True)
