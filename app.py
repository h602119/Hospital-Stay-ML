import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import gradio as gr

print('fetching training data')

training_data = pd.read_csv('./training_data.csv')
test_data = pd.read_csv('./test_data.csv')

#Swapping genders = F to 0.0 and Gender = M to 1.0
training_data['gender'] = training_data['gender'].replace({'F': 0.0, 'M': 1.0})

#Sorting the unique faculty ids
faculty_ids = sorted(training_data['facid'].unique())

#The current numb to
current_num = 0.0
for char in faculty_ids:
  training_data['facid'] = training_data['facid'].replace({char : current_num})
  current_num += 1.0

#The visit date 'vdate', is not important for the model
#Hence we will remove it.
training_data = training_data.drop('vdate', axis=1)

#Select all rows where a nan is present
rows_with_nan = training_data.isna().any(axis=1).sum()

#Remove all rows where a NAN value is found
training_data_cleaned = training_data.dropna()


#The data is already split into training and test data

#Split the training data into variables and result.
X = training_data_cleaned.drop('lengthofstay', axis=1)
y = training_data_cleaned['lengthofstay']
X_train, X_val, y_train, y_val = train_test_split(X, y)

#The test data has no 'length of stay' so I will have to predict this.
test_X = test_data.copy()


# Instantiate and fit the model
regressor = RandomForestRegressor(n_estimators=128, random_state=20)
regressor.fit(X_train, y_train)

#Using our trained model to predict visit length on the test data
predicted_test_y = regressor.predict(test_X)



def calculate_stay(rcount, gender, dialysisrenalendstage, asthma, irondef, pneum, substancedependance, psychologicaldisordermajor, depress, psychother, fibrosisandother, malnutrition, hemo, hematocrit, neutrophils, sodium, glucose, bloodureanitro, creatine, bmi, pulse, respiration, secondarydiagnosisnonicd9, facid_label):
  # Map labels A-E to numbers 0-5
  id = 12123123
  
  gender_to_number = {'F': 0.0, 'M': 1.0}
  gender = gender_to_number[gender]

  fac_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
  facid = fac_to_number[facid_label]
  
  estimated_length_of_stay = regressor.predict([[id, rcount, gender, dialysisrenalendstage, asthma, irondef, pneum, substancedependance, psychologicaldisordermajor, depress, psychother, fibrosisandother, malnutrition, hemo, hematocrit, neutrophils, sodium, glucose, bloodureanitro, creatine, bmi, pulse, respiration, secondarydiagnosisnonicd9, facid]])
  print(f"{id, rcount, gender, dialysisrenalendstage, asthma, irondef, pneum, substancedependance, psychologicaldisordermajor, depress, psychother, fibrosisandother, malnutrition, hemo, hematocrit, neutrophils, sodium, glucose, bloodureanitro, creatine, bmi, pulse, respiration, secondarydiagnosisnonicd9, facid}")

  los = round(estimated_length_of_stay[0]) 

  return f"Estimated length of stay: {los}"


hospital_app = gr.Interface(
  fn=calculate_stay,
  inputs=[
        gr.inputs.Number(label="rcount"),
        gr.inputs.Radio(['F', 'M'],default='F', label="Gender"),
        gr.inputs.Checkbox(label="dialysisrenalendstage"),
        gr.inputs.Checkbox(label="asthma"),
        gr.inputs.Checkbox(label="irondef"),
        gr.inputs.Checkbox(label="pneum"),
        gr.inputs.Checkbox(label="substancedependance"),
        gr.inputs.Checkbox(label="psychologicaldisordermajor"),
        gr.inputs.Checkbox(label="depress"),
        gr.inputs.Checkbox(label="psychother"),
        gr.inputs.Checkbox(label="fibrosisandother"),
        gr.inputs.Checkbox(label="malnutrition"),
        gr.inputs.Number(label="hemo"),
        gr.inputs.Number(label="hematocrit"),
        gr.inputs.Number(label="neutrophils"),
        gr.inputs.Number(label="sodium"),
        gr.inputs.Number(label="glucose"),
        gr.inputs.Number(label="bloodureanitro"),
        gr.inputs.Number(label="creatine"),
        gr.inputs.Number(label="bmi"),
        gr.inputs.Number(label="pulse"),
        gr.inputs.Number(label="respiration"),
        gr.inputs.Number(label="secondarydiagnosisnonicd9"),
        gr.inputs.Radio(['A', 'B', 'C', 'D', 'E'],default='A', label="facid")
    ],
    outputs=gr.outputs.Textbox()
)

hospital_app.launch(share=True)
