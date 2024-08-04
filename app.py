# import libraries
import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

# import ML libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# App title
st.title('Diabetes Detection System')

# header
st.header('This is a Diabetes Detection System')
st.write("Diabetes is a chronic disease that occurs when your blood glucose is too high. This application helps to effectively detect if someone has diabetes using Machine Learning. " )

# load the dataset
df = pd.read_csv('diabetes.csv')

# Display Sample Data to the User
st.header('Sample Data')
st.dataframe(df.head())

#Data Preprocessing

# List of columns to update
columns_to_update = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0 with NaN in the specified columns
df[columns_to_update] = df[columns_to_update].replace(0, np.nan)

# Replace the missing values 
def median_target(var):   
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp

# Glucose
df.loc[(df['Outcome'] == 0 ) & (df['Glucose'].isnull()), 'Glucose'] = 110.6
df.loc[(df['Outcome'] == 1 ) & (df['Glucose'].isnull()), 'Glucose'] = 142.3

# BloodPressure
df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 70.9
df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 75.3

# SkinThickness
df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 27.2
df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 33.0

# Insulin
df.loc[(df['Outcome'] == 0 ) & (df['Insulin'].isnull()), 'Insulin'] = 130.2
df.loc[(df['Outcome'] == 1 ) & (df['Insulin'].isnull()), 'Insulin'] = 206.8

# BMI
df.loc[(df['Outcome'] == 0 ) & (df['BMI'].isnull()), 'BMI'] = 30.1
df.loc[(df['Outcome'] == 1 ) & (df['BMI'].isnull()), 'BMI'] = 34.3

# Create our data features and target
X = df.drop(columns='Outcome')
y = df['Outcome']

# Select columns to fit the model
columns = ['Insulin', 'Glucose', 'BMI', 'Age', 'SkinThickness']
X = X[columns]

# Scaling the data
scaler = StandardScaler()
X =pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Sample features
st.subheader('Sample Features')
st.table(X.head())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# get user input 
name = st.text_input("What is your name?").capitalize()

# Create a function to get user input/get feature input from the user
if name != "":
    st.write('Hello {} Please complete the form below'.format(name))
else: 
    st.write('Please enter your name')

# Get user input
def get_user_input():
    Insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    Glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    Age = st.sidebar.slider('Age', 21, 81, 29)
    SkinThickness = st.sidebar.slider('SkinThickness', 0, 99, 23)


    # Store a dictionary into a variable
    user_data = {'Insulin': Insulin,
                 'Glucose': Glucose,
                 'BMI': BMI,
                 'Age': Age,
                 'SkinThickness': SkinThickness}
    
    # Transform the data into a dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

# return the user input
user_input = get_user_input()

# Display the user input
st.subheader('Below is the user input {}'.format(name))
st.dataframe(user_input)

# scale the user input
user_input_scaled = scaler.transform(user_input)
user_input_scaled = pd.DataFrame(user_input_scaled, columns=user_input.columns)
st.dataframe(user_input_scaled)

# create button to ask the user to get results
bt = st.button('Get Results')

if bt:
    # Create the model
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # Get the user input features prediction
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)

    # Display the result
    if prediction == 1:
        st.write('Hello {} you have diabetes'.format(name))
    else:
        st.write('Hello {} you do not have diabetes'.format(name))
    # Display the model accuracy
    st.write('Model Accuracy: ', round(metrics.accuracy_score(y_test, model.predict(X_test)),2)*100)
