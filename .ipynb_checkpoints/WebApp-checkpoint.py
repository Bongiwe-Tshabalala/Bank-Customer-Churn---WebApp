import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC

file = open("Churn_Model.pkl", 'rb')

pickle_file = pickle.load(file)

svm_model = pickle_file['classifier']
preprocessor = pickle_file['preprocesor']

st.title('Churn Customers - Prediction Web App')

st.write('Please complete the info below and click on Predict to get results')

CreditScore = st.number_input('Credit Score', format='%d', min_value=0)

Geography = st.selectbox('Country',['Germany','France','Spain'])

Gender = st.selectbox('Gender', ['Male','Female'])

Age = st.number_input('Age', format='%d', min_value=18)

Tenure = st.number_input('Tenure', format='%d', min_value=0)
                      
Balance = st.number_input('Balance')

NumOfProducts = st.number_input('Number of Products')

HasCrCard = st.checkbox('Has a Credit Card')

IsActiveMember = st.checkbox('Is an active member')

EstimatedSalary = st.number_input('Estimated Salary')

data = {'CreditScore':CreditScore, 'Geography':Geography, 'Gender':Gender, 'Age':Age, 'Tenure':Tenure, 'Balance':Balance, 'NumOfProducts':NumOfProducts, 'HasCrCard':HasCrCard, 'IsActiveMember':IsActiveMember, 'EstimatedSalary':EstimatedSalary}

df = pd.DataFrame(data, index=[0])

st.dataframe(df)

x_transformed = preprocessor.transform(df)

y_pred = svm_model.predict(x_transformed)

predict = st.button('Predict')

if predict:
    if CreditScore == 0.00 or Age == 00 or Tenure == 0.00 or Balance == 0.00:
        st.info("Some values were computed as 0.")
        if y_pred == 0:
            st.info('This customer will less likely leave in the near future')
        else:
            st.info('This customer will likely leave in the near future')
    else:  
        if y_pred == 0:
            st.info('This customer will less likely leave in the near future')
        else:
            st.info('This customer will likely leave in the near future')
