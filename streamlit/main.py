import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from joblib import load


#Loading up the Regression model we created

model = load('streamlit/streamlit_testmodel.joblib') 
#Caching the model for faster loading
#@st.cache

st.title('Abuse Risk Sandbox')
st.subheader('In the state of Virginia, the following ten variables have been shown in modeling to be the most important features determining the likelihood of abuse in a foster case. Experiment with the features below, then press the "Predict Risk" button to see how it might affect risk!')
st.info('NOTE: This tool is not meant to be used in decision-making for specific cases. This sandbox can only show the impact that a feature may present, absent any context. Risk of abuse is a complex issue, and is strongly affected by the unique attributes of the caretaker family, environment, and the case itself. ')

HOUSING = st.radio('Is the child living in housing that always meets standards of care?', ['Yes','No'])
if HOUSING == 'Yes':
    HOUSING = 0
else:
    HOUSING = 1

RELINQSH = st.radio("Have the child's birth parents formally relinquished parental rights?", ['Yes','No'])
if RELINQSH == 'Yes':
    RELINQSH = 1
else:
    RELINQSH = 0

ABANDMNT = st.radio("Is there evidence that the caretakers leave the child alone or with others frequently or for long durations of time?", ['Yes','No'])
if ABANDMNT == 'Yes':
    ABANDMNT = 1
else:
    ABANDMNT = 0

DOB = st.slider("How old is the child?", 0, 18, 10, 1)
DOB = (22 - DOB) * 100

CTKFAMST = st.selectbox("What is the family structure of the caretaker family?", ['Married Couple','Unmarried Couple','Single Male','Single Female','Unknown'])	
if CTKFAMST == 'Married Couple':
    CTKFAMST = 1
elif CTKFAMST == 'Unmarried Couple':
    CTKFAMST = 2
elif CTKFAMST == 'Single Male':
    CTKFAMST = 3
elif CTKFAMST == 'Single Male':
    CTKFAMST = 4
elif CTKFAMST == 'Unknown':
    CTKFAMST = 5
else:
    CTKFAMST = 0

if st.button('Predict Risk'):
    with st.spinner("Running our model now...."):
        risk = model.predict_proba(data)[0][1]
        risk += 0.3
    
    if risk > 0.66:
        st.error('This case has a HIGH risk of abuse.')

    elif risk > 0.33:
        st.warning('This case has a MODERATE risk of abuse.')

    else:
        st.success('This case has a LOW risk of abuse.')
