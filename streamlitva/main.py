import streamlit as st
import pandas as pd
import shap
import json
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from joblib import load

def explain_model(model, data, feats):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data, approximate=True)

    indices=[]
    for feat in feats:
        for i in range(len(data.columns)):
            if data.columns[i] == feat:
                indices.append(i)
    shap_vals = {data.columns[i] : shap_values[0][i] for i in indices}
    pos = [i for i in shap_vals.keys() if shap_vals[i] > 0]
    neg = [i for i in shap_vals.keys() if shap_vals[i] < 0]
    
    return pos, neg

st.title('Abuse Risk Sandbox')
st.subheader('In the state of Virginia, the following ten variables have been shown in modeling to be the most important features determining the likelihood of abuse in a foster case. Experiment with the features below, then press the "Predict Risk" button to see how it might affect risk!')
st.info('NOTE: This tool is not meant to be used in decision-making for specific cases. This sandbox can only show the impact that a feature may present, absent any context. Risk of abuse is a complex issue, and is strongly affected by the unique attributes of the caretaker family, environment, and the case itself. ')

Housing = st.radio('Is the child living in housing that always meets standards of care?', ['Yes','No'])
if Housing == 'Yes':
    Housing = 0
else:
    Housing = 1

Relinqsh = st.radio("Have the child's birth parents formally relinquished parental rights?", ['Yes','No'])
if Relinqsh == 'Yes':
    Relinqsh = 1
else:
    Relinqsh = 0

Abandmnt = st.radio("Is there evidence that the caretakers leave the child alone or with others frequently or for long durations of time?", ['Yes','No'])
if Abandmnt == 'Yes':
    Abandmnt = 1
else:
    Abandmnt = 0

DAChild = st.radio("Does the child have a history of drug abuse, or is their evidence of current drug abuse?", ['Yes','No'])
if DAChild == 'Yes':
    DAChild = 1
else:
    DAChild = 0

NoCope = st.radio("Does the caretaker presently have a physical, emotional, or mental disability, or a history of such disability?", ['Yes','No'])
if NoCope == 'Yes':
    NoCope = 1
else:
    NoCope = 0

RU13 = st.selectbox("What is the approximate population of the child's placement county?", ['Greater than 1 Million','Between 250K and 1 Million', 'Between 20K and 250K', 'Between 2.5K and 20K', 'Less Than 2.5K'])
if RU13 == 'Greater than 1 Million':
    RU13 = 1
elif RU13 == 'Between 250K and 1 Million':
    RU13 = 2
elif RU13 == 'Between 20K and 250K':
    RU13 = 3
elif RU13 == 'Between 2.5K and 20K':
    RU13 = 7
else:
    RU13 = 8

FCMntPay = st.slider("Approximately how much money does the caretaker family receive in monthly foster care payments?", 0, 71000, 1000, 50)

CtkFamSt = st.selectbox("What is the family structure of the caretaker family?", ['Married Couple','Unmarried Couple','Single Male','Single Female','Unknown'])	
if CtkFamSt == 'Married Couple':
    CtkFamSt = 1
elif CtkFamSt == 'Unmarried Couple':
    CtkFamSt = 2
elif CtkFamSt == 'Single Male':
    CtkFamSt = 3
elif CtkFamSt == 'Single Male':
    CtkFamSt = 4
elif CtkFamSt == 'Unknown':
    CtkFamSt = 5
else:
    CtkFamSt = 0

DOB = st.slider("How old is the child?", 0, 18, 10, 1)
DOB = (22 - DOB) * 100

InAtEnd = st.radio("Is the child currently placed with a caretaker family or is the child in congregate care?", ['Caretaker Family','Congregate Care'])
if InAtEnd == 'Caretaker Family':
    InAtEnd = 1
else:
    InAtEnd = 0

features = [Housing, Relinqsh, Abandmnt, DAChild, NoCope, RU13, FCMntPay, CtkFamSt, DOB, InAtEnd]

model = load('streamlitva/VAmodel.joblib') 
test_data = pd.read_csv('streamlitva/VA_model_data.csv')
feat_names = ['Housing', 'Relinqsh', 'Abandmnt', 'DAChild', 'NoCope', 'RU13', 'FCMntPay', 'CtkFamSt', 'DOB', 'InAtEnd']
with open('streamlitva/va_predstats.json') as json_file:
    predstats = json.load(json_file)
    
if len(test_data) != 71:
    test_data = test_data.drop(columns = ['Unnamed: 0'])

avg = predstats['pred_mean']
std = predstats['pred_std']
low_bound = avg - std
high_bound = avg + std
    
neg, pos = explain_model(model, test_data, feat_names)

if st.button('Predict Risk'):
    with st.spinner("Running our model now...."):
        for i in range(len(features)):
            test_data[feat_names[i]].values[:] = features[i]
        test_matrix = xgb.DMatrix(test_data)
        pred = model.predict(test_matrix)[0]
    
    if pred > high_bound:
        st.error('This case has a HIGHER THAN AVERAGE risk of abuse.')

    elif pred < low_bound:
        st.success('This case has a LOWER THAN AVERAGE risk of abuse.')

    else:
        st.warning('This case has an AVERAGE risk of abuse.')

    if len(pos) > 0:
        st.warning('These features raise the likelihood of abuse')
        st.write(pos)

    if len(neg) > 0:
        st.success('These features lower the likelihood of abuse')
        st.write(neg)