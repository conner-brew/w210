import streamlit as st
import pandas as pd
import shap
import json
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from joblib import load

def explain_model(model, data, feats):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    indices=[]
    for feat in feats:
        for i in range(len(data.columns)):
            if data.columns[i] == feat:
                indices.append(i)
    shap_vals = {data.columns[i] : shap_values[0][i] for i in indices}
    pos = [i for i in shap_vals.keys() if shap_vals[i] > 0.001]
    neg = [i for i in shap_vals.keys() if shap_vals[i] < -0.001]
    
    return pos, neg, shap_vals

sb = st.sidebar # defining the sidebar

st.subheader('Our sandbox is currently trained on foster care data from Virginia and California. Please select your state in the sidebar to view the appropriate information.')
state_names = ["Virginia", "California"]
state = sb.radio("", state_names, index=0)

st.title('Abuse Risk Sandbox')


st.info('NOTE: This tool is not meant to be used in decision-making for specific cases. This sandbox can only show the impact that a feature may present, absent any context. Risk of abuse is a complex issue, and is strongly affected by the unique attributes of the caretaker family, environment, and the case itself. ')

if state == 'Virginia':
    st.subheader('California')
    st.markdown('In the state of Virginia, the following ten variables have been shown in modeling to be the most important features determining the likelihood of abuse in a foster case. Experiment with the features below, then press the "Predict Risk" button to see how it might affect risk!')

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

    DOB = st.slider("How old is the child currently?", 0, 18, 10, 1)
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

    for i in range(len(features)):
        test_data[feat_names[i]].values[:] = features[i]

    test_matrix = xgb.DMatrix(test_data)

    pos, neg, shap_vals = explain_model(model, test_data, feat_names)

    for l in pos, neg:
        for i in range(len(l)):
            if l[i] == 'Housing':
                    l[i] = 'Standard or Substandard Housing (Yes/No)'
            elif l[i] == 'Relinqsh':
                    l[i] = 'Parents Relinquished Parental Rights (Yes/No)'
            elif l[i] == 'Abandmnt':
                    l[i] = 'Evidence of Caretaker Abandonment (Yes/No)'
            elif l[i] == 'DAChild':
                    l[i] = 'Evidence or History of Child Drug Abuse (Yes/No)'
            elif l[i] == 'NoCope':
                    l[i] = 'Evidence or History of Caretaker Disability (Yes/No)'
            elif l[i] == 'RU13':
                    l[i] = 'Population of the Residence County'
            elif l[i] == 'FCMntPay':
                    l[i] = 'Value of Monthly Foster Care Payments to Caretakers'
            elif l[i] == 'CtkFamSt':
                    l[i] = 'Caretaker Family Structure'
            elif l[i] == 'DOB':
                    l[i] = 'Child Age'
            elif l[i] == 'InAtEnd':
                    l[i] = 'Child in Congregate Care or Caretaker Family'

    if st.button('Predict Risk'):
        with st.spinner("Running our model now...."):
            pred = model.predict(test_matrix)[0]
        
        if pred > high_bound:
            st.error('This case has a HIGHER THAN AVERAGE risk of abuse.')

        elif pred < low_bound:
            st.success('This case has a LOWER THAN AVERAGE risk of abuse.')

        else:
            st.warning('This case has an AVERAGE risk of abuse.')

        if len(pos) > 0:
            st.warning('Your submission for these features RAISE likelihood of abuse:')
            for i in pos:
                st.markdown("- " + i)

        if len(neg) > 0:
            st.info('Your submission for these features LOWER likelihood of abuse:')
            for i in neg:
                st.markdown("- " + i)

if state == 'California':
     st.subheader('California')
     st.markdown('In the state of California, the following ten variables have been shown in modeling to be the most important features determining the likelihood of abuse in a foster case. Experiment with the features below, then press the "Predict Risk" button to see how it might affect risk!')

     AgeAtLatRem = st.slider('How old was the child when they left their last placement? If this is their first placement, how old were they when they entered Foster Care?', 0, 18, 10)
     
     NoCope = st.radio("Does the caretaker presently have a physical, emotional, or mental disability, or a history of such disability?", ['Yes','No'])
     if NoCope == 'Yes':
         NoCope = 1
     else:
         NoCope = 0

     DAChild = st.radio("Does the child have a history of drug abuse, or is their evidence of current drug abuse?", ['Yes','No'])
     if DAChild == 'Yes':
         DAChild = 1
     else:
         DAChild = 0

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

     EverAdpt = st.radio("Has this child ever been previously adopted?", ['Yes','No'])
     if EverAdpt == 'Yes':
         EverAdpt = 1
     else:
         EverAdpt = 0

     DOB = st.slider("How old is the child currently?", 0, 18, 10, 1)
     DOB = (22 - DOB) * 100

     White = st.selectbox("What is the child's race or ethnicity?", ['White','Black/ African American','Hispanic or Latinx','Asian','American Indian or Native Alaskan', 'Native Hawaiian/Pacific Islander','Unknown'])
     if White == 'White':
         White = 1
     else:
         White = 0

     TotalRem = st.slider("How many times has the child been removed from a foster care placement?", 0, 15, 1)
    
     NumPlep = st.slider("How many different settings has the child lived in during their current foster care placement?", 1, 100, 1)

     FCMntPay = st.slider("Approximately how much money does the caretaker family receive in monthly foster care payments?", 0, 71000, 1000, 50)

     features = [AgeAtLatRem, NoCope, DAChild, CtkFamSt, EverAdpt, DOB, White, TotalRem, NumPlep, FCMntPay]

     model = load('streamlitva/CAmodel.joblib') 
     test_data = pd.read_csv('streamlitva/CA_model_data.csv')
     feat_names = ['AgeAtLatRem', 'NoCope', 'DAChild', 'CtkFamSt', 'EverAdpt', 'DOB', 'White', 'TotalRem', 'NumPlep', 'FCMntPay']
     with open('streamlitva/ca_predstats.json') as json_file:
         predstats = json.load(json_file)
        
     if len(test_data) != 71:
         test_data = test_data.drop(columns = ['Unnamed: 0'])

     avg = predstats['pred_mean']
     std = predstats['pred_std']
     low_bound = avg - std
     high_bound = avg + std

     for i in range(len(features)):
         test_data[feat_names[i]].values[:] = features[i]

     test_matrix = xgb.DMatrix(test_data)

     pos, neg, shap_vals = explain_model(model, test_data, feat_names)

     for l in pos, neg:
         for i in range(len(l)):
             if l[i] == 'AgeAtLatRem':
                     l[i] = 'Age of Child at Their Last Removal, or When They Entered Foster Care'
             elif l[i] == 'EverAdpt':
                     l[i] = 'Child Previously Adopted (Yes/No)'
             elif l[i] == 'White':
                     l[i] = 'Child Ethnicity (Yes/No)'
             elif l[i] == 'DAChild':
                     l[i] = 'Evidence or History of Child Drug Abuse (Yes/No)'
             elif l[i] == 'NoCope':
                     l[i] = 'Evidence or History of Caretaker Disability (Yes/No)'
             elif l[i] == 'NumPlep':
                     l[i] = 'Number of Different Settings During Current Placement'
             elif l[i] == 'FCMntPay':
                     l[i] = 'Value of Monthly Foster Care Payments to Caretakers'
             elif l[i] == 'CtkFamSt':
                     l[i] = 'Caretaker Family Structure'
             elif l[i] == 'DOB':
                     l[i] = 'Child Age'
             elif l[i] == 'TotalRem':
                     l[i] = 'Number of Total Removals From Placement'

     if st.button('Predict Risk'):
         with st.spinner("Running our model now...."):
             pred = model.predict(test_matrix)[0]
        
         if pred > high_bound:
             st.error('This case has a HIGHER THAN AVERAGE risk of abuse.')

         elif pred < low_bound:
             st.success('This case has a LOWER THAN AVERAGE risk of abuse.')

         else:
             st.warning('This case has an AVERAGE risk of abuse.')

         if len(pos) > 0:
             st.warning('Your submission for these features RAISE likelihood of abuse:')
             for i in pos:
                 st.markdown("- " + i)

         if len(neg) > 0:
             st.info('Your submission for these features LOWER likelihood of abuse:')
             for i in neg:
                 st.markdown("- " + i)