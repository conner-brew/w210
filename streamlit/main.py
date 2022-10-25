import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from joblib import load


#Loading up the Regression model we created

model = load('streamlit/streamlit_testmodel.joblib') 
#Caching the model for faster loading
#@st.cache

st.title('Abuse Risk Sandbox')
st.header('In the state of Virginia, the following ten variables have been shown in modeling to be the most important features determining the likelihood of abuse in a foster case. Experiment with the features below, then press the "Predict Risk" button to see how it might affect risk!')
st.warning('NOTE: This tool is not meant to be used in decision-making for specific cases. This sandbox can only show the impact that a feature may present, absent any context. Risk of abuse is a complex issue, and is strongly affected by the unique attributes of the caretaker family, environment, and the case itself. ')

racethn = st.selectbox('Child race/ethnicity', ['White','Asian','Black','American Indian','Mixed-Race','Other','Unknown'])
if racethn == 'Asian':
    AMIAKN = 1
    WHITE = 0
elif racethn == 'White':
    WHITE = 1
    WHITE = 0
    AMIAKN = 0
elif racethn == 'Mixed-Race':
    WHITE = 1
    WHITE = 0
    AMIAKN = 0
else:
    WHITE = 0
    AMIAKN = 0

AACHILD = st.radio('Is there any evidence that the child misuses alcohol?', ['Yes','No'])
if AACHILD == 'Yes':
    AACHILD = 1
else:
    AACHILD = 0

PLACEOUT = st.radio('Is the child placed in the state of VA?', ['Yes','No'])
if PLACEOUT == 'Yes':
    PLACEOUT = 0
else:
    PLACEOUT = 1

LATREMLOS = st.slider("How many days since the child's last removal (if no prior removals, how many days since the child entered the system)?", 0, 7560, 1000)

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

REPDATYR = 2022

CASEGOAL = st.selectbox("What is the most recently identified case goal?",['Reunification','Live with other relatives','Adoption','Long Term Foster Care','Emancipation','Guardianship','No Goal Established'])
if CASEGOAL == 'Reunification':
    CASEGOAL = 1
elif CASEGOAL =='Live with other relatives':
    CASEGOAL = 2
elif CASEGOAL == 'Adoption':
    CASEGOAL = 3
elif CASEGOAL == 'Long Term Foster Care':
    CASEGOAL = 4
elif CASEGOAL == 'Emancipation':
    CASEGOAL = 5
elif CASEGOAL == 'Guardianship':
    CASEGOAL = 6
elif CASEGOAL == 'No Goal Established':
    CASEGOAL = 7
else:
    CASEGOAL = 0

CHBEHPRB = st.radio("Has the child exhibited behavioral problems at home, in school, or elsewhere in the community? This includes running away.", ['Yes','No'])
if CHBEHPRB == 'Yes':
    CHBEHPRB = 1
else:
    CHBEHPRB = 0

DAPARENT = st.radio("Is there evidence or history of drug abuse associated with the caretakers?", ['Yes','No'])
if DAPARENT == 'Yes':
    DAPARENT = 1
else:
    DAPARENT = 0

data = pd.DataFrame([[AACHILD, AMIAKN, PLACEOUT, LATREMLOS, CTKFAMST, REPDATYR, CASEGOAL, WHITE, CHBEHPRB, DAPARENT]],
 columns=['AACHILD', 'AMIAKN', 'PLACEOUT', 'LATREMLOS', 'CTKFAMST', 'REPDATYR', 'CASEGOAL', 'WHITE', 'CHBEHPRB', 'DAPARENT'])

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
