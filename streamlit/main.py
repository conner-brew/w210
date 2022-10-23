import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from joblib import load

#Loading up the Regression model we created
#model = GradientBoostingClassifier()
model = load('streamlit_testmodel.joblib') 

#Caching the model for faster loading
@st.cache


# Define the prediction function
def predict(AACHILD, AMIAKN, PLACEOUT, LATREMLOS, CTKFAMST, REPDATYR, CASEGOAL, WHITE, CHBEHPRB, DAPARENT):
    #Predicting the price of the carat
    prediction = model.predict_proba(pd.DataFrame([[AACHILD, AMIAKN, PLACEOUT, LATREMLOS, CTKFAMST, REPDATYR, CASEGOAL, WHITE, CHBEHPRB, DAPARENT]], columns=['AACHILD', 'AMIAKN', 'PLACEOUT', 'LATREMLOS', 'CTKFAMST', 'REPDATYR', 'CASEGOAL', 'WHITE', 'CHBEHPRB', 'DAPARENT']))
    return prediction

st.title('Abuse Risk Sandbox')
st.header('Lorum Ipsum:')

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

AACHILD = st.select_slider('Is there any evidence that the child misuses alcohol?', ['Yes','No'])
if AACHILD == 'Yes':
    AACHILD = 1
else:
    AACHILD = 0

PLACEOUT = st.select_slider('Is the child placed in the state of VA?', ['Yes','No'])
if PLACEOUT == 'Yes':
    PLACEOUT = 1
else:
    PLACEOUT = 0

LATREMLOS = st.number_input("How many days since the child's last removal (if no prior removals, how many days since the child entered the system)?", 0, 7560, 30)

CTKFAMST = st.selectbox("What is the family structure of the caretaker family?", ['Married Couple','Unmarried Couple','Single Male','Single Male','Unknown'])	
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

CHBEHPRB = st.select_slider("Has the child exhibited behavioral problems at home, in school, or elsewhere in the community? This includes running away.", ['Yes','No'])
if CHBEHPRB == 'Yes':
    CHBEHPRB = 1
else:
    CHBEHPRB = 0

DAPARENT = st.select_slider("Is there evidence or history of drug abuse associated with the caretakers?", ['Yes','No'])
if DAPARENT == 'Yes':
    DAPARENT = 1
else:
    DAPARENT = 0

if st.button('Predict Price'):
    with st.spinner("Running our model now...."):
        risk = predict(AACHILD, AMIAKN, PLACEOUT, LATREMLOS, CTKFAMST, REPDATYR, CASEGOAL, WHITE, CHBEHPRB, DAPARENT)
        risk += 0.3
    
    if risk > 0.66:
        st.warning('This case has a HIGH risk of abuse.')

    elif risk > 0.33:
        st.info('This case has a MODERATE risk of abuse.')

    else:
        st.success('This case has a LOW risk of abuse.')