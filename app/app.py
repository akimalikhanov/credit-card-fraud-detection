import streamlit as st
import pandas as pd
import numpy as np
import joblib


def prep_data(data):
    data_prep_pipe=joblib.load('models/data_pipe.joblib')
    return data_prep_pipe.transform(data)

def pred_fraud(data, model):
    tr_class, tr_prob=model.predict(data), model.predict_proba(data)
    if data.shape[0]>1:
        df_pred=pd.DataFrame(tr_prob, columns=['Prob Non Fraud', 'Prob Fraud'])
        df_pred['Status']=df_pred['Prob Non Fraud'].map(lambda x: "Not Fraudulent" if x>=0.5 else 'Fraudulent')
        return tr_class, df_pred
    else:
        return tr_class, tr_prob.flatten()

st.title('Credit Card Fraud Detection')
st.subheader("Prediction of fraudulent transactions")

uploaded_file=st.file_uploader("Choose a file. It can have a single or multiple transactions.", type=['csv'])
if uploaded_file is not None:
    input_data=pd.read_csv(uploaded_file)
    st.subheader('Input Data')
    st.write(input_data)
    prepped_input_data=prep_data(input_data)
    pred_model=joblib.load('models/xgb_final_model.sav')
    st.subheader('Prediction')
    transaction_class, transaction_prob=pred_fraud(prepped_input_data, pred_model)

    if transaction_prob.ndim==1:
        st.write(f'Fraud Probability: {transaction_prob[1]}')
        if transaction_class[0]==0:
            st.success('Transaction is Not Fraudulent')
        else:
            st.error('Transaction is Fraudulent')
    else:
        total, fraud=len(transaction_class), len(transaction_prob[transaction_prob['Prob Fraud']>=0.5])
        st.write(f'Total Transactions: {total}')
        st.write(f'No. of Fraudulent: {fraud}')
        st.write(f'No. of Normal: {total-fraud}')
        st.write(transaction_prob)
