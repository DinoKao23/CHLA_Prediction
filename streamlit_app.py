# python -m streamlit run hello.py

import sklearn
import streamlit as st
import pandas as pd
import datetime
import pickle

def load_data(file_path):
    return pd.read_csv(file_path)
model = pickle.load(open('random_forest.pkl', 'rb'))
today = datetime.datetime.today()
date_string = datetime.datetime(2000, 1, 1)


def main():
    st.set_page_config(page_title="CHLA Prediction", layout= 'wide', page_icon= '🏥')
    st.title("Patient Showup Prediction: ")

    hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
    st.markdown(hide_default_format, unsafe_allow_html=True)

    df = load_data("CHLA_clean_data_until_2023.csv")
    df['APPT_DATE'] = pd.to_datetime(df['APPT_DATE'])

    
    default_start_date = df['APPT_DATE'].min()
    default_end_date = df['APPT_DATE'].max()

    # Create columns
    c1, c2 = st.columns([1, 1])

    # Within c1
    with c1:
    # Date input for start date
        start_datetime = st.date_input("Choose Start Date", min_value=default_start_date, max_value=default_end_date, value=default_start_date)

# Within c2
    with c2:
        # Date input for end date
        end_datetime = st.date_input("Choose End Date", min_value=default_start_date, max_value=default_end_date, value=default_end_date)

    start_datetime = pd.to_datetime(start_datetime)
    end_datetime = pd.to_datetime(end_datetime)

    clinic = st.selectbox("Select an Clinic", df['CLINIC'].unique())

    if st.button("Predict"):    
        filtered_df = df[(df['APPT_DATE'] >= start_datetime) & (df['APPT_DATE'] <= end_datetime)]
        clinic_df = filtered_df[filtered_df['CLINIC'] == clinic]

        predict_df = clinic_df[['APPT_DATE', 'BOOK_DATE', 'LEAD_TIME', 'TOTAL_NUMBER_OF_NOSHOW', 'TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT', 'TOTAL_NUMBER_OF_CANCELLATIONS', 'TOTAL_NUMBER_OF_RESCHEDULED']]

        predict_df['APPT_DATE'] = pd.to_datetime(predict_df['APPT_DATE'])
        predict_df['BOOK_DATE'] = pd.to_datetime(predict_df['BOOK_DATE'])

        predict_df['NUM_OF_MONTH'] = predict_df['APPT_DATE'].dt.month

        # Now, all your features should be numerical, and you can attempt prediction
        features_list = predict_df.drop(['APPT_DATE','BOOK_DATE'],axis = 1).values
        try:
            prediction = model.predict(features_list)
            predict_df['prediction'] = prediction
            prediction_text_map = { 1: "Patient will not show up",0: "Patient will show up"}
            predict_df['CLINIC'] = clinic
            predict_df['prediction_text'] = predict_df['prediction'].map(prediction_text_map)
            st.write(predict_df)
        except ValueError as e:
            st.error("There are no clients in this range, please select other ranges.")
        
if __name__=='__main__':
    main()
