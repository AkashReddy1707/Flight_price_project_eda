import streamlit as st
import pandas as pd
import zipfile
from eda import flight_price_eda
import os

st.set_page_config(page_title="Flight Price EDA", layout="wide")
st.title("Exploratory Data Analysis for Flight Price Prediction")


uploaded_file = st.file_uploader("Upload your dataset (CSV or ZIP)", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df=pd.read_excel(uploaded_file)
    
    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    if st.button("Run EDA"):
        result=flight_price_eda(df,target_col="Price",result_dir="reports")
        st.success("EDA Completed! Check the 'reports' directory for results.")

        st.subheader("Outliers (IQR Method)")

        if not result["outliers"].empty:
            st.write(f" Number of outliers:{result['outliers'].shape[0]}")
            st.dataframe(result["outliers"].head())
        else:
            st.write("No outliers detected based on IQR method.")

        st.subheader("Summary Statistics")
        for item in result["summary"]:
            st.text(str(item))
        
        #Zip Reports

        zip_filename = "eda_reports.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for foldername, subfolders, filenames in os.walk("reports"):
               for filename in filenames:
                   file_path = os.path.join(foldername, filename)
                   zipf.write(file_path, os.path.relpath(file_path, "reports"))
        with open(zip_filename, "rb") as f:
            st.download_button(
                label="Download EDA Reports",
                data=f,
                file_name=zip_filename,
                mime="application/zip"
            )