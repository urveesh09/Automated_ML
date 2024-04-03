import streamlit as st
import pandas as pd
import os

from pycaret.classification import *
from pycaret.regression import setup as rsetup
from pycaret.regression import pull as rpull
from pycaret.regression import save_model as rsave_model
from pycaret.regression import compare_models as rcompare_models


# from pandas_profiling import ProfileReport
# from streamlit_pandas_profiling import st_pro
def run():
# Sidebar navigation
    with st.sidebar:
        st.title("AutoStreamML")
        choice = st.radio("Navigation", ["Upload", "ML (Regression)", "ML (Classification)"])
        st.info("This app allows you to build automated ML pipelines using Streamlit and PyCaret.")

    # Load dataset if exists
    if os.path.exists("source.csv"):
        df = pd.read_csv("source.csv", index_col=None)

    # Upload dataset
    if choice == "Upload":
        st.title("Upload Dataset")
        file = st.file_uploader("Upload Your Dataset (CSV format only)", type="csv")
        if file:
            df = pd.read_csv(file, index_col=None)
            st.subheader("Uploaded Dataset")
            df.to_csv('source.csv', index=None)
            st.dataframe(df)

    # Machine Learning tasks
    elif choice.startswith("ML"):
        task_type = choice.split("(")[1].split(")")[0]

        if task_type == "Regression":
            st.title("Auto ML Model Maker (Regression)")
            target = st.selectbox("Select the Target Column", df.columns)
            if st.button("Train Model"):
                rsetup(df, target=target)
                setup_df = rpull()
                st.info("Please do be patient, the larger the dataset, longer the wait ")
                st.info("ML Experiment Settings")
                st.dataframe(setup_df)
                st.info("ML Models Comparison")
                best_model = rcompare_models()
                compare_df = rpull()
                st.dataframe(compare_df)
                rsave_model(best_model, 'best_model')
                with open("best_model.pkl", 'rb') as f:
                    st.download_button("Download the best Model", f, "trained_model.pkl")

        elif task_type == "Classification":
            st.title("Auto ML Model Maker (Classification)")
            target = st.selectbox("Select the Target Column", df.columns)
            if st.button("Train Model"):
                setup(df, target=target)
                setup_df = pull()
                st.info("ML Experiment Settings")
                st.info("Please do be patient, the larger the dataset, longer the wait ")
                st.dataframe(setup_df)
                st.info("ML Models Comparison")
                best_model = compare_models()
                compare_df = pull()
                st.dataframe(compare_df)
                save_model(best_model, 'best_model')
                with open("best_model.pkl", 'rb') as f:
                    st.download_button("Download the best Model", f, "trained_model.pkl")



if __name__ == '__main__':
    run()