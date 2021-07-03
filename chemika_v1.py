#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#st.set_page_config(layout="wide")  # this needs to be the first Streamlit command called

st.title("Hello, Welcome to chemika.ai")

st.markdown("*An AI tool to simplify and automate feature engineering and machine learning model building*")

st.sidebar.title("Control Panel")
left_col, middle_col, right_col = st.beta_columns(3)

tick_size = 12
axis_title_size = 16


st.sidebar.header("Data Preprocessing")
st.sidebar.subheader("Outlier Detection and removal")

IQR = st.sidebar.slider(
    "Inter Quartile Range (IQR)",
    min_value=0.0,
    max_value=5.0,
    value=1.5,
    step=0.1,
    help="The higher this number, more outliers you will be removing, good for skewed data",
)



ZSCORE= st.sidebar.slider(
    "Z-SCORE",
    min_value=0.0,
    max_value=3.0,
    value=2.0,
    step=0.1,
    help="The higher this number, more outliers you will be removing, good for normal data",
)


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  dataset = pd.read_csv(uploaded_file)

dataset.drop('Unnamed: 0', axis=1, inplace=True)

dataset.shape

df = dataset.iloc[:,:7]

st.subheader('Dataset description')
st.table(dataset.head(10))

num_cols = set(df._get_numeric_data().columns)
cat_cols = set(df.columns) - num_cols

st.write('Numerical columns: ', num_cols)
st.write('Categorical columns: ', cat_cols)
st.write(df.describe().round(2))

for col in cat_cols:
    print(df[col].value_counts())


def plot_corr(df):
    dataset_corr = df.corr().round(4)
    mask = np.zeros_like(dataset_corr.round(4))
    mask[np.triu_indices_from(mask)] = True
    
    with sns.axes_style("whitegrid"):
        f, ax = plt.subplots(figsize=(10, 8))
        ax = sns.heatmap(dataset_corr.round(2), mask=mask, vmax=1, center = 0, vmin=-1, square=True, cmap='PuOr', linewidths=.5,
                         annot=True, annot_kws={"size": 25}, fmt='.1f')
        plt.title('Heatmap (Correlations) of Features in the Dataset', fontsize=20)
        plt.xlabel('Features', fontsize=20)
        plt.ylabel('Features', fontsize=20)
    plt.show()
    st.pyplot(f)

def box_plot(df, num_cols):
    dataset_boxplot = (df[num_cols] - df[num_cols].median())/df[num_cols].std()
    f2, ax = plt.subplots(figsize=(10, 8))
    ax = sns.boxplot(data=dataset_boxplot, orient="h", palette="Set2")
    plt.title('First Call Resolution by Days.', fontsize=12)
    plt.ylabel('Days', fontsize=12)
    plt.xlabel('FCR', fontsize=12)
    plt.show()
    st.pyplot(f2)


def remove_outliers(dataset, num_cols, iqr_th = 1.5):
    import copy
    df = copy.copy(dataset)
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3 - q1
    l_b = q1 - iqr_th*IQR
    u_b = q3 + iqr_th*IQR
    
    df = df[~((df<l_b) | (df>u_b)).any(axis=1)]
    
    plot_corr(df)
    box_plot(df, num_cols)
    outliers_percent = (dataset.shape[0] - df.shape[0])/dataset.shape[0]*100
    
    return df, outliers_percent


df, outliers_percent = remove_outliers(df, num_cols, iqr_th = IQR)

st.write('percentage of outliers: ', outliers_percent)

st.sidebar.subheader("Modeling")

test_size = st.sidebar.slider(
    "Test size or unseen data",
    min_value=0.1,
    max_value=0.5,
    value=0.3,
    step=0.1,
    help="Test data to test your model on unseen data",
)

st.write(df.shape[0])
df_train = df.sample(frac=1- test_size, random_state=123)
df_test = df.drop(df_train.index)

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

st.write('Data for Modeling: ' + str(df_train.shape))
st.write('Unseen Data For Predictions: ' + str(df_test.shape))

target = st.sidebar.selectbox('Select target from following dropdown:',(df_train.columns))

st.write('You selected:', target)


from pycaret.regression import *
exp_reg101 = setup(data = df, target = target, session_id=123, silent=True)

import time

progress_bar = st.progress(0)
progress_text = st.empty()
for i in range(51):
    time.sleep(.1)
    progress_bar.progress(i)
    progress_text.text(f"Training multiple models and comparing performances: {i}%")
 
best = compare_models(exclude = ['ransac'], fold=5)

for i in range(51,101):
    time.sleep(.1)
    progress_bar.progress(i)
    progress_text.text(f"Training multiple models and comparing performances: {i}%")



models = pull()



st.write(models)


