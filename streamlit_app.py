# %%
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import statsmodels.stats.multicomp as mc
import json
import os

# --- 1. CONFIGURATION AND CACHING ---
st.set_page_config(layout="wide", page_title="Crime Analysis Dashboard")

@st.cache_resource
def load_and_process_data():
    """
    Loads all data, performs complex preprocessing, and fits models.
    Runs only once for performance.
    """
    try:
        # Load the primary data used in your provided code
        complete_data = pd.read_csv('small_complete_data.csv')

        # --- Data Cleaning and OLS Prep (From your script) ---
        temp_data = complete_data.dropna(subset=['Crime Count']).copy()
        temp_data[['prcp', 'wspd']] = temp_data[['prcp', 'wspd']].fillna(0)
        temp_data = temp_data.dropna(subset=['tavg'])
        temp_data['Crime Count'] = temp_data['Crime Count'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True).replace('', pd.NA).astype(float)
        cols_to_convert = ['tavg', 'prcp', 'wspd', 'Crime Count']
        temp_data[cols_to_convert] = temp_data[cols_to_convert].apply(pd.to_numeric, errors='coerce')
        temp_data = temp_data.dropna(subset=cols_to_convert)

        X_ols = temp_data[['tavg', 'prcp', 'wspd']]
        y_ols = temp_data['Crime Count']
        X_ols = sm.add_constant(X_ols)

        X_train_ols, _, y_train_ols, _ = train_test_split(X_ols, y_ols, test_size=0.2, random_state=42)
        model_ols = sm.OLS(y_train_ols.astype('float64'), X_train_ols.astype('float64')).fit()
        summary_text = model_ols.summary().as_text()

        # --- K-Means Post-Analysis (From your script) ---
        clean_data_kmeans = complete_data.dropna(subset=['tavg']).copy()
        comp = mc.MultiComparison(clean_data_kmeans['tavg'], clean_data_kmeans['cluster'])
        tukey_result = comp.tukeyhsd()
        tukey_summary = tukey_result.summary().as_text()

        

        return {
            "summary_text": summary_text,
            "complete_data": complete_data,
            "temp_data_ols": temp_data,
            "tukey_summary": tukey_summary
        }

    except FileNotFoundError as e:
        st.error(f"Missing Data File: {e}. Please ensure all required data files are in the root directory.")
        return None

# Load all cached data and models
data = load_and_process_data()

if data is None:
    st.stop() # Stop execution if data loading failed

# Extract required variables
summary_text = data["summary_text"]
complete_data = data["complete_data"]
temp_data_ols = data["temp_data_ols"]
tukey_summary = data["tukey_summary"]
# knn_figures = data["knn_figures"] # For when you add KNN


# --- 2. TAB FUNCTIONS (Plotting the Figures) ---

def linear_regression_tab():
    st.header("OLS Regression & Scatter Plots")

    st.subheader("OLS Regression Summary")
    st.code(summary_text, language='text')

    st.subheader("Crime Count vs Avg Temp Scatter Plot with OLS Trendline")
    fig_scatter_avg_temp = px.scatter(temp_data_ols, x="tavg", y="Crime Count", trendline="ols", title="Crime Count vs Avg Temperature")
    st.plotly_chart(fig_scatter_avg_temp, use_container_width=True)

    st.subheader("Crime Count vs Precipitation Scatter Plot with OLS Trendline")
    fig_scatter_prcp = px.scatter(temp_data_ols, x="prcp", y="Crime Count", trendline="ols", title="Crime Count vs Precipitation")
    st.plotly_chart(fig_scatter_prcp, use_container_width=True)

    st.subheader("Crime Count vs Wind Speed Scatter Plot with OLS Trendline")
    fig_scatter_wind = px.scatter(temp_data_ols, x="wspd", y="Crime Count", trendline="ols", title="Crime Count vs Wind Speed")
    st.plotly_chart(fig_scatter_wind, use_container_width=True)


def kmeans_analysis_tab():
    st.header("K-Means Clustering Analysis")

    st.subheader("Clustered Offense Categories")
    counts = complete_data.groupby(['Offense_cat', 'cluster']).size().reset_index(name='count')
    counts['percent'] = counts.groupby('Offense_cat')['count'].transform(lambda x: x / x.sum() * 100)
    fig_bar = px.bar(counts, x="Offense_cat", y="percent", color="cluster", title="Percent Stacked Bar Chart of Offense_cat by Cluster", barmode="stack")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Boxplot of Average Temperature (tavg) by Cluster")
    fig_box_tavg = px.box(complete_data, x="cluster", y="tavg", title="Boxplot of tavg by Cluster (ordered)", category_orders={"cluster": [0,1,2,3,4,5,6]})
    st.plotly_chart(fig_box_tavg, use_container_width=True)

    st.subheader("Tukey’s HSD Results (tavg vs. Cluster)")
    st.code(tukey_summary, language='text')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Boxplot of Hour Reported by Cluster")
        fig_box_hour = px.box(complete_data, x="cluster", y="HourReported", title="Boxplot of HourReported by Cluster (ordered)", category_orders={"cluster": [0,1,2,3,4,5,6]})
        st.plotly_chart(fig_box_hour)
    with col2:
        st.subheader("Boxplot of Precipitation (prcp) by Cluster")
        fig_box_prcp = px.box(complete_data, x="cluster", y="prcp", title="Boxplot of prcp by Cluster (ordered)", category_orders={"cluster": [0,1,2,3,4,5,6]})
        st.plotly_chart(fig_box_prcp)


# --- Placeholder Tabs for the Rest of the App ---




# --- 3. MAIN APPLICATION LAYOUT ---

st.title("A Geographic & Predictive Analysis of Crime In Charlottesville")

# Create tabs for navigation
tab2, tab3 = st.tabs([
    "Linear Regression & Plots", 
    "KMeans Analysis"
])



with tab2:
    linear_regression_tab()

with tab3:
    kmeans_analysis_tab()



