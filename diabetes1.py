import pandas as pd
import plotly.express as px
import streamlit as st

with open('style.css') as f:
    custom_css = f.read()
# Load the heart dataset
df_heart = pd.read_csv("heart_dataset.csv")

# ---- SIDEBAR ----
st.sidebar.header("Please Filter Here:")
age_range = st.sidebar.slider("Select Age Range:", min_value=int(df_heart["age"].min()), max_value=int(df_heart["age"].max()), value=(25, 60))
sex = st.sidebar.selectbox("Select Gender:", options=["Male", "Female", "All"], index=2)

# Apply filters to the dataframe
df_selection_heart = df_heart.query("@age_range[0] <= age <= @age_range[1]")
if sex != "All":
    df_selection_heart = df_selection_heart[df_selection_heart["sex"] == 1 if sex == "Male" else 0]

# Check if the dataframe is empty:
if df_selection_heart.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop()

# ---- MAINPAGE ----
st.title(":heartbeat: Heart Health Dashboard")
st.markdown("##")

# TOP KPI's
average_age = round(df_selection_heart["age"].mean(), 1)
average_cholesterol = round(df_selection_heart["chol"].mean(), 1)

left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Average Age:")
    st.subheader(f"{average_age} years")
with middle_column:
    st.subheader("Average Cholesterol:")
    st.subheader(f"{average_cholesterol} mg/dl")

st.markdown("""---""")

# HEART DISEASE DISTRIBUTION [PIE CHART]
heart_disease_distribution = df_selection_heart["target"].value_counts()
fig_heart_disease = px.pie(
    heart_disease_distribution,
    names=heart_disease_distribution.index,
    title="<b>Heart Disease Distribution</b>",
    hole=0.5,
    color_discrete_sequence=["#0083B8", "#F22E2E"],
    template="plotly_white",
)
fig_heart_disease.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
)

# HEART DISEASE BY GENDER [BAR CHART]
heart_disease_by_gender = df_selection_heart.groupby(by=["sex", "target"]).size().reset_index(name="Count")
fig_heart_gender = px.bar(
    heart_disease_by_gender,
    x="sex",
    y="Count",
    color="target",
    title="<b>Heart Disease by Gender</b>",
    labels={"sex": "Gender", "target": "Heart Disease"},
    color_discrete_sequence=["#F22E2E", "#0083B8"],
    template="plotly_white",
)
fig_heart_gender.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Female", "Male"]),
    yaxis=(dict(showgrid=False)),
)

# CHOLESTEROL LEVELS [BOX PLOT]
fig_cholesterol = px.box(
    df_selection_heart,
    x="target",
    y="chol",
    points="all",
    title="<b>Cholesterol Levels by Heart Disease</b>",
    labels={"target": "Heart Disease", "chol": "Cholesterol"},
    color_discrete_sequence=["#0083B8", "#F22E2E"],
    template="plotly_white",
)
fig_cholesterol.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["No Heart Disease", "Heart Disease"]),
)

left_column, middle_column, right_column = st.columns(3)
left_column.plotly_chart(fig_heart_disease, use_container_width=True)
middle_column.plotly_chart(fig_heart_gender, use_container_width=True)
right_column.plotly_chart(fig_cholesterol, use_container_width=True)

st.markdown("""
    <style>
        body {
            background-color: #f0f0f0;
        }
    </style>
""", unsafe_allow_html=True)
