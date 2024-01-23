import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import numpy as np
import plotly.express as px
import pandas as pd
import json
import requests
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import h5py
import keras
from streamlit_lottie import st_lottie
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

import base64
from io import BytesIO

# loading the saved models

diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))
#skin_cancer_model = load_model('skin_cancer_model_with_history.h5')
brain_disease_model = load_model('brain_tumor_model.h5')
model_path = "brain_tumor_prediction_mri_model.h5"

st.set_page_config(
    page_title="Multiple Disease Prediction System",
    page_icon=":heart:",
    layout="wide",
)

# option menu for diseases
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Home Page','KPI overview - Diabetes','KPI overview - Heart',  'Diabetes Prediction', 'Heart Disease Prediction',
                            'Brain Tumor Classification', 'Skin Cancer Classification'],
                           icons=['circle', 'circle', 'circle', 'circle','circle','circle','circle'],
                           default_index=0)

if selected == 'Home Page':

    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)


    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()


    # Load Lottie files
    lottie_coding = load_lottiefile("Animation3.json")
    lottie_hello = load_lottieurl("https://lottie.host/a2906b77-5497-4524-9fa6-56ed19c8eb53/ZSL8HrAYcw.json")
    # lottie_coding2 = load_lottiefile("Animation3.json")
    # Set page configuration

    # Lottie animation for coding
    st_lottie(
        lottie_coding,
        speed=0.5,
        reverse=False,
        loop=True,
        quality="low",
        height=100,
    )

    # Display Lottie animations and description on the home page
    st.title("Welcome to the Disease Prediction App!")

    # App description
    st.markdown(
        """

        This app uses AI to predict the likelihood of certain health problems.

        **Disclaimer:** The predictions provided by this app are based on artificial intelligence and should not be taken as professional medical advice. 
        Consult with your healthcare provider for accurate health assessments.

        Feel free to explore the disease prediction sections using the sidebar!
        """
    )

    # Lottie animation for saying hello
    st_lottie(
        lottie_hello,
        speed=1.5,
        reverse=False,
        loop=True,
        quality="low",
        height=400,
    )

elif selected == 'KPI overview - Diabetes':
    df_diabetes = pd.read_csv("C:\\Users\\HP\\PycharmProjects\\pythonProject6\\csv_file_datasets\\diabetes_dataset.csv")



    st.sidebar.header("Filter Data:")
    # sidebar for diabetes
    pregnancies_range = st.sidebar.slider("Select Pregnancies Range:", min_value=int(df_diabetes["Pregnancies"].min()),
                                          max_value=int(df_diabetes["Pregnancies"].max()), value=(0, 10))
    glucose_range = st.sidebar.slider("Select Glucose Range:", min_value=int(df_diabetes["Glucose"].min()),
                                      max_value=int(df_diabetes["Glucose"].max()), value=(80, 200))
    outcome = st.sidebar.selectbox("Select Outcome:", options=["0 (No Diabetes)", "1 (Diabetes)", "All"], index=2)

    # Apply filters to the dataframe
    df_selection_diabetes = df_diabetes.query("@pregnancies_range[0] <= Pregnancies <= @pregnancies_range[1]")
    df_selection_diabetes = df_selection_diabetes.query("@glucose_range[0] <= Glucose <= @glucose_range[1]")
    if outcome != "All":
        df_selection_diabetes = df_selection_diabetes[df_selection_diabetes["Outcome"] == int(outcome[0])]

    # Second sidebar for additional filters
    st.sidebar.header("Additional Filters:")
    # Add your additional filters here...

    # Check if the dataframe is empty:
    if df_selection_diabetes.empty:
        st.warning("No data available based on the current filter settings!")
        st.stop()

    # ---- MAINPAGE ----
    st.title(":pill: Diabetes Dashboard")
    st.markdown("##")

    # KPIs section
    st.subheader("Key Performance Indicators")

    # TOP KPI's
    average_glucose = round(df_selection_diabetes["Glucose"].mean(), 1)
    average_blood_pressure = round(df_selection_diabetes["BloodPressure"].mean(), 1)

    # Display KPIs
    st.write("**Average Glucose:**", f"{average_glucose} mg/dl")
    st.write("**Average Blood Pressure:**", f"{average_blood_pressure} mm Hg")

    # ---- CHARTS ----
    # GLUCOSE LEVELS [BOX PLOT]
    fig_glucose = px.box(
        df_selection_diabetes,
        x="Outcome",
        y="Glucose",
        points="all",
        title="<b>Glucose Levels by Diabetes Outcome</b>",
        labels={"Outcome": "Diabetes Outcome", "Glucose": "Glucose"},
        color_discrete_sequence=["#0083B8", "#F22E2E"],
        template="plotly_white",
    )
    fig_glucose.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["No Diabetes", "Diabetes"]),
    )
    st.plotly_chart(fig_glucose, use_container_width=True)

    # BLOOD PRESSURE LEVELS [BOX PLOT]
    fig_blood_pressure = px.box(
        df_selection_diabetes,
        x="Outcome",
        y="BloodPressure",
        points="all",
        title="<b>Blood Pressure Levels by Diabetes Outcome</b>",
        labels={"Outcome": "Diabetes Outcome", "BloodPressure": "Blood Pressure"},
        color_discrete_sequence=["#0083B8", "#F22E2E"],
        template="plotly_white",
    )
    fig_blood_pressure.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["No Diabetes", "Diabetes"]),
    )
    st.plotly_chart(fig_blood_pressure, use_container_width=True)

    # Comparison of Diabetes Outcome [PIE CHART]
    diabetes_outcome_distribution = df_selection_diabetes["Outcome"].value_counts()
    fig_diabetes_outcome = px.pie(
        diabetes_outcome_distribution,
        names=diabetes_outcome_distribution.index,
        title="<b>Diabetes Outcome Distribution</b>",
        hole=0.5,
        color_discrete_sequence=["#0083B8", "#F22E2E"],
        template="plotly_white",
    )
    fig_diabetes_outcome.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_diabetes_outcome, use_container_width=True)

    st.markdown("""---""")
    # ... Add more visualizations for other columns here ...

    # ---- HIDE STREAMLIT STYLE ----
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

elif selected == 'KPI overview - Heart':
    df_heart = pd.read_csv("C:\\Users\\HP\\PycharmProjects\\pythonProject6\\csv_file_datasets\\heart_dataset.csv")

    # ---- SIDEBAR ----
    st.sidebar.header("Please Filter Here:")
    age_range = st.sidebar.slider("Select Age Range:", min_value=int(df_heart["age"].min()),
                                  max_value=int(df_heart["age"].max()), value=(25, 60))
    sex = st.sidebar.selectbox("Select Gender:", options=["Male", "Female", "All"], index=2)

    # Apply filters to the dataframe
    df_selection_heart = df_heart.query("@age_range[0] <= age <= @age_range[1]")
    if sex != "All":
        df_selection_heart = df_selection_heart[df_selection_heart["sex"] == (1 if sex == "Male" else 0)]

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

    # ---- HIDE STREAMLIT STYLE ----
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Diabetes Prediction Page
elif selected == 'Diabetes Prediction':
    classifier = diabetes_model


    def predictor(pregnancies, glucose, bloodPressure, skinThickness, insulin, BMI, diabetesPedigreeFunction, age):
        prediction = classifier.predict(
            pd.DataFrame(
                [[pregnancies, glucose, bloodPressure, skinThickness, insulin, BMI, diabetesPedigreeFunction, age]],
                columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                         'DiabetesPedigreeFunction', 'Age']))
        return prediction


    def load_lottiefile1(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)


    lottie_coding2 = load_lottiefile1("Animation1_1.json")
    st_lottie(
        lottie_coding2,
        speed=0.5,
        reverse=False,
        loop=True,
        quality="low",
        height=250,
    )
    st.title('Diabetes Prediction')
    st.header('Enter the required details below')
    age = st.number_input('Age:')
    pregnancies = st.number_input('Number of Pregnancies: ')
    glucose = st.text_input('Glucose Level')
    bloodPressure = st.text_input('Blood Pressure value')
    skinThickness = st.text_input('Skin Thickness value')
    insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    diabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    if st.button('Predict Diabetes'):
        my_predict = predictor(pregnancies, glucose, bloodPressure, skinThickness, insulin, BMI,
                               diabetesPedigreeFunction, age)
        if my_predict == 1:
            st.success('There is a high risk of diabetes')
        else:
            st.success('There is a low risk of diabetes')

# Heart Disease Prediction Page
elif selected == 'Heart Disease Prediction':
    # Load the pre-trained model
    pickle_in = open("heart_model.pkl", 'rb')
    SVMPipeline = pickle.load(pickle_in)


    # Function to make predictions
    def predictor(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
        prediction = SVMPipeline.predict(
            pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                         columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                                  'oldpeak', 'slope', 'ca', 'thal'])
        )
        return prediction


    # Streamlit app
    st.title('Heart Disease Prediction')


    st.header('Enter the required details below')

    # Input fields with user-friendly selections
    age = st.number_input('Age:', help='Enter your age in years.')
    sex_options = ['Male', 'Female']
    sex = st.selectbox('Sex:', sex_options, help='Select your gender.')

    cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
    cp = st.selectbox('Chest Pain Type:', cp_options, help='Select your chest pain type.')

    trestbps = st.number_input('Resting Blood Pressure (trestbps):', help='Enter resting blood pressure in mm Hg.')
    chol = st.number_input('Serum Cholesterol (chol):', help='Enter serum cholesterol in mg/dl.')

    fbs_options = ['True', 'False']
    fbs = st.selectbox('Fasting Blood Sugar:', fbs_options, help='Select fasting blood sugar level.')

    restecg_options = ['Normal', 'ST-T Wave Normality', 'Left Ventricular Hypertrophy']
    restecg = st.selectbox('Resting Electrocardiographic Results:', restecg_options, help='Select resting ECG results.')

    thalach = st.number_input('Maximum Heart Rate Achieved (thalach):', help='Enter maximum heart rate achieved.')

    exang_options = ['Yes', 'No']
    exang = st.selectbox('Exercise Induced Angina:', exang_options, help='Select if you have exercise-induced angina.')

    oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak):',
                              help='Enter ST depression induced by exercise.')

    slope = st.number_input('Slope of the Peak Exercise ST Segment:',
                            help='Enter slope of the peak exercise ST segment.')

    ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy:',
                         help='Enter number of major vessels colored by fluoroscopy.')

    thal_options = ['0', '1', '2', '3']
    thal = st.selectbox('Thalassemia:', thal_options, help='Select your thalassemia type.')

    # Mapping user-friendly selections to numerical values
    sex_mapping = {'Male': 1, 'Female': 0}
    cp_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    fbs_mapping = {'True': 1, 'False': 0}
    restecg_mapping = {'Normal': 0, 'ST-T Wave Normality': 1, 'Left Ventricular Hypertrophy': 2}
    exang_mapping = {'Yes': 1, 'No': 0}
    thal_mapping = {'0': 0, '1': 1, '2': 2, '3': 3}

    # Button to make the prediction
    if st.button('Predict Heart Disease'):
        # Map user-friendly selections to numerical values
        sex = sex_mapping[sex]
        cp = cp_mapping[cp]
        fbs = fbs_mapping[fbs]
        restecg = restecg_mapping[restecg]
        exang = exang_mapping[exang]
        thal = thal_mapping[thal]

        # Make the prediction
        my_predict = predictor(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

        # Display the prediction
        if my_predict == 1:
            st.success('There is a high risk of heart disease.')
        else:
            st.success('Low risk of heart disease.')
elif selected == 'Brain Tumor Classification':
    loaded_model = tf.keras.models.load_model(model_path)

    # Streamlit app
    st.title("Brain Tumor Prediction")


    def load_lottiefile4(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)


    lottie_coding2 = load_lottiefile4("Animation_brain.json")
    st_lottie(
        lottie_coding2,
        speed=0.5,
        reverse=False,
        loop=True,
        quality="low",
        height=250,
    )

    # Upload MRI image through Streamlit
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.subheader("Uploaded MRI Image:")
        st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)

        # Preprocess the image
        img = image.load_img(uploaded_file, target_size=(150, 150))  # Resize to (150, 150)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        # Make prediction
        prediction = loaded_model.predict(img_array)

        # Map the class index to the tumor type
        class_mapping = {0: 'Glioma Tumor', 1: 'No Tumor', 2: 'Meningioma Tumor', 3: 'Pituitary Tumor'}
        predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
        tumor_type = class_mapping[predicted_class]

        # Display prediction
        st.subheader("Prediction:")
        confidence_score = prediction[0][predicted_class]
        st.write(f"Predicted Class: {tumor_type}")
        st.write(f"Confidence Score: {confidence_score:.2%}")

        # Display original and processed images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image:")
            st.image(img, caption="Original Image", use_column_width=True)

        with col2:
            st.subheader("Processed Image:")
            st.image(image.array_to_img(img_array[0]), caption="Processed Image", use_column_width=True)

    # Display the training metrics plot
    st.subheader("Training Metrics Visualization")

    # Your existing code for plotting training metrics
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    train_acc = [0.65, 0.72, 0.78, 0.81, 0.84, 0.87, 0.89, 0.91, 0.92, 0.93, 0.94, 0.95]
    val_acc = [0.55, 0.62, 0.68, 0.72, 0.75, 0.78, 0.80, 0.82, 0.83, 0.85, 0.86, 0.87]
    train_loss = [0.45, 0.40, 0.35, 0.32, 0.28, 0.25, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12]
    val_loss = [0.55, 0.50, 0.45, 0.42, 0.38, 0.35, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22]

    # Create figure with subplot
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Training and Validation Accuracy', 'Training and Validation Loss'],
                        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]])

    # Design
    layout = go.Layout(
        title='Epochs vs. Training and Validation Accuracy/Loss',
        title_font=dict(size=18, family='monospace', color='darkgrey'),
        showlegend=False,
        xaxis=dict(title='Epochs', showline=True, showgrid=False),
        yaxis=dict(showline=True, showgrid=False),
        template='plotly_dark'
    )

    # Add traces
    fig.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines+markers', name='Training Accuracy',
                             marker=dict(color='green', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Validation Accuracy',
                             marker=dict(color='red', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name='Training Loss',
                             marker=dict(color='green', size=10)), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Validation Loss',
                             marker=dict(color='red', size=10)), row=1, col=2)

    # Shaded areas for loss plot
    fig.add_trace(go.Scatter(
        x=epochs + epochs[::-1],
        y=train_loss + val_loss[::-1],
        fill='toself',
        fillcolor='rgba(0, 128, 0, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        hoverinfo="skip",
    ), row=1, col=2)

    # Vertical lines for key points (e.g., best epoch)
    fig.add_trace(go.Scatter(x=[epochs[np.argmax(val_acc)]], y=[val_acc[np.argmax(val_acc)]],
                             mode='markers',
                             marker=dict(color='gold', size=10, symbol='star'),
                             name='Best Epoch',
                             hoverinfo="x+y+name"), row=1, col=1)

    # Annotations
    fig.add_annotation(x=epochs[np.argmax(val_acc)], y=val_acc[np.argmax(val_acc)],
                       text=f'Best Epoch\nAccuracy: {val_acc[np.argmax(val_acc)]:.4f}',
                       showarrow=True,
                       arrowhead=5,
                       ax=-50,
                       ay=-50)

    # Table with summary statistics
    table_trace = go.Table(
        header=dict(values=['Metric', 'Training', 'Validation']),
        cells=dict(values=[['Accuracy', 'Loss'], [f'{train_acc[-1]:.4f}', f'{train_loss[-1]:.4f}'],
                           [f'{val_acc[-1]:.4f}', f'{val_loss[-1]:.4f}']]),
        domain=dict(x=[0, 0.45], y=[0, 1]),
        columnwidth=[150, 150],  # Adjust the column width as needed
        name='Summary Table'
    )
    fig.add_trace(table_trace)

    # Update layout
    fig.update_layout(layout)

    # Additional design features
    fig.update_traces(marker=dict(line=dict(width=2, color='white')),
                      selector=dict(mode='markers'))

    # Add hover information
    fig.update_layout(hovermode="x unified")
    fig.update_traces(hoverinfo="y+name")

    # Show figure
    st.plotly_chart(fig)

# Brain Prediction Page
elif selected == 'Brain Tumor Detection':
    @st.cache(allow_output_mutation=True)
    def load_image(uploaded_file):
        with BytesIO() as buffer:
            buffer.write(uploaded_file.read())
            buffer.seek(0)

            image = tf.image.decode_jpeg(buffer.getvalue())
            num_channels = image.shape[-1]

            if num_channels == 1:
                # The image is grayscale
                pass
            else:
                image = tf.image.rgb_to_grayscale(image)

            image = tf.image.resize(image, (250, 250))
            image = tf.expand_dims(image, axis=0)
            return image


    def main():
        st.title('Brain Tumor Detection')

        def load_lottiefile1(filepath: str):
            with open(filepath, "r") as f:
                return json.load(f)

        lottie_coding2 = load_lottiefile1("Animation_brain.json")
        st_lottie(
            lottie_coding2,
            speed=0.5,
            reverse=False,
            loop=True,
            quality="low",
            height=250,
        )

        uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = load_image(uploaded_file)
            prediction = brain_disease_model.predict(image)

            if prediction[0][0] > 0.5:
                st.success('The image is likely to contain a brain tumor.')
            else:
                st.success('The image is unlikely to contain a brain tumor.')


    if __name__ == '__main__':
        main()

elif selected == "Skin Cancer Classification":
    # Load the training model
    model = load_model('skin_cancer_model_with_history4.keras')

    # Streamlit app
    st.title("Skin Cancer Detection")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    # Display the uploaded image
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the image
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image

        # Make prediction
        prediction = model.predict(img_array)
        predicted_label = "Malignant" if prediction[0, 0] > 0.5 else "Benign"

        # Display the prediction
        st.write(f"Prediction: {predicted_label}")
        st.write(f"Confidence: {prediction[0, 0]:.2%}")

# Footer

footer = """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-gH2yIJqKdNHPEq0n4Mqa/HGKIhSkIHeL5AyhkYV8i59U5AR6csBvApHHNl/vI1Bx" crossorigin="anonymous">
    <footer>
        <div style='visibility: visible;margin-top:7rem;justify-content:center;display:flex;'>
            <p style="font-size:1.1rem;">
                Made by Njomza Rexhepi
                &nbsp;
                <a href="https://www.linkedin.com/in/">
                    <svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" fill="white" class="bi bi-linkedin" viewBox="0 0 16 16">
                        <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
                    </svg>          
                </a>
                &nbsp;
                <a href="https://github.com/NjomzaRexhepi">
                    <svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" fill="white" class="bi bi-github" viewBox="0 0 16 16">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                </a>
            </p>
        </div>
    </footer>
"""
st.markdown(footer, unsafe_allow_html=True)