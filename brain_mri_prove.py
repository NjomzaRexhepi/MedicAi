import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Load the pre-trained EfficientNet model
model_path = "brain_tumor_prediction_mri_model.h5"  # Replace with your model file path
loaded_model = tf.keras.models.load_model(model_path)

# Streamlit app
st.title("Brain Tumor Prediction")

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
fig = make_subplots(rows=1, cols=2, subplot_titles=['Training and Validation Accuracy', 'Training and Validation Loss'],
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
