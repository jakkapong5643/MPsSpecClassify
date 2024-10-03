import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import torchvision.transforms as transforms
from PIL import Image
import plotly.graph_objects as go

# Set up the Streamlit app layout and styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.1);
    }
    .stApp {
        background-color: #ffffff;
    }
    .title {
        color: #000000;
        font-family: 'Arial', sans-serif;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #f7f7f7;
        padding: 20px;
        border-radius: 10px;
    }
    .stFileUploader, .stDataFrame, h2, h3, h4, h5, h6, p {
        color: #000000;
        background-color: #ffffff;
        border: 1px solid #cccccc;
        padding: 10px;
        border-radius: 5px;
    }
    .css-1vbqpe4.edgvbvh3 {
        background-color: #f0f0f0;
        border: 2px dashed #007bff;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #333;
        font-size: 16px;
    }
    .css-1vbqpe4.edgvbvh3:hover {
        background-color: #e9ecef;
        border-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title('MPsSpecClassify')
image = Image.open('Logo.png')
st.sidebar.image(image, caption='', use_column_width=True)

show_home = True
show_contact = False
show_tutorial = False

# Page navigation buttons
if st.sidebar.button("MPsSpecClassify"):
    show_home = True
    show_contact = False
    show_tutorial = False

if st.sidebar.button("Contact Us"):
    show_home = False
    show_contact = True
    show_tutorial = False

if st.sidebar.button("Tutorial"):
    show_home = False
    show_contact = False
    show_tutorial = True

if show_home:
    uploaded_file = st.file_uploader("", type="csv")

    col1, col2, col3 = st.columns(3)

    with col1:
        show_spectrum_plot = st.checkbox("Show Spectrum", value=True)

    with col2:
        show_Preprocess = st.checkbox("Show Preprocess", value=False)

    with col3:
        show_image = st.checkbox("Show Spectrogram", value=False)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    model = torch.load('alexnet_pretrained.pth')
    feature_extractor = model


    def infer_single_image(image_path, model, transform):
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)  
        
        with torch.no_grad():
            features = model(image)
            features = features.view(features.size(0), -1)  
        return features.cpu().numpy()

    def add_features_to_dataframe(inference_features, image_path):
        features_df = pd.DataFrame(inference_features, columns=[f'{i}' for i in range(inference_features.shape[1])])
        filename = os.path.basename(image_path)
        features_df['filename'] = f"<span class='dataframe_filename'>{filename}</span>"
        features_df['label'] = None  
        return features_df

    def plot_spectrum(df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Unnamed: 0'], 
            y=df['Unnamed: 1'],
            mode='lines',
            name='Spectrum',
            line=dict(color='#b84848')
        ))

        fig.update_layout(
            title='Plot Spectrum',
            xaxis_title='Wavelength (cm⁻¹)',
            yaxis_title='Intensity (-)',
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            hovermode='closest',
            xaxis=dict(
                showgrid=False, 
                autorange='reversed', 
                tickfont=dict(color='black', size=20),  
                title_font=dict(color='black', family='Times New Roman', size=20)  
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='#d9d9d9',
                gridwidth=0.5,
                tickfont=dict(color='black', size=20),
                title_font=dict(color='black', family='Times New Roman', size=20)
            )
        )
        return fig

    def plot_spectrum_Clean(df, line_color='#b84848'):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.columns,  
            y=df.iloc[0],  
            mode='lines',
            name='Spectrum Preprocess',
            line=dict(color=line_color)
        ))

        fig.update_layout(
            title='Spectrum (Baseline Correction Polynomial)',
            xaxis_title='Wavelength (cm⁻¹)',
            yaxis_title='Intensity (-)',
            plot_bgcolor='#ffffff', 
            paper_bgcolor='#ffffff',  
            hovermode='closest',
            xaxis=dict(
                showgrid=False, 
                autorange='reversed', 
                tickfont=dict(color='black', size=20),  
                title_font=dict(color='black', family='Times New Roman', size=20)  
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='#d9d9d9',
                gridwidth=0.5,
                tickfont=dict(color='black', size=20),
                title_font=dict(color='black', family='Times New Roman', size=20)
            )
        )

        return fig

    def generate_spectrogram_image(data):
        frequencies, times, Sxx = spectrogram(data, fs=1)
        plt.figure(figsize=(10, 5))
        plt.imshow(10 * np.log10(Sxx), aspect='auto', cmap='inferno', origin='lower', extent=[times.min(), times.max(), frequencies.min(), frequencies.max()])
        plt.axis('off')
        plt.savefig("image.png", bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()

    def polynomial_baseline_correction(x, degree=2):
        coeffs = np.polyfit(range(len(x)), x, degree)
        baseline = np.polyval(coeffs, range(len(x)))
        return x - baseline

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file).dropna()
        a = df.set_index('Unnamed: 0').T
        a = a.iloc[0]
        a = pd.DataFrame(a).transpose()

        col2 = pd.read_csv('colTrue2.csv')

        # st.plotly_chart(plot_spectrum(df))

        df_plot = df.set_index('Unnamed: 0').T
        df_plot = df_plot.apply(polynomial_baseline_correction, axis=1)

        if show_spectrum_plot:
            st.subheader("Spectrum Plot")
            fig = plot_spectrum(df)
            st.plotly_chart(fig)

        if show_Preprocess:
            st.subheader("Spectrum Preprocess Plot")
            st.plotly_chart(plot_spectrum_Clean(df_plot))

        generate_spectrogram_image(df['Unnamed: 1'].values)

        if show_image:
            st.subheader("Spectrogram")
            st.image("image.png", use_column_width=True)

        image_path = 'image.png'
        inference_features = infer_single_image(image_path, feature_extractor, transform)
        inference_df = add_features_to_dataframe(inference_features, image_path)
        inference_df = inference_df.reset_index(drop=True)
        a = a.reset_index(drop=True)

        merged_df = pd.concat([inference_df, a], axis=1)
        merged_df = merged_df.drop(columns=['filename', 'label'])
        merged_df = pd.DataFrame(merged_df)
        
        merged_df.columns = merged_df.columns.astype(str)
        cols_in_col = col2.columns
        cols_in_col = [str(col) for col in cols_in_col]

        merged_df = merged_df[merged_df.columns[merged_df.columns.isin(cols_in_col)]]

        model_path = 'ModelTrue2.pkl'
        st.subheader("Predictions")
        merged_df.columns = merged_df.columns.astype(str)
        merged_df = merged_df.apply(polynomial_baseline_correction, axis=1)

        class_name = ['Polyamide (PA)','Polyethylene (PE)','Polyethylene terephthalate (PET)','Polypropylene (PP)','Polystyrene (PS)']
        class_name = np.array(class_name)

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            predictions = model.predict(merged_df)
            probabilities = model.predict_proba(merged_df)

            results = []

            for i in range(len(predictions)):
                class_confidences = [(cls, int(probabilities[i][j] * 100)) for j, cls in enumerate(class_name)]
                
                sorted_confidences = sorted(class_confidences, key=lambda x: x[1], reverse=True)
                
                predicted_index = np.argmax(probabilities[i])
                predicted_class = class_name[predicted_index]
                
                confidence_details = ''.join([
                    f"<div style='margin:0; padding:0; display:inline-block; font-size:20px; color:black;'>{cls}: {prob}%</div><br>"
                    for cls, prob in sorted_confidences
                ])
                
                result = {
                    'Index': i,
                    'Predicted Class': predicted_class,
                    'Confidences': confidence_details
                }
                results.append(result)

            for result in results:
                st.markdown(f"**Predicted Class:** <span style='color:black; font-size:24px;'> {result['Predicted Class']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidences:**<br><div style='background-color:white; padding:10px;'><span style='color:black; font-size:20px;'>{result['Confidences']}</span></div>", unsafe_allow_html=True)

        else:
            st.error(f"Model file not found at {model_path}")

        if os.path.exists("image.png"):
            os.remove("image.png")



if show_contact:
    st.title("Contact Us")
    st.header("Contact Us")
    st.write(
        "Email: pensiri.a@phuket.psu.ac.th"
    )

if show_tutorial:
    st.title("Tutorial")
    st.header("Usage Tutorial")
    st.write(
        ""
    )
    st.write(
        ""
    )
    st.write(
        ""
    )
    st.subheader("1. Uploading Files")
    st.image("1.png", use_column_width=True)

    st.write(
        "Click on the 'Upload' button in the sidebar to upload a CSV file. The CSV should contain the spectral data."
    )
    
    st.subheader("2. Display Options")
    st.image("2.png", use_column_width=True)

    st.image("3.png", use_column_width=True)

    st.write(
        "The graph will display points showing absorption or transmission values related to wavelength, which aids in analyzing which wavelengths of light are being absorbed."
    )
    
    st.image("4.png", use_column_width=True)
    st.write(
        "This will show a graph of the processed data after applying baseline correction using a polynomial."
    )

    st.image("5.png", use_column_width=True)
    st.write(
        "The spectrogram will be displayed as a 2D graph with the x-axis representing time and the y-axis representing frequency. "
        "Colors or indicators will be used to show the energy levels of the signal at each time and frequency interval."
    )
    
    st.subheader("3. Making Predictions")
    st.image("6.png", use_column_width=True)
    st.write(
        "After uploading the CSV file and selecting the display options: The application will process the data and extract features using the trained model. "
        "Then, it will predict the type of polymer based on the spectrum features. The predicted class and confidence score will be displayed below in the prediction section."
    )
