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


st.set_page_config(
    page_title="MPsSpecClassify",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root{
      --primary:#0ea5a0; /* teal */
      --accent:#6366f1;  /* indigo */
      --warn:#f59e0b;    /* amber */
      --text:#0f172a;    /* slate-900 */
      --muted:#64748b;   /* slate-500 */
      --bg:#ffffff;
      --bg2:#f8fafc;
      --border:#e2e8f0;
      --radius:14px;
      --shadow:0 8px 24px rgba(2, 6, 23, 0.08);
    }

    .stApp { background: var(--bg); color: var(--text) !important; }
    .main  { background: var(--bg); padding: 8px 0 24px; }

    .app-title{
      font-size: 34px; font-weight: 800; letter-spacing:.3px;
      text-align:center; margin: 8px 0 18px;
      background: linear-gradient(90deg, var(--primary), var(--accent));
      -webkit-background-clip: text; background-clip: text; color: transparent;
    }

    .glass {
      background: var(--bg2);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px 18px;
      box-shadow: var(--shadow);
      margin-bottom: 16px;
    }

    section[data-testid="stSidebar"] {
      background: var(--bg2);
      border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] h1, h2, h3, h4, h5, h6 {
      color: var(--text) !important;
    }

    .stButton>button {
      border-radius: 999px !important;
      border: 1px solid var(--primary) !important;
      background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
      color: white !important; font-weight: 700 !important;
      padding: 0.5rem 1.1rem !important;
      box-shadow: var(--shadow) !important;
    }
    .stButton>button:hover { filter: brightness(0.95); }

    div[data-testid="stFileUploader"] {
      border: 2px dashed var(--primary) !important;
      background: var(--bg2);
      border-radius: var(--radius);
      padding: 18px;
    }

    .stPlotlyChart { border-radius: var(--radius); overflow: hidden; }

    div[data-testid="stDataFrame"] {
      background: var(--bg); border:1px solid var(--border);
      border-radius: var(--radius); box-shadow: var(--shadow);
    }

    .bar { height: 10px; background: #e5e7eb; border-radius: 999px; overflow:hidden; }
    .bar > span { display:block; height:100%; background: linear-gradient(90deg,var(--primary),var(--accent)); }

    .img-cap { color: var(--muted); font-size: 12px; margin-top: 4px; text-align:center; }

    .markdown-text-container, .stMarkdown { color: var(--text) !important; }
    h1,h2,h3,h4,h5,h6,p { color: var(--text) !important; }
div[data-testid="stFileUploader"]{
  border: 2px dashed var(--primary) !important;
  background: var(--bg2);
  border-radius: var(--radius);
  padding: 18px 18px 14px 18px;
  box-shadow: var(--shadow);
}

div[data-testid="stFileUploader"] > div {
  gap: 10px;
}

div[data-testid="stFileUploader"] .stMarkdown, 
div[data-testid="stFileUploader"] p, 
div[data-testid="stFileUploader"] label {
  color: var(--text) !important;
  font-weight: 600;
}

div[data-testid="stFileUploader"] button {
  border-radius: 999px !important;
  border: 1px solid var(--primary) !important;
  background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  padding: 0.5rem 1.1rem !important;
  box-shadow: var(--shadow) !important;
}

div[data-testid="stFileUploader"] button:hover {
  filter: brightness(0.95);
  transform: translateY(-1px);
  transition: all .15s ease;
}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileDetails"] {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 12px;
}

div[data-testid="stFileUploader"] svg {
  color: var(--primary);
}

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<div class="app-title">MPsSpecClassify</div>', unsafe_allow_html=True)


st.sidebar.title('MPsSpecClassify')
image = Image.open('Logo.png')
st.sidebar.image(image, caption='', use_container_width=True)

st.sidebar.markdown("### Navigation")
st.sidebar.divider()

show_home = True
show_contact = False
show_tutorial = False

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


def style_fig(fig, title):
    fig.update_layout(
        title=title,
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=16, color="black"),
        title_font=dict(color="black"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(font=dict(color="black"))
    )
    fig.update_xaxes(tickfont=dict(color="black"), title_font=dict(color="black"))
    fig.update_yaxes(tickfont=dict(color="black"), title_font=dict(color="black"))
    return fig


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

    feature_extractor = torch.load('feature_extractor_full.pth', weights_only=False)


    def infer_single_image(image_path, model, transform):
        image = Image.open(image_path).convert('RGB')
        images = transform(image)
        images = images.unsqueeze(0)  # batch

        with torch.no_grad():
            features = model(images)
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
            x=df['A1'],
            y=df['A2'],
            mode='lines',
            name='Spectrum',
            line=dict(color='#b84848')
        ))
        fig = style_fig(fig, "Plot Spectrum")

        fig.update_xaxes(
            title_text='Wavelength (cm⁻¹)',
            title_font=dict(color='black'),
            tickfont=dict(color='black', size=16),
            autorange='reversed'
        )
        fig.update_yaxes(
            title_text='Intensity (-)',
            title_font=dict(color='black'),
            tickfont=dict(color='black', size=16),
            gridcolor="#e2e8f0"
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
        fig = style_fig(fig, "Spectrum (Baseline Correction Polynomial)")

        fig.update_xaxes(
            title_text='Wavelength (cm⁻¹)',
            title_font=dict(color='black'),
            tickfont=dict(color='black', size=16),
            autorange='reversed'
        )
        fig.update_yaxes(
            title_text='Intensity (-)',
            title_font=dict(color='black'),
            tickfont=dict(color='black', size=16),
            gridcolor="#e2e8f0"
        )
        return fig

    def generate_spectrogram_image(data):
        frequencies, times, Sxx = spectrogram(data, fs=1)
        plt.figure(figsize=(10, 5))
        plt.imshow(10 * np.log10(Sxx), aspect='auto', cmap='inferno', origin='lower',
                   extent=[times.min(), times.max(), frequencies.min(), frequencies.max()])
        plt.axis('off')
        plt.savefig("image.png", bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()

    def polynomial_baseline_correction(x, degree=2):
        coeffs = np.polyfit(range(len(x)), x, degree)
        baseline = np.polyval(coeffs, range(len(x)))
        return x - baseline


    if uploaded_file is not None:
        with st.spinner('Processing spectrum...'):
            df = pd.read_csv(uploaded_file, header=None).dropna()
            df.columns = [f'A{i+1}' for i in range(df.shape[1])]

            a = df.set_index('A1').T
            a = a.iloc[0]
            a = pd.DataFrame(a).transpose()

            col2_df = pd.read_csv('colTrue2.csv')

            df_plot = df.set_index('A1').T
            df_plot = df_plot.apply(polynomial_baseline_correction, axis=1)

        if show_spectrum_plot:
            with st.container():
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                st.subheader("Spectrum Plot")
                fig = plot_spectrum(df)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if show_Preprocess:
            with st.container():
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                st.subheader("Spectrum Preprocess Plot")
                fig2 = plot_spectrum_Clean(df_plot)
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner('Generating spectrogram...'):
            generate_spectrogram_image(df['A2'].values)

        if show_image:
            with st.container():
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                st.subheader("Spectrogram")
                st.image("image.png", use_container_width=True)
                st.markdown('<div class="img-cap">Spectrogram generated from uploaded spectrum</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner('Running inference...'):
            image_path = 'image.png'
            inference_features = infer_single_image(image_path, feature_extractor, transform)
            inference_df = add_features_to_dataframe(inference_features, image_path)
            inference_df = inference_df.reset_index(drop=True)

            a = a.reset_index(drop=True)

            merged_df = pd.concat([inference_df, a], axis=1)
            merged_df = merged_df.drop(columns=['filename', 'label'])
            merged_df = pd.DataFrame(merged_df)

            merged_df.columns = merged_df.columns.astype(str)
            cols_in_col = col2_df.columns
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

                    results.append({
                        'Index': i,
                        'Predicted Class': predicted_class,
                        'Confidences': confidence_details,
                        'Sorted': sorted_confidences
                    })

                for result in results:
                    with st.container():
                        st.markdown('<div class="glass">', unsafe_allow_html=True)
                        st.markdown(f"**Predicted Class:** <span style='color:#0f172a; font-size:24px;'> {result['Predicted Class']}</span>", unsafe_allow_html=True)
                        st.markdown("**Confidences:**", unsafe_allow_html=True)
                        for cls, prob in result['Sorted']:
                            st.markdown(
                                f"""
                                <div style="margin:6px 0;">
                                  <div style="display:flex; justify-content:space-between; font-weight:600;">
                                    <span>{cls}</span><span>{prob}%</span>
                                  </div>
                                  <div class="bar"><span style="width:{prob}%"></span></div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.error(f"Model file not found at {model_path}")

        if os.path.exists("image.png"):
            os.remove("image.png")


if show_contact:
    st.title("Contact Us")
    with st.container():
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.header("Contact Us")
        st.write("Email: pensiri.a@phuket.psu.ac.th")
        st.markdown('</div>', unsafe_allow_html=True)

if show_tutorial:
    st.title("Tutorial")
    with st.container():
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.header("Usage Tutorial")
        st.write("")
        st.write("")
        st.write("")
        st.subheader("1. Uploading Files")
        st.image("1.png", use_container_width=True)

        st.subheader("2. Display Options")
        st.image("2.png", use_container_width=True)
        st.image("3.png", use_container_width=True)
        st.write(
            "The graph will display points showing absorption or transmission values related to wavelength, which aids in analyzing which wavelengths of light are being absorbed."
        )

        st.image("4.png", use_container_width=True)
        st.write(
            "This will show a graph of the processed data after applying baseline correction using a polynomial."
        )

        st.image("5.png", use_container_width=True)
        st.write(
            "The spectrogram will be displayed as a 2D graph with the x-axis representing time and the y-axis representing frequency. "
            "Colors or indicators will be used to show the energy levels of the signal at each time and frequency interval."
        )

        st.subheader("3. Making Predictions")
        st.image("6.png", use_container_width=True)
        st.write(
            "After uploading the CSV file and selecting the display options: The application will process the data and extract features using the trained model. "
            "Then, it will predict the type of polymer based on the spectrum features. The predicted class and confidence score will be displayed below in the prediction section."
        )
        st.markdown('</div>', unsafe_allow_html=True)

