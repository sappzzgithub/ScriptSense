import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import sys
import pathlib
from fpdf import FPDF
import tempfile
import plotly.graph_objects as go
import base64

# Add scripts folder to import path
sys.path.append(str(pathlib.Path(__file__).parent.parent / "scripts"))
from graphology_features import extract_graphology_features

# Background with blur and black overlay
def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(image_file_path):
    bin_str = get_base64_of_bin_file(image_file_path)
    page_bg_img = f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.7)), 
                    url("data:image/png;base64,{bin_str}") no-repeat center center fixed;
        background-size: cover;
        backdrop-filter: blur(5px);
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    h1, h2, h3, h4, h5, h6, p, div, span {{
        color: #f5f5f5 !important;
        font-family: 'Segoe UI', sans-serif !important;
    }}
    .trait-score {{
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 8px;
        background-color: rgba(255, 255, 255, 0.1);
        padding: 8px;
        border-radius: 8px;
        color: white !important;
    }}
    .centered-title {{
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 0.2rem;
    }}
    .stTextInput > div > input {{
        background-color: #ffffffcc;
        color: #000;
    }}
    .stFileUploader {{
        background-color: rgba(255, 255, 255, 0.2) !important;
        color: black !important;
        border-radius: 10px;
        padding: 1rem;
        font-weight: 500;
        font-size: 15px;
        box-shadow: 0 0 8px rgba(255, 255, 255, 0.2);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Traits
traits = ["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness"]

trait_descriptions = {
    "Agreeableness": "You tend to be compassionate, cooperative, and value getting along with others.",
    "Conscientiousness": "You are responsible, organized, and strive for achievement with strong impulse control.",
    "Extraversion": "You enjoy being around people, are full of energy, and often experience positive emotions.",
    "Neuroticism": "You may experience emotional instability, moodiness, and irritability.",
    "Openness": "You are imaginative, curious, and open to new experiences and ideas."
}

# CNN Model
class PersonalityCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

@st.cache_resource
def load_model():
    model = PersonalityCNN()
    model_path = pathlib.Path(__file__).parent.parent / "models" / "/Users/sakshizanjad/Desktop/ScriptSense--1/scripts/models/personality_cnn.pth"
    model.load_state_dict(torch.load(str(model_path), map_location=torch.device("cpu")))
    model.eval()
    return model

def predict_cnn(image, model):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        predicted = np.argmax(probs)
    return traits[predicted], probs

def create_gauge_chart(trait, value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': trait},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "mediumseagreen"},
            'steps': [
                {'range': [0, 50], 'color': "#ffe6e6"},
                {'range': [50, 75], 'color': "#ffffcc"},
                {'range': [75, 100], 'color': "#e6ffe6"},
            ]
        }
    ))
    fig.update_layout(height=250, margin=dict(t=0, b=0, l=0, r=0))
    return fig

def generate_pdf(name, image_path, pred_trait, scores, graph_features, fun_paragraph):
    def clean_text(text):
        return text.replace("→", "->").replace("—", "-").encode("latin1", "replace").decode("latin1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, clean_text("Personality Analysis Report"), ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, clean_text(f"Name: {name}"), ln=True)
    pdf.cell(0, 10, clean_text("Uploaded Handwriting Sample:"), ln=True)
    pdf.image(image_path, w=100)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, clean_text("CNN Prediction"), ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, clean_text(f"Predicted Trait: {pred_trait}"), ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, clean_text("Graphology Features"), ln=True)
    pdf.set_font("Arial", "", 12)
    for item in graph_features:
        feature = clean_text(item["Attribute"])
        category = clean_text(item["Writing Category"])
        behavior = clean_text(item["Psychological Personality Behavior"])
        pdf.multi_cell(0, 8, f"- {feature}: {category} -> {behavior}")

    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, clean_text("Personality Snapshot"), ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, clean_text(fun_paragraph))

    pdf_path = os.path.join(tempfile.gettempdir(), "personality_report.pdf")
    pdf.output(pdf_path)
    return pdf_path

# UI
st.set_page_config(page_title="Hybrid Personality Predictor", layout="centered")
bg_path = pathlib.Path("/Users/sakshizanjad/Desktop/ScriptSense--1/image.jpg")  
set_background(str(bg_path))

st.markdown("""
    <div class="centered-title">
        🧠 Personality Predictor from Handwriting (Hybrid)
    </div>
    <p style='text-align: center; font-size: 18px; margin-top: -0.5rem;'>Upload a handwriting image to analyze personality using both AI and Graphology.</p>
""", unsafe_allow_html=True)

name = st.text_input("Enter your name")
uploaded_file = st.file_uploader("Upload Handwriting Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Sample", use_column_width=True)

    if st.button("Analyze Personality"):
        with st.spinner("Running analysis..."):
            model = load_model()
            pred_trait, scores = predict_cnn(image, model)
            graph_features = extract_graphology_features(uploaded_file)

            traits_summary = [item['Psychological Personality Behavior']
                              for item in graph_features
                              if item['Psychological Personality Behavior'] != "Insufficient data to determine"]

            highlighted_traits = sorted(zip(traits, scores), key=lambda x: x[1], reverse=True)[:2]
            top_traits_desc = ". ".join([
                f"Your {trait.lower()} score suggests that {trait_descriptions[trait]}"
                for trait, _ in highlighted_traits
            ])

            if traits_summary:
                fun_paragraph = (
                    f"Hey {name}! ✨ Based on your handwriting analysis, you exhibit traits like "
                    f"{', '.join(t.lower() for t in traits_summary[:-1])}"
                    f"{', and ' + traits_summary[-1].lower() if len(traits_summary) > 1 else ''}.\n\n"
                    f"From a CNN perspective, you show strong alignment with traits like "
                    f"{highlighted_traits[0][0]} and {highlighted_traits[1][0]}. {top_traits_desc}.\n\n"
                    f"Altogether, this paints a vibrant and multi-dimensional picture of who you are! 🖌️"
                )
            else:
                fun_paragraph = f"{name}, we couldn't gather enough from your writing to make a full personality profile."

            temp_img_path = os.path.join(tempfile.gettempdir(), "uploaded_image.png")
            image.save(temp_img_path)

            pdf_path = generate_pdf(name, temp_img_path, pred_trait, scores, graph_features, fun_paragraph)

        st.markdown(f"<h3>CNN Prediction</h3>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style='
              background-color: rgba(255, 255, 255, 0.1);
              color: #ffffff;
              padding: 15px;
              border-radius: 10px;
              font-size: 18px;
              font-weight: bold;
              text-align: center;
              margin-bottom: 20px;
              font-family: "Segoe UI", sans-serif;
            '>
                🧠 Predicted Trait: {pred_trait}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(f"<h3>Personality Trait Scores</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])

        for i, t in enumerate(traits):
            if t == pred_trait:
                with col1:
                    st.plotly_chart(create_gauge_chart(t, scores[i]), use_container_width=True)
            else:
                with col2:
                     st.markdown(f"<div class='trait-score'>{t}: {scores[i]*100:.2f}%</div>", unsafe_allow_html=True)

        st.markdown(f"<h3>Graphology Features</h3>", unsafe_allow_html=True)
        for item in graph_features:
            st.markdown(
                f"<p>- <b>{item['Attribute']}</b>: {item['Writing Category']} -> {item['Psychological Personality Behavior']}</p>",
                unsafe_allow_html=True
            )

        st.markdown(f"<h3>Personality Snapshot</h3>", unsafe_allow_html=True)
        st.markdown(f"<p>{fun_paragraph}</p>", unsafe_allow_html=True)

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Download Full Report (PDF)",
                data=f,
                file_name=f"{name}_Personality_Report.pdf",
                mime="application/pdf"
            )
