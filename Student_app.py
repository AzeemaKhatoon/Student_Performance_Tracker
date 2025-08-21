import streamlit as st
import numpy as np
import joblib
import base64

# Load the scaler model
with open("scaler.joblib", "rb") as file:
    scaler = joblib.load(file)

# Load the regression model
with open("Student_performance_model.joblib", "rb") as file:
    model = joblib.load(file)

# Page Layout
st.set_page_config(page_title="Student Performance Tracker")

# Title and Markdown
st.title("🎯 Student Performance Tracker")
st.subheader("Track Your Future Performance...")

# Background Image Code
def get_base64_of_bin_file(bin_file):
    """
    Reads a binary file and returns its base64 encoded string.
    """
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    """
    Sets a local PNG or JPG image as the background of the Streamlit app using base64.
    """
    bin_str = get_base64_of_bin_file(png_file)
    
    # Adjust mime type based on image type (jpg or png)
    mime_type = "jpeg" if png_file.lower().endswith((".jpg", ".jpeg")) else "png"
    
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/{mime_type};base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Just provide the filename of image (it must be in the same folder or use full path correctly)
set_png_as_page_bg("Background_image.jpg")

# Information related to previous performance
hr_std = st.number_input("⏱️ :primary[Studied Hours]", min_value=0.0, max_value=24.0, step=0.5)
pr_scr = st.number_input("🏆 :primary[Previous Score]", min_value=0.0, max_value=100.0, step=0.1)
sp_pp_solv = st.number_input("📄 :primary[Sample Papers Solved]", min_value=0, max_value=50, step=1)
hr_slp = st.number_input("😴 :primary[Sleep Hours]", min_value=0.0, max_value=24.0, step=0.5)
act = st.radio("⚽ :primary[Extracurricular Activity]", ["Yes","No"])
if act == "Yes":
    activity = 1
else:
    activity = 0  

# Prepare input in correct order
data_input = [[hr_std, pr_scr, activity, hr_slp, sp_pp_solv]]

# Scale the input
scaled_input = scaler.transform(data_input)

if st.button("📊 Check Performance"):
    pred = model.predict(scaled_input)
    
    # Using metric display for a clean look
    st.success("✅ Prediction Completed!")
    st.metric(label="📈 :primary[Pridicted Performance]", value=f"{round(pred[0], 2)}")