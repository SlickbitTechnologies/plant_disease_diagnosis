import streamlit as st
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

st.header("Plant Disease Diagnosis")

uploaded_file = st.file_uploader("Upload image to find disease of plant", type=["jpg", "jpeg", "png","webp"])

genai.configure(api_key= os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

def upload_to_gemini(image_path, mime_type=None):
    file = genai.upload_file(image_path, mime_type=mime_type)
    return file

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with open("temp_uploaded_image.jpeg", "wb") as f:
        f.write(uploaded_file.getvalue())

    with st.spinner("Analyzing the image for plant diseases..."):
        try:
            gemini_file = upload_to_gemini("temp_uploaded_image.jpeg", mime_type="image/jpeg")

            chat_session = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            gemini_file,
                            "Identify the disease affecting the plant and provide its cure or treatment steps.",
                            "Highlights the disease name and important terms",
                            "Also mention at the end what is the plant and requirements to maintain it healthy."
                            "If no disease found return 'No disease found.'"

                        ],
                    }
                ]
            )

            response = chat_session.send_message("Please analyze the uploaded image.")

            st.subheader("Diagnosis and Cure")
            st.write(response.text)

        except Exception as e:
            st.error(f"An error occurred: {e}")
