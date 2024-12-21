import streamlit as st


def add_public_images_to_session_state():
    import os
    from PIL import Image

    public_folder = "public"
    if not os.path.exists(public_folder):
        raise FileNotFoundError(f"The folder '{public_folder}' does not exist.")

    for filename in os.listdir(public_folder):
        if filename.endswith(".jpeg"):
            image_path = os.path.join(public_folder, filename)
            image = Image.open(image_path)
            st.session_state[filename] = image
