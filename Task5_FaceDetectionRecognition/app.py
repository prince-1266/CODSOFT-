"""
Face Detection & Recognition
CODSOFT AI Internship | Task 5
Run: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from src.face_recognizer import FaceRecognizer

st.set_page_config(page_title="Face Detection & Recognition", page_icon="🎯", layout="centered")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    .stStatusWidget {display: none;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return FaceRecognizer()

recognizer = load_model()

st.title("Face Detection & Recognition")
st.caption("CODSOFT AI Internship — Task 5")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Detect", "Recognize", "Register", "People & Train"])


def get_image(source, file_key, cam_key):
    """Get image from either file upload or camera based on selected source."""
    if source == "Upload file":
        file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"], key=file_key)
        if file:
            return cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    else:
        photo = st.camera_input("Take a photo", key=cam_key)
        if photo:
            return cv2.imdecode(np.frombuffer(photo.read(), np.uint8), cv2.IMREAD_COLOR)
    return None


# ── DETECT ──
with tab1:
    st.header("Detect Faces")
    source = st.radio("Input method", ["Upload file", "Use camera"], horizontal=True, key="det_method")
    image = get_image(source, "det_file", "det_cam")

    if image is not None:
        faces = recognizer.detector.detect(image)
        result = recognizer.detector.draw_boxes(image, faces)
        col1, col2 = st.columns(2)
        col1.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original")
        col2.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"{len(faces)} face(s) found")

# ── RECOGNIZE ──
with tab2:
    st.header("Recognize Faces")
    if not recognizer.is_trained:
        st.info("Train the model first in People & Train tab.")
    source = st.radio("Input method", ["Upload file", "Use camera"], horizontal=True, key="rec_method")
    image = get_image(source, "rec_file", "rec_cam")

    if image is not None:
        result, matches = recognizer.recognize(image)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Result")
        if matches:
            for i, m in enumerate(matches):
                st.write(f"**Face {i+1}:** {m['name']} — {m['match']}% match")
        else:
            st.write("No faces found.")

# ── REGISTER ──
with tab3:
    st.header("Register a Face")
    name = st.text_input("Person's name")
    method = st.radio("How to add photos?", ["Upload files", "Use camera"], horizontal=True, key="reg_method")

    if method == "Upload files":
        files = st.file_uploader(
            "Upload face photos (5-10 recommended)",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key="reg"
        )
        if files and name:
            if st.button("Register"):
                count = 0
                for f in files:
                    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
                    ok, _ = recognizer.register_face(img, name.strip())
                    if ok:
                        count += 1
                if count:
                    st.success(f"Registered {count} face(s) for {name}. Now train the model.")
                else:
                    st.error("No faces detected. Use clearer photos.")
    else:
        if not name:
            st.warning("Enter the person's name above first.")
        else:
            st.write(f"Capturing for: **{name}**")
            photo = st.camera_input("Take a photo", key="reg_cam")
            if photo:
                img = cv2.imdecode(np.frombuffer(photo.read(), np.uint8), cv2.IMREAD_COLOR)
                ok, msg = recognizer.register_face(img, name.strip())
                if ok:
                    st.success(msg + " Click 'Clear photo' and take another one.")
                else:
                    st.error(msg)

# ── PEOPLE & TRAIN ──
with tab4:
    st.header("Registered People")
    people = recognizer.get_people()

    if people:
        for pname, count in people.items():
            col1, col2 = st.columns([4, 1])
            col1.write(f"**{pname}** — {count} image(s)")
            if col2.button("Delete", key=f"d_{pname}"):
                recognizer.delete_person(pname)
                st.rerun()
    else:
        st.write("No people registered yet.")

    st.divider()
    st.header("Train Model")
    total = sum(people.values()) if people else 0
    num_people = len(people)

    if st.button("Train Model", disabled=(num_people < 2)):
        stats, msg = recognizer.train()
        if stats:
            st.success(msg)
            st.cache_resource.clear()
            st.rerun()
        else:
            st.error(msg)

    if num_people < 2:
        st.caption("Need at least 2 different people registered to train.")
