import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Nettoyeur de plans", page_icon="🏠", layout="wide")
st.title("🏠 Nettoyeur de plans architecturaux")
st.write("Déposez un plan JPG/JPEG/PNG.")

def clean_plan(image_pil):
    img = np.array(image_pil.convert("RGB"))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(
        hsv, np.array([125, 45, 60]), np.array([179, 255, 255])
    )
    color_mask = cv2.dilate(color_mask, np.ones((3,3), np.uint8), iterations=1)
    result = cv2.inpaint(bgr, color_mask, 3, cv2.INPAINT_TELEA)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    text_mask = np.zeros_like(gray)

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 3 <= w <= 180 and 3 <= h <= 45 and 8 <= area <= 1800:
            if w / max(h, 1) < 15:
                text_mask[labels == i] = 255

    text_mask = cv2.dilate(text_mask, np.ones((2,2), np.uint8), iterations=1)
    result = cv2.inpaint(result, text_mask, 2, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

uploaded = st.file_uploader("Déposez votre plan ici", type=["jpg","jpeg","png"])

if uploaded:
    original = Image.open(uploaded).convert("RGB")
    with st.spinner("Nettoyage du plan…"):
        cleaned = clean_plan(original)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original")
        st.image(original, use_container_width=True)
    with c2:
        st.subheader("Résultat")
        st.image(cleaned, use_container_width=True)

    buf = io.BytesIO()
    cleaned.save(buf, format="JPEG", quality=95)
    st.download_button(
        "⬇️ Télécharger le plan nettoyé",
        data=buf.getvalue(),
        file_name="plan_nettoye.jpg",
        mime="image/jpeg",
        use_container_width=True,
    )
