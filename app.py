import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Nettoyeur de plans", page_icon="🏠", layout="wide")
st.title("🏠 Nettoyeur de plans architecturaux")
st.write("Déposez un plan JPG/JPEG/PNG.")


def normalize_gray(gray):
    # Corrige doucement les fonds gris/jaunis sans écraser les traits fins.
    g = gray.astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), 25)
    norm = (g / np.maximum(bg, 1.0)) * 210.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    return norm


def line_masks(gray):
    # Recherche de grandes lignes architecturales (murs, cloisons, menuiseries).
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 9)
    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, hker)
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vker)
    protect = cv2.bitwise_or(h, v)
    protect = cv2.dilate(protect, np.ones((3, 3), np.uint8), iterations=1)
    return protect


def colored_annotation_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Couleurs saturées : rouge, magenta, bleu, vert, orange, etc.
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((sat > 75) & (val > 60)).astype(np.uint8) * 255
    # Evite de considérer de minuscules variations de compression comme annotation.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def text_like_mask(gray, protect):
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    mask = np.zeros_like(gray)

    H, W = gray.shape
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 8 or area > max(2500, int(W * H * 0.002)):
            continue
        if w > W * 0.35 or h > H * 0.35:
            continue
        aspect = w / max(h, 1)
        # Petits objets typiques des caractères, chiffres, légendes et repères.
        textish = ((3 <= w <= 260 and 3 <= h <= 70 and 0.15 <= aspect <= 18)
                   or (area <= 600 and w <= 100 and h <= 100))
        if not textish:
            continue

        comp = (labels == i).astype(np.uint8) * 255
        # Un composant qui touche fortement une grande ligne est potentiellement un mur
        # portant une inscription : on évite de l'effacer aveuglément.
        overlap = cv2.countNonZero(cv2.bitwise_and(comp, protect))
        ratio = overlap / max(area, 1)
        if ratio > 0.22:
            continue
        mask[labels == i] = 255

    # Regroupe les caractères proches pour éviter les petites bavures séparées.
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def clean_plan(image_pil):
    rgb = np.array(image_pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    original = bgr.copy()

    gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = normalize_gray(gray0)
    protect = line_masks(gray)

    # 1) Couleurs d'annotations : on les retire, sauf là où une ligne architecturale
    #    protégée passe dessous.
    color = colored_annotation_mask(bgr)
    color = cv2.bitwise_and(color, cv2.bitwise_not(protect))

    # 2) Texte/cotations : masque prudent, avec protection des longues lignes.
    text = text_like_mask(gray, protect)

    # Les annotations colorées sont prioritaires ; le texte noir reste prudent.
    remove = cv2.bitwise_or(color, text)

    # Réduit les masques isolés et évite les gros aplats.
    remove = cv2.morphologyEx(remove, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    remove = cv2.dilate(remove, np.ones((2, 2), np.uint8), iterations=1)
    remove[protect > 0] = 0

    # Reconstruction locale. NS est plus douce sur les plans que TELEA dans beaucoup de cas.
    result = cv2.inpaint(original, remove, 3, cv2.INPAINT_NS)

    # Renforce les grandes lignes protégées à partir de l'original pour éviter les murs
    # trop effacés après reconstruction.
    p = protect > 0
    result[p] = original[p]

    # Petit nettoyage des halos de couleur restants, sans toucher aux lignes protégées.
    remaining = colored_annotation_mask(result)
    remaining[protect > 0] = 0
    result = cv2.inpaint(result, remaining, 2, cv2.INPAINT_NS)
    result[protect > 0] = original[protect > 0]

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


uploaded = st.file_uploader("Déposez votre plan ici", type=["jpg", "jpeg", "png"])

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
