import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

VERSION = "V3 — protection architecturale"

st.set_page_config(page_title="Nettoyeur de plans", page_icon="🏠", layout="wide")
st.title("🏠 Nettoyeur de plans architecturaux")
st.caption(VERSION)
st.write("Déposez un plan JPG/JPEG/PNG.")


def resize_work(img, max_side=2600):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        img2 = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        img2 = img.copy()
    return img2, scale


def color_annotation_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Strongly saturated colored markup: magenta/red/blue/green/orange.
    masks = []
    ranges = [
        ((145, 35, 45), (179, 255, 255)),
        ((0, 45, 45), (15, 255, 255)),
        ((16, 45, 45), (45, 255, 255)),
        ((46, 40, 45), (100, 255, 255)),
        ((101, 35, 45), (144, 255, 255)),
    ]
    for lo, hi in ranges:
        masks.append(cv2.inRange(hsv, np.array(lo), np.array(hi)))
    m = np.maximum.reduce(masks)
    # Very light gray drafting lines are not selected because saturation is low.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    return cv2.dilate(m, np.ones((3,3), np.uint8), iterations=1)


def dark_ink_mask(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Local contrast threshold handles gray/yellow/aged backgrounds better than a fixed threshold.
    bg = cv2.GaussianBlur(gray, (0,0), 15)
    norm = cv2.divide(gray, np.maximum(bg, 1), scale=180)
    bw = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 7)
    return bw


def line_protection(bw):
    """Protect long, coherent architectural strokes while avoiding isolated text blobs."""
    h, w = bw.shape
    protected = np.zeros_like(bw)

    # Horizontal/vertical morphology extracts long drafting strokes.
    for length in [18, 35, 70, 120]:
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
        protected = cv2.bitwise_or(protected, cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk))
        protected = cv2.bitwise_or(protected, cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk))

    # Close small gaps so walls remain continuous under overlaid text.
    protected = cv2.morphologyEx(protected, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    # Hough lines reinforce long straight walls and dimension lines. They are protected
    # only where the supporting dark pixels are present, reducing false protection.
    edges = cv2.Canny(bw, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=max(25, min(h,w)//80),
                            minLineLength=max(25, min(h,w)//18), maxLineGap=8)
    if lines is not None:
        for x1,y1,x2,y2 in lines.reshape(-1, 4):
            dx, dy = x2-x1, y2-y1
            length = (dx*dx + dy*dy) ** 0.5
            if length < max(25, min(h,w)//25):
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            near_axis = angle < 7 or angle > 83
            if near_axis:
                cv2.line(protected, (x1,y1), (x2,y2), 255, 2)

    # Do not allow the protected layer to become an enormous filled region.
    return protected


def candidate_text_mask(bw, protected):
    """Find compact dark components likely to be lettering/cotation marks.
    Components touching long protected strokes are trimmed rather than protected wholesale.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    mask = np.zeros_like(bw)
    H, W = bw.shape

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 6 or area > max(2600, int(H*W*0.0015)):
            continue
        if w > 260 and h > 70:
            continue
        aspect = w / max(h, 1)
        fill = area / max(w*h, 1)
        # Typical text/number glyph groups are compact, small or moderately elongated.
        likely = ((2 <= h <= 70 and 2 <= w <= 240 and area >= 8) or
                  (w <= 420 and h <= 28 and area >= 12))
        if not likely:
            continue
        # Avoid removing dense architectural symbols/furniture blobs wholesale.
        if fill > 0.72 and w > 20 and h > 20:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        # If most of this component is coincident with a long stroke, keep that stroke
        # and remove only the residual compact ink around it.
        overlap = cv2.bitwise_and(comp, protected)
        removable = cv2.subtract(comp, overlap)
        if cv2.countNonZero(removable) >= max(4, int(0.12*area)):
            mask = cv2.bitwise_or(mask, removable)

    return mask


def remove_text_without_cutting_lines(bgr, remove_mask, protected):
    # Expand the mask slightly around glyphs, but subtract protected architectural strokes.
    m = cv2.dilate(remove_mask, np.ones((3,3), np.uint8), iterations=1)
    # Keep a narrow protected corridor around coherent lines.
    protect_wide = cv2.dilate(protected, np.ones((3,3), np.uint8), iterations=1)
    m = cv2.subtract(m, protect_wide)

    # Two-pass inpainting: small radius first, then a very restrained second pass.
    out = cv2.inpaint(bgr, m, 2.0, cv2.INPAINT_NS)
    return out


def repair_protected_lines(original, cleaned, protected):
    """Put back dark architectural strokes, but only where they were confidently detected."""
    p = cv2.GaussianBlur(protected, (3,3), 0)
    alpha = (p.astype(np.float32) / 255.0)[...,None] * 0.92
    # Recover original pixels along protected strokes; this preserves wall continuity.
    return (cleaned.astype(np.float32)*(1-alpha) + original.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)


def clean_plan(image_pil):
    original = np.array(image_pil.convert("RGB"))
    bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    work, scale = resize_work(bgr)

    color = color_annotation_mask(work)
    bw = dark_ink_mask(work)
    protected = line_protection(bw)

    # Color annotations are always candidates, but architectural lines underneath are protected.
    color_remove = cv2.subtract(color, cv2.dilate(protected, np.ones((5,5), np.uint8), iterations=1))

    text_remove = candidate_text_mask(bw, protected)
    remove = cv2.bitwise_or(color_remove, text_remove)

    # Avoid giant connected removals which can destroy a drawing region.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(remove, 8)
    safe = np.zeros_like(remove)
    H, W = remove.shape
    for i in range(1, n):
        x,y,w,h,area = stats[i]
        if area <= int(H*W*0.015) and not (w > W*0.55 and h > H*0.20):
            safe[labels == i] = 255
    remove = safe

    cleaned = remove_text_without_cutting_lines(work, remove, protected)
    cleaned = repair_protected_lines(work, cleaned, protected)

    # Mild local contrast recovery; do not flatten the original paper/background.
    cleaned = cv2.detailEnhance(cleaned, sigma_s=8, sigma_r=0.12)

    if scale < 1:
        cleaned = cv2.resize(cleaned, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_CUBIC)

    return Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))


uploaded = st.file_uploader("Déposez votre plan ici", type=["jpg", "jpeg", "png"])

if uploaded:
    original = Image.open(uploaded).convert("RGB")
    with st.spinner("Nettoyage architectural V3…"):
        cleaned = clean_plan(original)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original")
        st.image(original, width="stretch")
    with c2:
        st.subheader("Résultat V3")
        st.image(cleaned, width="stretch")

    buf = io.BytesIO()
    cleaned.save(buf, format="JPEG", quality=96, subsampling=0)
    st.download_button(
        "⬇️ Télécharger le plan nettoyé",
        data=buf.getvalue(),
        file_name="plan_nettoye_V3.jpg",
        mime="image/jpeg",
        width="stretch",
    )
