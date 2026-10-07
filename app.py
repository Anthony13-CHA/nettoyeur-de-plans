import base64, io, json, os, re
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from openai import OpenAI

st.set_page_config(page_title="Nettoyeur de plans", page_icon="🏠", layout="wide")
st.title("🏠 Nettoyeur de plans architecturaux")
st.caption("V5 — analyse visuelle des annotations et protection de la géométrie")

MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-6-luna")


def jpeg_data_url(pil, max_side=1800):
    img = pil.copy().convert("RGB")
    scale = min(1.0, max_side / max(img.size))
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), img.size


def ask_vision(pil):
    key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not key:
        st.error("La V5 nécessite une clé OpenAI dans les Secrets Streamlit : OPENAI_API_KEY.")
        st.stop()
    client = OpenAI(api_key=key)
    data_url, small_size = jpeg_data_url(pil)
    prompt = '''Tu analyses un plan architectural 2D. Nous voulons supprimer UNIQUEMENT les informations ajoutées au dessin : textes/noms de pièces, cotations, chiffres de dimensions, repères, flèches, légendes, cartouches et annotations colorées (magenta/rouge/etc.).

IMPORTANT : ne supprime jamais les murs, cloisons, contours, portes, fenêtres, escaliers, équipements sanitaires, meubles dessinés comme partie du plan ou autres traits architecturaux.

Retourne UNIQUEMENT un JSON valide de cette forme :
{"remove_boxes":[{"x":0,"y":0,"w":0,"h":0,"kind":"text|dimension|colored_annotation|legend|other"}]}

Les coordonnées sont NORMALISÉES entre 0 et 1000 par rapport à l'image. Regroupe chaque bloc de texte/cotation/annotation dans une boîte assez serrée. Pour une annotation qui chevauche un mur, la boîte doit couvrir l'annotation mais pas une grande zone inutile autour. N'inclus pas les murs ou portes comme boîtes à supprimer.'''
    r = client.responses.create(
        model=MODEL,
        input=[{"role":"user","content":[
            {"type":"input_text","text":prompt},
            {"type":"input_image","image_url":data_url,"detail":"high"}
        ]}
    ])
    text = r.output_text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("Le modèle n'a pas retourné le JSON attendu.")
    obj = json.loads(m.group(0))
    boxes=[]
    sw, sh = small_size
    for b in obj.get("remove_boxes", []):
        try:
            x=float(b["x"])/1000*sw; y=float(b["y"])/1000*sh
            w=float(b["w"])/1000*sw; h=float(b["h"])/1000*sh
            if w>2 and h>2: boxes.append((x,y,w,h,b.get("kind","other")))
        except Exception: pass
    return boxes, small_size


def reconstruct_in_box(img, mask):
    # Preserve long, straight architectural strokes that cross removal regions.
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,50,150)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(18,int(min(img.shape[:2])*0.025)),minLineLength=max(20,int(min(img.shape[:2])*0.04)),maxLineGap=8)
    protect=np.zeros(mask.shape,np.uint8)
    if lines is not None:
        for l in lines[:,0]:
            x1,y1,x2,y2=map(int,l)
            length=((x2-x1)**2+(y2-y1)**2)**0.5
            angle=abs(np.degrees(np.arctan2(y2-y1,x2-x1)))
            if length>=max(25,min(img.shape[:2])*0.035) and (angle<8 or angle>82):
                cv2.line(protect,(x1,y1),(x2,y2),255,2)
    # Only protect a stroke where it is surrounded by dark pixels on both sides.
    final_mask=mask.copy(); final_mask[protect>0]=0
    return final_mask


def clean_plan(pil):
    orig=np.array(pil.convert("RGB")); bgr=cv2.cvtColor(orig,cv2.COLOR_RGB2BGR)
    boxes,_=ask_vision(pil)
    h,w=bgr.shape[:2]
    sx=w/1800 if w>1800 else 1.0
    sy=h/1800 if h>1800 else 1.0
    # boxes were measured on the max-side resized image
    small_w=min(w,1800); small_h=int(h*small_w/w) if w>1800 else h
    fx=w/small_w; fy=h/small_h
    mask=np.zeros((h,w),np.uint8)
    for x,y,bw,bh,kind in boxes:
        X=int(x*fx); Y=int(y*fy); W=int(bw*fx); H=int(bh*fy)
        pad=max(2,int(min(w,h)*0.0025))
        # color annotations get a little more margin; text remains tight
        p=pad*2 if kind=="colored_annotation" else pad
        cv2.rectangle(mask,(max(0,X-p),max(0,Y-p)),(min(w-1,X+W+p),min(h-1,Y+H+p)),255,-1)
    # Remove saturated colored marks even if vision missed a small fragment.
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    color=cv2.inRange(hsv,np.array([125,45,60]),np.array([179,255,255]))
    color2=cv2.inRange(hsv,np.array([0,60,70]),np.array([15,255,255]))
    cm=cv2.morphologyEx(cv2.bitwise_or(color,color2),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    mask=cv2.bitwise_or(mask,cm)
    mask=cv2.dilate(mask,np.ones((3,3),np.uint8),1)
    safe=reconstruct_in_box(bgr,mask)
    # A modest NS inpaint, followed by local contrast normalization only where needed.
    cleaned=cv2.inpaint(bgr,safe,3,cv2.INPAINT_NS)
    return Image.fromarray(cv2.cvtColor(cleaned,cv2.COLOR_BGR2RGB)), boxes

uploaded=st.file_uploader("Déposez votre plan ici",type=["jpg","jpeg","png"])
if uploaded:
    original=Image.open(uploaded).convert("RGB")
    with st.spinner("Analyse visuelle du plan puis nettoyage…"):
        try:
            cleaned, boxes=clean_plan(original)
        except Exception as e:
            st.error(f"Erreur pendant l'analyse V5 : {e}")
            st.stop()
    st.caption(f"V5 : {len(boxes)} zones parasites détectées par l'analyse visuelle.")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Original"); st.image(original,width="stretch")
    with c2:
        st.subheader("Résultat"); st.image(cleaned,width="stretch")
    buf=io.BytesIO(); cleaned.save(buf,format="JPEG",quality=95)
    st.download_button("⬇️ Télécharger le plan nettoyé",data=buf.getvalue(),file_name="plan_nettoye_v5.jpg",mime="image/jpeg",width="stretch")
