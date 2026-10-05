# Nettoyeur de plans — V2

Version améliorée du prototype Streamlit.

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fonctionnement

Le moteur combine normalisation du fond, détection prudente des éléments textuels, détection des annotations colorées et protection des longues lignes architecturales avant reconstruction locale.

Cette V2 ne nécessite pas Tesseract ni de binaire système supplémentaire, afin de rester simple à déployer sur Streamlit Community Cloud.

## Déploiement Streamlit

Remplacer `app.py`, `requirements.txt` et `README.md` dans le dépôt GitHub relié à Streamlit Community Cloud.
