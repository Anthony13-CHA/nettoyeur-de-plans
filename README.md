# Nettoyeur de plans — V5

V5 utilise une analyse visuelle via l'API OpenAI pour localiser les textes, cotations et annotations avant une suppression locale prudente.

## Configuration Streamlit

Dans **Manage app → Settings → Secrets**, ajouter :

```toml
OPENAI_API_KEY = "votre_cle_api"
```

Optionnel :

```toml
OPENAI_VISION_MODEL = "gpt-6-luna"
```

La clé API ne doit jamais être mise dans GitHub ni dans `app.py`.
