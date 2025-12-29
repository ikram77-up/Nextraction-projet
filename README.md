Installation
python -m venv venv

installation des dependences 
pip install -r requirements.txt

Data (corpurs.json)
{
    "startup_id": "devops_toolkit",
    "doc_id": "deck",
    "section": "Produit",
    "chunk_id": "devops_toolkit_product_1",
    "text": "La solution automatise les pipelines CI/CD et améliore la collaboration entre les équipes de développement et d’exploitation."
  },

  --lancement de API
  depuis le dossier racine de projet nextraction 
  uvicorn src.api:app --reload

  Puis on ouvre dans navigateur a travers ce lien 
  http://127.0.0.1:8000/docs

--pipeline de recherche 
Indexation

Tokenisation

Embeddings sémantiques

Recherche

BM25 (lexicale)

Similarité cosinus (sémantique)

Fusion

Reciprocal Rank Fusion (RRF)

Réordonnancement

Re-ranking via modèle léger

Évaluation

L’évaluation se fait via :
evaluate.py