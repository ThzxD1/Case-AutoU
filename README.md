# AutoU - Classificador de Emails (Demo)

App FastAPI que classifica emails (Produtivo/Improdutivo) e sugere resposta.

## Rodar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HUGGINGFACE_API_TOKEN=seu_token  # opcional (zero-shot)
uvicorn app.main:app --reload --port 8000
# abrir http://127.0.0.1:8000
