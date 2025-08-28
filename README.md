# 📧 AutoU - Classificador de E-mails

Aplicação **FastAPI + OpenAI** que classifica e-mails como **Produtivo** (requer ação) ou **Improdutivo** (mensagem social/agradecimento/spam) e sugere uma resposta automática em PT-BR.

---

## 🚀 Requisitos

- Python **3.11+**
- `pip` atualizado
- Chave da OpenAI: crie em https://platform.openai.com/account/api-keys

---

## ⚙️ Executar localmente

### 1) Clonar o repositório
```bash
git clone https://github.com/seu-usuario/Case-AutoU.git
cd Case-AutoU
``` 

### 2) Criar e ativar a venv
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Instalar dependências
```bash
pip install -r requirements.txt

```

### 4) Configurar variáveis de ambiente
```bash
cat > .env << 'EOF'
OPENAI_API_KEY=sk-proj-sua_chave_aqui
OPENAI_MODEL=gpt-4o-mini
EOF

```

### 5) Rodar o servidor
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload


```

🧪 Como testar

Cole o texto de um e-mail no formulário ou envie um arquivo .txt/.pdf.
A app retorna: Categoria, Fonte (openai) e Resposta sugerida.

Exemplos:
```bash
Produtivo

Assunto: Erro de login
Bom dia, não consigo acessar o sistema. Após digitar a senha aparece “acesso negado”.
Podem verificar?
```
```bash
Improdutivo

Assunto: Agradecimento
Boa tarde, apenas para agradecer pelo atendimento de ontem. Parabéns à equipe!
```