from pathlib import Path
import os, re, io, sys, traceback, json
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pypdf import PdfReader
from dotenv import load_dotenv

from openai import OpenAI  

app = FastAPI(title="AutoU - Classificador de Emails (OpenAI)")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
TEMPLATES_DIR = PROJECT_DIR / "templates"
STATIC_DIR = PROJECT_DIR / "static"

load_dotenv(PROJECT_DIR / ".env")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.get("/ping")
def ping():
    return {"ok": True}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY não definido. Configure seu .env.", file=sys.stderr)

client = OpenAI(api_key=OPENAI_API_KEY)  

CANDIDATE_LABELS = ["Produtivo", "Improdutivo"]

GRATITUDE_PATTERNS = [
    r"\bobrigad[oa]s?\b",
    r"\bmuito\s+obrigad[oa]s?\b",
    r"\bagradeç[oa]\b",
    r"\bagradecemos\b",
    r"\bvaleu\b",
    r"\bgratid[aã]o\b",
    r"\bparab(é|e)ns?\b",
    r"\bfeliz\s+(anivers[áa]rio|natal|ano\s+novo|dia)\b",
    r"\bboa\s+(semana|tarde|noite|manhã)\b",
    r"\bexcelente\s+semana\b",
    r"\bsucesso[s]?\b"
]

ACTIONABLE_HINTS = [
    r"\b(poderia|pode|podes|consegue|conseguir|tem como|favor|gentileza)\b",
    r"\b(dúvida|duvidas|dúvidas)\b",
    r"\b(como|quando|onde|qual|quais|quanto|por que|por quê)\b",
    r"\b(chamado|ticket|protocolo|case|caso|#\d{2,})\b",
    r"\b(segue|anexo|em anexo|envio|enviando)\b",
    r"\b(preciso|necessito|solicito|solicitamos|requisito)\b",
    r"\b(aprovar|aprovação|liberar|liberação|autorizar|autorização)\b",
    r"\b(agendar|agendamento|reuni[aã]o|meeting)\b",
    r"\b(erro|falha|bug|problema|incidente)\b",
    r"\b(urgente|prioridade|prazo|deadline)\b",
    r"\b(confirma(r|ção)|verificar|atualizar|status)\b"
]

def is_gratitude_only(text: str) -> bool:
    """
    True se houver forte sinal de agradecimento/felicitação
    e NÃO houver pedido, pergunta ou pista de ação.
    """
    if not text:
        return False
    t = text.lower()
    has_gratitude = any(re.search(p, t) for p in GRATITUDE_PATTERNS)
    has_action = any(re.search(p, t) for p in ACTIONABLE_HINTS) or "?" in t
    return has_gratitude and not has_action

def short_improd_reply(text: str) -> str:
    return "Agradecemos a mensagem! Seguimos à disposição caso precise de algo."
# --- fim heurística ---

def extract_text_from_upload(upload: UploadFile) -> str:
    """Extrai texto de .pdf ou trata bytes como texto (.txt/.eml)."""
    if upload is None:
        return ""
    filename = (upload.filename or "").lower().strip()
    data = upload.file.read() if upload.file else b""
    try:
        upload.file.seek(0)
    except Exception:
        pass

    if filename.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            return "\n".join(parts).strip()
        except Exception:
            pass  

    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc, errors="ignore")
        except Exception:
            continue
    return ""

def preprocess(text: str) -> str:
    
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text

def classify_and_reply_openai(text: str) -> dict:
    from openai import OpenAI
    from openai import AuthenticationError, APIError, RateLimitError

    if not OPENAI_API_KEY:
        return {
            "category": "Indefinido",
            "reply": "OPENAI_API_KEY não configurada.",
            "source": "openai",
            "rule_applied": "config:error"
        }

    # Primeiro, tente a heurística local para gratidão-only
    if is_gratitude_only(text):
        return {
            "category": "Improdutivo",
            "reply": short_improd_reply(text),
            "source": "heuristic",
            "rule_applied": "heurística:gratidão-only"
        }

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        system_prompt = (
            "Você é um classificador de emails. Política de negócio:\n"
            "- Classifique apenas como 'Produtivo' OU 'Improdutivo'.\n"
            "- 'Improdutivo' inclui mensagens de FELICITAÇÕES/AGRADECIMENTO que não contenham pedido, pergunta ou ação clara.\n"
            "- 'Produtivo' quando houver solicitação, pergunta, follow-up de chamado, envio de documentação, prazos, ou qualquer ação requerida.\n"
            "- Responda SOMENTE em JSON: {\"category\":\"Produtivo|Improdutivo\",\"reply\":\"<texto curto PT-BR>\"}.\n"
            "Exemplos:\n"
            "1) \"Obrigado pelo excelente atendimento!\" -> {\"category\":\"Improdutivo\",\"reply\":\"Agradecemos o retorno! Ficamos à disposição quando precisar.\"}\n"
            "2) \"Obrigado. Pode confirmar se o chamado #123 foi encerrado?\" -> {\"category\":\"Produtivo\",\"reply\":\"Claro! Vou verificar o status do chamado #123 e retorno em seguida.\"}\n"
            "3) \"Parabéns pelo serviço.\" -> {\"category\":\"Improdutivo\",\"reply\":\"Muito obrigado! Seguimos à disposição.\"}\n"
        )
        user_prompt = (
            f"Texto do email:\n'''{text}'''\n\n"
            "Rótulos válidos: [\"Produtivo\",\"Improdutivo\"].\n"
            "Devolva JSON: {\"category\":\"Produtivo|Improdutivo\",\"reply\":\"<texto curto>\"}"
        )

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content if resp.choices else "{}"

        # Robustez contra formatação
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Tentativa de "extrair" JSON bruto se vier com texto ao redor
            m = re.search(r"\{.*\}", content, flags=re.DOTALL)
            data = json.loads(m.group(0)) if m else {}

        category = data.get("category")
        reply = data.get("reply") or "Olá! Recebemos sua mensagem. Poderia detalhar melhor o objetivo?"
        if category not in CANDIDATE_LABELS:
            category = "Indefinido"

        # OVERRIDE pós-modelo: se por acaso o modelo ainda classificou errado
        rule_applied = "modelo"
        if is_gratitude_only(text):
            category = "Improdutivo"
            reply = short_improd_reply(text)
            rule_applied = "modelo+override:gratidão-only"

        return {"category": category, "reply": reply, "source": "openai", "rule_applied": rule_applied}

    except AuthenticationError:
        return {
            "category": "Indefinido",
            "reply": "Erro de autenticação na OpenAI (verifique OPENAI_API_KEY).",
            "source": "openai",
            "rule_applied": "api:error:auth"
        }
    except RateLimitError:
        return {
            "category": "Indefinido",
            "reply": "Limite de uso da OpenAI atingido. Tente novamente em instantes.",
            "source": "openai",
            "rule_applied": "api:error:rate"
        }
    except APIError as e:
        return {
            "category": "Indefinido",
            "reply": f"Erro na API da OpenAI: {e}",
            "source": "openai",
            "rule_applied": "api:error"
        }
    except Exception as e:
        print("[OPENAI EXC]", repr(e), file=sys.stderr)
        return {
            "category": "Indefinido",
            "reply": "Não foi possível processar no momento.",
            "source": "openai",
            "rule_applied": "unexpected:error"
        }



@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/process", response_class=HTMLResponse)
async def process(
    request: Request,
    email_text: Optional[str] = Form(default=""),
    file: Optional[UploadFile] = File(default=None),
):
    try:
        raw_text = (email_text or "").strip()
        if file and (file.filename or "").strip():
            raw_text = extract_text_from_upload(file)

        if not raw_text:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "result": {
                    "category": "Indefinido",
                    "reply": "Nenhum texto foi enviado. Cole o conteúdo do e-mail ou envie um arquivo .txt/.pdf.",
                    "chars": 0,
                    "preview": "",
                    "source": "nenhum",
                }
            })

        pre = preprocess(raw_text)

        out = classify_and_reply_openai(pre)
        category = out["category"]
        reply = out["reply"]
        source = out.get("source", "openai")

        print(f"[classify] category={category} source={source}", file=sys.stderr)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {
                "category": category,
                "reply": reply,
                "chars": len(pre),
                "preview": (pre[:600] + ("..." if len(pre) > 600 else "")),
                "source": source,
            }
        })
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return HTMLResponse(
            "<h3>Erro ao processar</h3><p>Verifique o console para detalhes.</p>",
            status_code=500
        )
