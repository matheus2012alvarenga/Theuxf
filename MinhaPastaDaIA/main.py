import os
import json
import shutil
import sqlite3
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

DB_FILE = "chat_ia.db"
MODEL_NAME = "gemini-2.5-flash"  # atualizado para versão mais recente
UPLOAD_DIR = "temp_uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# BANCO DE DADOS
# ──────────────────────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS historico (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                mode TEXT DEFAULT 'tutor',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON historico(user_id)')
        conn.commit()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando servidor — verificando banco de dados...")
    init_db()
    yield
    logger.info("Encerrando servidor.")

app = FastAPI(title="Guardião IA", version="6.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    logger.critical("GEMINI_API_KEY não encontrada!")

client = genai.Client(api_key=api_key)

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS POR MODO
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "tutor": (
        "Você é o Guardião IA, um Tutor escolar super didático e inteligente para estudantes brasileiros.\n"
        "DIRETRIZES:\n"
        "- Use Markdown com **negrito**, *itálico*, listas e tabelas quando ajudar.\n"
        "- Para problemas de matemática ou ciências: resolva passo a passo com detalhes.\n"
        "- Para perguntas diretas: responda de forma clara e objetiva primeiro.\n"
        "- Use exemplos do cotidiano brasileiro quando possível.\n"
        "- Termine com uma dica de estudo ou curiosidade relacionada ao tema."
    ),
    "story": (
        "Você é o Guardião IA, um Contador de Histórias mágico e criativo para crianças brasileiras.\n"
        "DIRETRIZES:\n"
        "- Crie histórias envolventes com personagens memoráveis.\n"
        "- Use parágrafos curtos, diálogos dinâmicos e descrições vívidas.\n"
        "- Inclua uma mensagem ou lição positiva ao final.\n"
        "- Use emojis com moderação para tornar a leitura mais divertida.\n"
        "- Adapte o vocabulário para ser acessível a crianças de 6-14 anos.\n"
        "- Use **negrito** para destacar momentos importantes da narrativa."
    ),
    "debate": (
        "Você é o Guardião IA no Modo Debate — um argumentador equilibrado e perspicaz.\n"
        "DIRETRIZES:\n"
        "- Para qualquer tema, apresente SEMPRE os dois lados: prós e contras.\n"
        "- Use uma tabela Markdown comparativa quando possível.\n"
        "- Cite exemplos reais e dados quando relevantes.\n"
        "- Seja neutro: não tome partido, apenas apresente argumentos de forma justa.\n"
        "- Termine com: 'E você, o que acha? 🤔' para estimular o pensamento crítico.\n"
        "- Use seções bem definidas: **Argumentos A FAVOR** e **Argumentos CONTRA**."
    ),
    "code": (
        "Você é o Guardião IA no Modo Código — um professor de programação paciente e prático.\n"
        "DIRETRIZES:\n"
        "- Sempre use blocos de código Markdown com a linguagem especificada (ex: ```python).\n"
        "- Explique o código linha por linha quando solicitado.\n"
        "- Sugira boas práticas e evite antipadrões.\n"
        "- Para iniciantes: use analogias do mundo real para explicar conceitos.\n"
        "- Para avançados: discuta eficiência, complexidade e trade-offs.\n"
        "- Sempre teste mentalmente o código antes de responder.\n"
        "- Inclua exemplos de uso após cada bloco de código."
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# AUXILIARES
# ──────────────────────────────────────────────────────────────────────────────
def salvar(user_id: str, role: str, text: str, mode: str = "tutor"):
    try:
        with sqlite3.connect(DB_FILE, timeout=10.0) as conn:
            conn.execute(
                "INSERT INTO historico (user_id, role, text, mode) VALUES (?, ?, ?, ?)",
                (user_id, role, text, mode)
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"DB write error: {e}")

def recuperar(user_id: str, limite: int = 8) -> list:
    try:
        with sqlite3.connect(DB_FILE, timeout=10.0) as conn:
            rows = conn.execute(
                """SELECT role, text FROM (
                       SELECT id, role, text FROM historico
                       WHERE user_id = ? ORDER BY id DESC LIMIT ?
                   ) ORDER BY id ASC""",
                (user_id, limite)
            ).fetchall()
        return [
            types.Content(role=r, parts=[types.Part.from_text(text=t)])
            for r, t in rows
        ]
    except sqlite3.Error as e:
        logger.error(f"DB read error: {e}")
        return []


class ClearRequest(BaseModel):
    user_id: str

class QuizRequest(BaseModel):
    user_id: str

# ──────────────────────────────────────────────────────────────────────────────
# ROTAS
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(
    user_id: str = Form(...),
    message: str = Form(...),
    mode: str = Form("tutor"),
    file: Optional[UploadFile] = File(None),
):
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY ausente no servidor.")
    if not user_id.strip() or not message.strip():
        raise HTTPException(status_code=400, detail="user_id ou message vazio.")

    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["tutor"])
    historico = recuperar(user_id)
    salvar(user_id, "user", message, mode)

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.75,
        max_output_tokens=2048,
    )

    local_path = None
    gemini_file = None

    try:
        parts = []

        if file and file.filename:
            local_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(local_path, "wb") as buf:
                shutil.copyfileobj(file.file, buf)
            gemini_file = client.files.upload(file=local_path)
            parts.append(gemini_file)

        parts.append(types.Part.from_text(text=message))
        payload = historico + [types.Content(role="user", parts=parts)]

        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=payload,
            config=config,
        )

        text = resp.text or "Não consegui formular uma resposta."
        salvar(user_id, "model", text, mode)
        return {"response": text}

    except APIError as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(status_code=502, detail="Erro de comunicação com o Gemini.")
    except Exception as e:
        logger.error(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no servidor.")
    finally:
        if local_path and os.path.exists(local_path):
            try: os.remove(local_path)
            except: pass
        if gemini_file:
            try: client.files.delete(name=gemini_file.name)
            except: pass


@app.post("/api/clear")
async def clear(req: ClearRequest):
    try:
        with sqlite3.connect(DB_FILE, timeout=10.0) as conn:
            conn.execute("DELETE FROM historico WHERE user_id = ?", (req.user_id,))
            conn.commit()
        return {"status": "ok"}
    except sqlite3.Error:
        raise HTTPException(status_code=500, detail="Falha ao limpar histórico.")


@app.get("/api/export/{user_id}")
async def export(user_id: str):
    try:
        with sqlite3.connect(DB_FILE, timeout=10.0) as conn:
            rows = conn.execute(
                "SELECT role, text, mode, timestamp FROM historico WHERE user_id = ? ORDER BY id ASC",
                (user_id,)
            ).fetchall()

        if not rows:
            return PlainTextResponse("Nenhum histórico encontrado.", status_code=404)

        lines = [
            f"╔══════════════════════════════════════╗",
            f"║     GUARDIÃO IA — RELATÓRIO DE ESTUDO ║",
            f"╚══════════════════════════════════════╝",
            f"Aluno  : {user_id}",
            f"Data   : {rows[0][3].split(' ')[0] if rows else '—'}",
            f"",
        ]
        for role, text, mode, ts in rows:
            autor = "Você" if role == "user" else "Guardião IA"
            lines.append(f"[{ts}] [{mode.upper()}] {autor}:")
            lines.append(text)
            lines.append("─" * 50)
            lines.append("")

        content = "\n".join(lines)
        return PlainTextResponse(
            content,
            headers={"Content-Disposition": f"attachment; filename=aula_{user_id}.txt"}
        )
    except sqlite3.Error:
        raise HTTPException(status_code=500, detail="Erro ao exportar.")


@app.post("/api/quiz")
async def quiz(req: QuizRequest):
    historico = recuperar(req.user_id, limite=14)
    if len(historico) < 2:
        raise HTTPException(status_code=400, detail="Converse mais antes de gerar um desafio!")

    prompt = (
        "Com base nos tópicos desta conversa, crie um quiz com exatamente 3 perguntas de múltipla escolha.\n"
        "Responda SOMENTE com JSON válido no formato:\n"
        '{"quizzes":[{"question":"...","options":["A) ...","B) ...","C) ...","D) ..."],"answer":"A"}]}\n'
        "Sem texto adicional, sem markdown, apenas JSON puro."
    )

    config = types.GenerateContentConfig(
        system_instruction=(
            "Você é um avaliador acadêmico rigoroso. "
            "Responda APENAS com JSON válido, sem nenhum texto adicional."
        ),
        temperature=0.5,
        response_mime_type="application/json",
    )

    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=historico + [types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )],
            config=config,
        )
        raw = resp.text.strip()
        # Remove possíveis blocos markdown
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        logger.error(f"Quiz error: {e}")
        raise HTTPException(status_code=500, detail="Falha ao gerar quiz. Tente novamente.")


@app.get("/")
async def index():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
