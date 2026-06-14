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

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

DB_FILE = "chat_ia.db"
MODEL_NAME = "gemini-3.5-flash"
UPLOAD_DIR = "temp_uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# BANCO DE DADOS & LIFESPAN
# -----------------------------------------------------------------------------
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historico (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                role TEXT,
                text TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON historico(user_id)')
        conn.commit()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando o servidor e verificando banco de dados...")
    init_db()
    yield
    logger.info("Encerrando o servidor...")

app = FastAPI(title="Guardião IA", version="5.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.critical("A variável GEMINI_API_KEY não foi encontrada!")
    api_key = ""

client = genai.Client(api_key=api_key)

# -----------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# -----------------------------------------------------------------------------
def salvar_mensagem_no_banco(user_id: str, role: str, text: str):
    try:
        with sqlite3.connect(DB_FILE, timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO historico (user_id, role, text) VALUES (?, ?, ?)", (user_id, role, text))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Erro ao salvar no banco: {str(e)}")

def recuperar_historico_do_banco(user_id: str, limite: int = 6) -> list:
    try:
        with sqlite3.connect(DB_FILE, timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, text FROM (
                    SELECT id, role, text FROM historico 
                    WHERE user_id = ? 
                    ORDER BY id DESC LIMIT ?
                ) ORDER BY id ASC
            """, (user_id, limite))
            rows = cursor.fetchall()
        
        contents_list = []
        for role, text in rows:
            contents_list.append(
                types.Content(role=role, parts=[types.Part.from_text(text=text)])
            )
        return contents_list
    except sqlite3.Error as e:
        logger.error(f"Erro ao ler do banco: {str(e)}")
        return []

class ClearRequest(BaseModel):
    user_id: str

class QuizRequest(BaseModel):
    user_id: str

# -----------------------------------------------------------------------------
# ROTAS
# -----------------------------------------------------------------------------
@app.post("/api/chat")
async def chat_endpoint(
    user_id: str = Form(...),
    message: str = Form(...),
    mode: str = Form("tutor"),
    file: Optional[UploadFile] = File(None)
):
    if not api_key:
        raise HTTPException(status_code=500, detail="Chave API do Gemini ausente no servidor.")
        
    if not user_id.strip() or not message.strip():
        raise HTTPException(status_code=400, detail="ID de usuário ou mensagem vazios.")
    
    historico_previo = recuperar_historico_do_banco(user_id)
    salvar_mensagem_no_banco(user_id, "user", message)
    
    if mode == "story":
        system_prompt = (
            "Você é um Contador de Histórias mágico e lúdico para crianças.\n"
            "Use marcações Markdown como **negrito** para destacar nomes importantes. "
            "Use parágrafos curtos e emojis! Mantenha tudo seguro."
        )
    else:
        system_prompt = (
            "Você é um Tutor de IA super didático e inteligente.\n"
            "DIRETRIZ DE RESPOSTA FATO/RESULTADO: Se o usuário pedir um dado direto, entregue IMEDIATAMENTE em formato Markdown.\n"
            "DIRETRIZ DE RESOLUÇÃO: Aja de forma guiada passo a passo usando títulos e listas estruturadas."
        )

    config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.7)
    local_file_path = None
    uploaded_gemini_file = None

    try:
        user_parts = []

        if file and file.filename:
            local_file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(local_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_gemini_file = client.files.upload(file=local_file_path)
            user_parts.append(uploaded_gemini_file)

        user_parts.append(types.Part.from_text(text=message))
        conteudo_atual = types.Content(role="user", parts=user_parts)
        payload_contents = historico_previo + [conteudo_atual]
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=payload_contents,
            config=config
        )
        
        resposta_texto = response.text if response.text else "Não consegui formular uma resposta."
        salvar_mensagem_no_banco(user_id, "model", resposta_texto)
        return {"response": resposta_texto}
        
    except APIError as api_err:
        logger.error(f"Erro na API do Gemini: {str(api_err)}")
        raise HTTPException(status_code=502, detail="Erro de comunicação com o Gemini.")
    except Exception as e:
        logger.error(f"Erro interno: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno no servidor.")
    finally:
        if local_file_path and os.path.exists(local_file_path):
            try: os.remove(local_file_path)
            except: pass
        if uploaded_gemini_file:
            try: client.files.delete(name=uploaded_gemini_file.name)
            except: pass

@app.post("/api/clear")
async def clear_chat(request: ClearRequest):
    try:
        with sqlite3.connect(DB_FILE, timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM historico WHERE user_id = ?", (request.user_id,))
            conn.commit()
        return {"status": "success"}
    except sqlite3.Error:
        raise HTTPException(status_code=500, detail="Falha ao limpar o histórico.")

@app.get("/api/export/{user_id}")
async def export_chat(user_id: str):
    try:
        with sqlite3.connect(DB_FILE, timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT role, text, timestamp FROM historico WHERE user_id = ? ORDER BY id ASC", (user_id,))
            rows = cursor.fetchall()
            
        if not rows:
            return PlainTextResponse("Nenhum histórico encontrado.", status_code=404)

        conteudo = f"--- RELATÓRIO DE ESTUDOS: GUARDIÃO IA ---\nID Aluno: {user_id}\n\n"
        for role, text, timestamp in rows:
            autor = "Aluno" if role == "user" else "Tutor IA"
            conteudo += f"[{timestamp}] {autor}:\n{text}\n\n{'-'*40}\n\n"

        return PlainTextResponse(conteudo, headers={"Content-Disposition": f"attachment; filename=aula_{user_id}.txt"})
    except sqlite3.Error:
        raise HTTPException(status_code=500, detail="Erro ao exportar.")

# SISTEMA NOVO: Geração automática de Quizzes baseados na conversa do usuário
@app.post("/api/quiz")
async def generate_quiz(request: QuizRequest):
    historico = recuperar_historico_do_banco(request.user_id, limite=12)
    if not historico or len(historico) < 2:
        raise HTTPException(status_code=400, detail="Converse um pouco com a IA antes de gerar um desafio!")

    prompt_quiz = (
        "Com base nos tópicos que conversamos nesta sessão, gere um quiz estrito com exatamente 3 perguntas de múltipla escolha para testar o conhecimento do aluno.\n"
        "Você DEVE responder APENAS e estritamente com um JSON válido no seguinte formato de objeto:\n"
        '{"quizzes": [{"question": "Texto da pergunta", "options": ["A) Opção 1", "B) Opção 2", "C) Opção 3", "D) Opção 4"], "answer": "A"}]}'
    )
    
    config = types.GenerateContentConfig(
        system_instruction="Você é um avaliador acadêmico rigoroso que gera outputs estruturados puramente em JSON.",
        temperature=0.6,
        response_mime_type="application/json"
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=historico + [types.Content(role="user", parts=[types.Part.from_text(text=prompt_quiz)])],
            config=config
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Erro ao construir Quiz: {str(e)}")
        raise HTTPException(status_code=500, detail="O Gemini falhou ao estruturar o quiz. Tente novamente.")

@app.get("/")
async def read_index():
    return FileResponse('index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
