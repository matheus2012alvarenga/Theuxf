import os
import shutil
import sqlite3
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

# Configuração de Logs Profissionais
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

DB_FILE = "chat_ia.db"
MODEL_NAME = "gemini-2.5-flash"
UPLOAD_DIR = "temp_uploads"

# Garante que o diretório temporário exista
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# GERENCIAMENTO DE CICLO DE VIDA (LIFESPAN) & BANCO DE DADOS
# -----------------------------------------------------------------------------
def init_db():
    """Cria a tabela e garante índices para buscas rápidas."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historico (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                role TEXT,
                text TEXT
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON historico(user_id)')
        conn.commit()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia a inicialização e encerramento seguro do servidor."""
    logger.info("Iniciando o servidor e verificando banco de dados...")
    init_db()
    yield
    logger.info("Encerrando o servidor...")

# Inicialização do FastAPI com Lifespan moderno
app = FastAPI(
    title="Guardião IA - Enterprise Backend",
    description="Backend robusto com suporte multimodal total (Texto, Imagem, Vídeo, Áudio e Arquivos).",
    version="3.0.0",
    lifespan=lifespan
)

# Configuração Estrita de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.critical("CRITICAL: A variável GEMINI_API_KEY não foi encontrada no ambiente!")
    api_key = ""

# Inicializa o cliente oficial do Gemini
client = genai.Client(api_key=api_key)

# -----------------------------------------------------------------------------
# FUNÇÕES AUXILIARES DO BANCO DE DADOS
# -----------------------------------------------------------------------------
def salvar_mensagem_no_banco(user_id: str, role: str, text: str):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO historico (user_id, role, text) VALUES (?, ?, ?)", (user_id, role, text))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Erro ao salvar no banco de dados: {str(e)}")

def recuperar_historico_do_banco(user_id: str) -> list:
    """Recupera as últimas interações mantendo a estrutura oficial do SDK."""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, text FROM (
                    SELECT id, role, text FROM historico 
                    WHERE user_id = ? 
                    ORDER BY id DESC LIMIT 6
                ) ORDER BY id ASC
            """, (user_id,))
            rows = cursor.fetchall()
        
        contents_list = []
        for role, text in rows:
            contents_list.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=text)]
                )
            )
        return contents_list
    except sqlite3.Error as e:
        logger.error(f"Erro ao ler do banco de dados: {str(e)}")
        return []

class ClearRequest(BaseModel):
    user_id: str

# -----------------------------------------------------------------------------
# ROTA PRINCIPAL DO CHAT (SUPORTE A TEXTO E MULTIMÍDIA)
# -----------------------------------------------------------------------------
@app.post("/api/chat")
async def chat_endpoint(
    user_id: str = Form(...),
    message: str = Form(...),
    mode: str = Form("tutor"),
    file: Optional[UploadFile] = File(None)
):
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Desculpe, a inteligência artificial está temporariamente desligada (Chave ausente)."
        )
        
    if not user_id.strip() or not message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Desculpe, ocorreu um erro: O ID do usuário e a mensagem não podem estar vazios."
        )
    
    # Busca o histórico prévio e salva a nova entrada do usuário
    historico_previo = recuperar_historico_do_banco(user_id)
    salvar_mensagem_no_banco(user_id, "user", message)
    
    # Definição das diretrizes de comportamento da IA
    if mode == "story":
        system_prompt = (
            "Você é um Contador de Histórias mágico e lúdico para crianças.\n"
            "Use marcações Markdown como **negrito** para destacar nomes e ações importantes. "
            "Use parágrafos bem espaçados e abuse de emojis! Mantenha tudo seguro e livre de violência."
        )
    else:
        system_prompt = (
            "Você é um Tutor de IA super didático e inteligente.\n"
            "DIRETRIZ DE RESPOSTA FATO/RESULTADO: Se o usuário pedir um dado direto, uma curiosidade, "
            "uma informação factual do mundo real ou o resultado de uma busca, ENTREGUE O RESULTADO IMEDIATAMENTE e de forma direta.\n"
            "DIRETRIZ DE RESOLUÇÃO DE EXERCÍCIOS: Apenas quando o aluno enviar problemas escolares, equações ou tarefas "
            "para resolver, aja de forma guiada passo a passo usando títulos (###) e listas (* ou 1.) sem dar a resposta de bandeja de início."
        )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.7,
    )

    local_file_path = None
    uploaded_gemini_file = None

    try:
        # Inicializa a lista de partes do bloco de conteúdo atual do usuário
        user_parts = []

        # Se houver um arquivo (Imagem, Vídeo, Áudio, PDF, etc.)
        if file and file.filename:
            local_file_path = os.path.join(UPLOAD_DIR, file.filename)
            
            # Salva o arquivo temporariamente no servidor local
            with open(local_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"Arquivo recebido e salvo localmente: {local_file_path}")

            # Envia o arquivo para a API de Arquivos do Gemini (Suporta mídias grandes como vídeos)
            logger.info("Enviando arquivo para a infraestrutura do Gemini...")
            uploaded_gemini_file = client.files.upload(file=local_file_path)
            logger.info(f"Arquivo carregado com sucesso no Gemini. URI: {uploaded_gemini_file.uri}")
            
            # Adiciona o objeto do arquivo nas partes que serão processadas
            user_parts.append(uploaded_gemini_file)

        # Adiciona a mensagem de texto do usuário
        user_parts.append(types.Part.from_text(text=message))
        
        conteudo_atual = types.Content(role="user", parts=user_parts)
        payload_contents = historico_previo + [conteudo_atual]
        
        # Envia todo o contexto montado para o modelo
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=payload_contents,
            config=config
        )
        
        resposta_texto = response.text if response.text else "Não consegui formular uma resposta. Pode reescrever de outra forma?"
        
        salvar_mensagem_no_banco(user_id, "model", resposta_texto)
        return {"response": resposta_texto}
        
    except APIError as api_err:
        logger.error(f"Erro nativo da API do Gemini: {str(api_err)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="A IA está processando muitas requisições ou o arquivo enviado é incompatível. Tente novamente."
        )
    except Exception as e:
        logger.error(f"Erro inesperado no servidor: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Desculpe, ocorreu um erro interno ao processar sua mensagem."
        )
    finally:
        # Limpeza absoluta de espaço em disco no servidor local após o processamento
        if local_file_path and os.path.exists(local_file_path):
            try:
                os.remove(local_file_path)
                logger.info(f"Espaço limpo: Arquivo temporário {local_file_path} removido.")
            except Exception as clean_err:
                logger.error(f"Falha ao apagar arquivo temporário: {str(clean_err)}")

# -----------------------------------------------------------------------------
# OUTRAS ROTAS DO SISTEMA
# -----------------------------------------------------------------------------
@app.post("/api/clear")
async def clear_chat(request: ClearRequest):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM historico WHERE user_id = ?", (request.user_id,))
            conn.commit()
        return {"status": "success"}
    except sqlite3.Error as e:
        logger.error(f"Erro ao limpar banco: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Desculpe, falha ao tentar limpar sua conversa no banco de dados."
        )

@app.get("/")
async def read_index():
    return FileResponse('index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
