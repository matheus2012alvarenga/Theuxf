import os
import sqlite3
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import APIError  # Captura erros nativos da API do Gemini
from dotenv import load_dotenv

# Configuração de Logs Profissionais
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="Guardião IA - Enterprise Backend",
    description="Backend robusto com segurança de arquivos, tratamento de erros e gestão de contexto.",
    version="2.0.0"
)

# Configuração Estrita de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, substitua pelo domínio real do seu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_FILE = "chat_ia.db"
MODEL_NAME = "gemini-2.5-flash"
MAX_FILE_SIZE = 5 * 1024 * 1024  # Limite de segurança: 5MB por imagem
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"]

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.critical("CRITICAL: A variável GEMINI_API_KEY não foi encontrada no ambiente!")
    api_key = ""

# Inicializa o cliente oficial
client = genai.Client(api_key=api_key)


def init_db():
    """Cria a tabela e garante índices para buscas rápidas por user_id."""
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
        # Índice para otimizar a velocidade de leitura do histórico quando a base crescer
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON historico(user_id)')
        conn.commit()

init_db()


def salvar_mensagem_no_banco(user_id: str, role: str, text: str):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO historico (user_id, role, text) VALUES (?, ?, ?)", (user_id, role, text))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Erro ao salvar no banco de dados: {str(e)}")


def recuperar_historico_do_banco(user_id: str) -> list:
    """Recupera e formata o histórico limitando o peso do contexto."""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            # Seleciona as últimas 6 interações de forma performática
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


@app.post("/api/chat")
async def chat_endpoint(
    user_id: str = Form(...),
    message: str = Form(...),
    mode: str = Form("tutor"),
    file: UploadFile = File(None)
):
    # Verificação de infraestrutura básica
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
    
    # Busca histórico prévio antes de registrar a entrada atual
    historico_previo = recuperar_historico_do_banco(user_id)
    salvar_mensagem_no_banco(user_id, "user", message)
    
    # Definição dos prompts com controle de comportamento estrito
    if mode == "story":
        system_prompt = (
            "Você é um Contador de Histórias mágico e lúdico para crianças.\n"
            "Use marcações Markdown como **negrito** para destacar nomes e ações importantes. "
            "Use parágrafos bem espaçados e abuse de emojis! Mantenha tudo seguro e livre de violência."
        )
    else:
        system_prompt = (
            "Você é um Tutor de IA super didático para estudantes menores de idade.\n"
            "Use títulos em Markdown (###) para separar os passos da explicação. "
            "Use listas (* ou 1.) para detalhar resoluções. Nunca dê a resposta de bandeja! "
            "Termine sempre fazendo uma pergunta reflexiva para testar o aprendizado do aluno."
        )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.6,  # Reduzido levemente para evitar respostas inventadas (alucinações)
        safety_settings=[
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
        ]
    )

    try:
        # Tratamento de arquivo com validações de segurança em camadas
        if file and file.filename:
            # Camada de Segurança 1: Tipo de Arquivo (MIME Type)
            if file.content_type not in ALLOWED_MIME_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Desculpe, formato inválido. Envie apenas imagens JPG, PNG ou WEBP."
                )

            file_bytes = await file.read()
            
            # Camada de Segurança 2: Tamanho máximo do arquivo
            if len(file_bytes) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="A imagem enviada é muito pesada! Escolha uma foto de até 5MB."
                )
                
            if not file_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, 
                    detail="Desculpe, o arquivo enviado está vazio ou corrompido."
                )

            image_part = types.Part.from_bytes(data=file_bytes, mime_type=file.content_type)
            text_part = types.Part.from_text(text=message)
            
            conteudo_atual = types.Content(role="user", parts=[image_part, text_part])
            payload_contents = historico_previo + [conteudo_atual]
        else:
            conteudo_atual = types.Content(role="user", parts=[types.Part.from_text(text=message)])
            payload_contents = historico_previo + [conteudo_atual]
        
        # Envio dos dados higienizados para a API do Gemini
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=payload_contents,
            config=config
        )
        
        resposta_texto = response.text if response.text else "Não consegui formular uma resposta. Pode reescrever de outra forma?"
        
        salvar_mensagem_no_banco(user_id, "model", resposta_texto)
        return {"response": resposta_texto}
        
    except APIError as api_err:
        # Tratamento específico para falhas na API do Google Gemini (Ex: Cota estourada, erro nos servidores deles)
        logger.error(f"Erro nativo da API do Gemini: {str(api_err)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="A IA está processando muitas requisições no momento. Aguarde um minutinho e envie de novo!"
        )
    except HTTPException as http_err:
        # Repassa os erros de validação que nós criamos nas camadas acima
        raise http_err
    except Exception as e:
        # Fallback genérico para capturar qualquer outro imprevisto de runtime
        logger.error(f"Erro inesperado no servidor: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Desculpe, ocorreu um erro interno ao processar sua mensagem. Tente novamente."
        )


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