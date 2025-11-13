# ===========================================
# Instalações necessárias (coloque no terminal):
# pip install streamlit langchain-google-genai langchain-chroma chromadb langchain-community sentence-transformers pypdf pyyaml chonk
# ===========================================

import streamlit as st
import os
import glob
import shutil
import yaml
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ==============================
# Configurações iniciais
# ==============================
st.set_page_config(page_title="Consulta", layout="wide")

# ==============================
# Carregar chave do Gemini
# ==============================
try:
    with open("key.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    API_KEY = config["KEY"]
except Exception:
    st.error("❌ Erro: arquivo key.yaml não encontrado ou mal formatado.")
    st.stop()

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0.3
)

# ==============================
# Configurações de diretório
# ==============================
CHROMA_PATH = "./chroma"
PDF_FOLDER = "./manuais"

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# ==============================
# Função: indexar PDFs com CHONK
# ==============================
def indexar_pdfs():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    documentos = []

    # 🔹 Divisor de texto oficial do LangChain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # Tamanho máximo do pedaço
        chunk_overlap=100,   # Sobreposição entre pedaços (mantém contexto)
        separators=["\n\n", "\n", ".", "?", "!", " "]  # Corta de forma natural
    )

    for arquivo in glob.glob(os.path.join(PDF_FOLDER, "*.pdf")):
        loader = PyPDFLoader(arquivo)
        pages = loader.load()

        for page in pages:
            chunks = text_splitter.split_text(page.page_content)
            for i, chunk in enumerate(chunks):
                novo_doc = page.copy()
                novo_doc.page_content = chunk
                novo_doc.metadata["manual"] = os.path.basename(arquivo).replace(".pdf", "")
                novo_doc.metadata["chunk"] = i + 1
                documentos.append(novo_doc)

    if not documentos:
        st.warning("⚠️ Nenhum PDF encontrado na pasta ./manuais/")
        return None

    vectorstore = Chroma.from_documents(
        documents=documentos,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="manuais-empresa"
    )
    return len(documentos)


# ==============================
# Função: consultar PDFs
# ==============================
def consultar(pergunta):
    if not os.path.exists(CHROMA_PATH):
        return "⚠️ Execute a indexação primeiro!"

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="manuais-empresa"
    )

    results = vectorstore.similarity_search_with_score(pergunta, k=3)
    if not results:
        return "❌ Nenhuma informação relevante encontrada."

    contexto = ""
    for doc, score in results:
        manual = doc.metadata.get("manual", "Desconhecido")
        chunk_id = doc.metadata.get("chunk", "?")
        similaridade = 1 - score
        contexto += f"📘 Manual **{manual}** (parte {chunk_id}, relevância: {similaridade:.2f})\n\n"
        contexto += f"{doc.page_content[:700]}\n\n"

    prompt = f"""
Você é um assistente da empresa. Com base nos manuais abaixo, responda de forma objetiva e prática:

{contexto}

Pergunta: {pergunta}

Resposta baseada nos manuais:
"""
    resposta = gemini.invoke(prompt).content
    return resposta


# ==============================
# INTERFACE STREAMLIT
# ==============================
st.title("Manuais da Empresa")

# --- Aba lateral ---
with st.sidebar:
    st.header("⚙️ Configurações")
    if st.button("Indexar PDFs Novamente"):
        with st.spinner("Indexando PDFs e dividindo em pedaços..."):
            qtd = indexar_pdfs()
            if qtd:
                st.success(f"✅ {qtd} pedaços indexados com sucesso!")
            else:
                st.warning("Nenhum documento indexado.")
        time.sleep(1)

    st.markdown("---")
    if os.path.exists(PDF_FOLDER):
        pdfs = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
        if pdfs:
            st.write("📂 Arquivos detectados:")
            for pdf in pdfs:
                st.write(f"- {os.path.basename(pdf)}")
        else:
            st.warning("Nenhum PDF encontrado.")

# --- Área principal ---
pergunta = st.text_area("Digite sua pergunta:", height=100)

if st.button("🔍 Consultar"):
    if pergunta.strip() == "":
        st.warning("Digite uma pergunta primeiro.")
    else:
        with st.spinner("Buscando resposta nos manuais..."):
            resposta = consultar(pergunta)
        st.markdown("### 🧠 Resposta:")
        st.write(resposta)

# --- Rodapé ---
st.markdown("---")
st.caption("Desenvolvido por Guilherme Gabriel Santana - 2025")
