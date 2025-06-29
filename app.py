"""
app.py – Chat sobre PDF usando Together AI (LLMs open-source)
Requisitos (requirements.txt):
streamlit
langchain-community
sentence-transformers
faiss-cpu
pypdf
tiktoken
together-ai
"""

import os, time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import together
from together.error import RateLimitError

# ─────────────────────────────────────────────────────────
# 1) Configuração da Together AI
#    Salve no Streamlit Cloud → Settings → Secrets:
#    TOGETHER_API_KEY = "sua_chave"
# ─────────────────────────────────────────────────────────
together.api_key = st.secrets["TOGETHER_API_KEY"]
MODEL_NAME   = "mistralai/Mistral-7B-Instruct-v0.2"   # escolha outro na doc se quiser
MAX_OUT      = 256    # tokens de saída
K_CHUNKS     = 2      # nº de trechos que enviamos ao LLM

def ask_together(prompt: str) -> str:
    """Envia prompt ao Together AI e devolve texto."""
    resp = together.Complete.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=MAX_OUT,
        temperature=0.1,
        stop=["</s>"],
    )
    return resp["choices"][0]["text"].strip()

# ─────────────────────────────────────────────────────────
# 2) Interface Streamlit
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chat (Together AI)", page_icon="📄🤖")
st.title("📄💬 Assistente PDF – Together AI")

pdf_file = st.file_uploader("💾 Envie um PDF", type="pdf")

if pdf_file:
    # ── Salvar arquivo em /tmp
    tmp_path = f"/tmp/{pdf_file.name}"
    with open(tmp_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    # ── Carregar e dividir texto
    docs  = PyPDFLoader(tmp_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    # ── Embeddings + índice FAISS
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectordb  = FAISS.from_documents(chunks, emb)
    retriever = vectordb.as_retriever(search_kwargs={"k": K_CHUNKS})

    st.success("✅ PDF processado! Pergunte algo abaixo.")
    query = st.text_input("Pergunta")

    if query:
        with st.spinner("Gerando resposta…"):
            # Montar contexto
            ctx_docs = retriever.get_relevant_documents(query)
            context  = "\n\n".join(d.page_content for d in ctx_docs)

            prompt = (
                "Você é um assistente que responde apenas a partir do contexto fornecido.\n"
                "Responda em português, de forma clara e objetiva.\n\n"
                f"Contexto:\n{context}\n\nPergunta: {query}\nResposta:"
            )

            try:
                answer = ask_together(prompt)
                st.markdown("### 🤖 Resposta")
                st.write(answer)

                # Mostrar fontes
                st.markdown("### 📚 Fontes")
                for i, d in enumerate(ctx_docs, 1):
                    src = d.metadata.get("source", pdf_file.name)
                    page = d.metadata.get("page", None)
                    lbl = f"{i}. {src}"
                    if page is not None:
                        lbl += f" – página {page+1}"
                    st.write(lbl)

            except RateLimitError:
                st.error(
                    "⚠️ Você atingiu o limite gratuito da Together AI.\n"
                    "Aguarde alguns minutos ou ajuste o prompt/tokens."
                )
else:
    st.info("📂 Faça upload do PDF para começar.")
