import os, time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# --- API KEY (via Secrets) --------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

MODEL_NAME = "models/gemini-1.5-pro"      # troque se list_models mostrar outro
MAX_OUT    = 256                          # menos tokens = menos quota
K_CHUNKS   = 2                            # recupera só 2 trechos (menos tokens)

# --- Streamlit UI ----------------------
st.set_page_config(page_title="PDF → Chat com Gemini", page_icon="📄🤖")
st.title("📄💬 Chat sobre seu PDF (Gemini)")

pdf_file = st.file_uploader("Faça upload do PDF", type="pdf")

if pdf_file:
    # salvar temp
    tmp = f"/tmp/{pdf_file.name}"
    with open(tmp, "wb") as f: f.write(pdf_file.getbuffer())

    # carregar e dividir
    docs = PyPDFLoader(tmp).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)\
             .split_documents(docs)

    # embeddings + índice FAISS
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                model_kwargs={"device": "cpu"})
    vectordb = FAISS.from_documents(chunks, emb)
    retriever = vectordb.as_retriever(search_kwargs={"k": K_CHUNKS})

    st.success("✅ PDF processado! Pergunte algo:")
    q = st.text_input("Pergunta")

    if q:
        with st.spinner("Gerando resposta…"):
            ctx = "\n\n".join(
                d.page_content for d in retriever.get_relevant_documents(q)
            )
            prompt = (f"Contexto:\n{ctx}\n\n"
                      f"Pergunta: {q}\n"
                      f"Responda em português, de forma clara e objetiva.")

            try:
                model  = genai.GenerativeModel(MODEL_NAME)
                resp   = model.generate_content(
                            prompt,
                            generation_config={
                               "max_output_tokens": MAX_OUT,
                               "temperature": 0.1
                            }
                         )
                st.markdown("### 🤖 Resposta")
                st.write(resp.text)

            except ResourceExhausted as e:       # erro 429 de quota
                st.error("⚠️ Você atingiu o limite gratuito da Gemini API.\n"
                         "· Aguarde alguns minutos ou tente amanhã.\n"
                         "· Ou ative billing no Console Google Cloud para aumentar a cota.")
else:
    st.info("📂 Faça upload do PDF para começar.")
