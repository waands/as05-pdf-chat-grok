import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# =============================
# Configuração Gemini
# =============================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# =============================
# Streamlit config
# =============================
st.set_page_config(page_title="Assistente PDF com Gemini", page_icon="📄🤖")
st.title("📄💬 Assistente PDF com Gemini (Google AI)")

uploaded_file = st.file_uploader("Faça upload de um PDF", type=["pdf"])

if uploaded_file:
    # Salvar temporariamente
    temp_path = f"/tmp/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Carregar e dividir
    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectordb = FAISS.from_documents(chunks, embeddings)

    st.success("✅ PDF processado! Agora faça sua pergunta.")

    query = st.text_input("Digite sua pergunta:", placeholder="Ex.: Qual é o objetivo do trabalho?")

    if query:
        with st.spinner("Consultando Gemini..."):
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Montar prompt
            messages = [
                f"Contexto:\n{context}\n\nPergunta: {query}\n\nResponda em português, de forma clara e objetiva."
            ]

            # 1) Trocar para gemini-1.0-pro (menos tokens)
            model = genai.GenerativeModel("models/gemini-1.0-pro")

            # 2) Reduzir contexto e resposta
            retriever = vectordb.as_retriever(search_kwargs={"k": 2})   # em vez de 3
            response = model.generate_content(
                messages,
                generation_config={
                    "max_output_tokens": 300,     # antes 500
                    "temperature": 0.1
                }
            )

            # 3) Capturar erro de quota
            try:
                answer = response.text
            except Exception as e:
                st.error(
                    "⚠️ Limite gratuito do Gemini atingido. "
                    "Tente novamente em alguns minutos ou ative billing para mais requisições."
                )
                st.stop()



            st.markdown("### 🤖 Resposta")
            st.write(answer)
else:
    st.info("👆 Envie um PDF acima para começar.")
