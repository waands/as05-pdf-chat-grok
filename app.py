import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import openai

# =============================
# Configuração Grok
# =============================
os.environ["XAI_API_KEY"] = "xai-bUNry1gZDukbMQGe0YxSLD5k7lw1rUlGk2uOrwnFc1EmLUSE35Sh5z1GVgGBqBWa3mKFfcykMXZEZRZo"
client = openai.OpenAI(api_key=os.environ["XAI_API_KEY"], base_url="https://api.x.ai/v1")

# =============================
# Streamlit config
# =============================
st.set_page_config(page_title="Assistente PDF com Grok", page_icon="📄🤖")
st.title("📄💬 Assistente PDF com Grok (xAI)")

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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)

    st.success("✅ PDF processado! Agora faça sua pergunta.")

    query = st.text_input("Digite sua pergunta:", placeholder="Ex.: Qual é o objetivo do trabalho?")

    if query:
        with st.spinner("Consultando Grok..."):
            # Recuperar documentos relevantes
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            messages = [
                {
                    "role": "system",
                    "content": "Você é um assistente que responde apenas com base nos documentos fornecidos. Responda sempre em português, de forma clara e objetiva."
                },
                {
                    "role": "user",
                    "content": f"Contexto:\n{context}\n\nPergunta: {query}\n\nResposta:"
                }
            ]

            response = client.chat.completions.create(
                model="grok-beta",
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )

            answer = response.choices[0].message.content
            st.markdown("### 🤖 Resposta")
            st.write(answer)
else:
    st.info("👆 Envie um PDF acima para começar.")
