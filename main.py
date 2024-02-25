import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
import tempfile
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

st.set_page_config(page_title='RAG')
st.title('🦜🔗 LangChain RAG')
st.caption('Optimization of the RAG pipeline in progress...')

st.sidebar.write('[My Portfolio](https://lancen.streamlit.app/)')
st.sidebar.caption("Made by [Lance Nguyen](https://www.linkedin.com/in/lancedin/)")

st.sidebar.info(
    """
    Info: Connect your data with a LLM! Upload any PDF(s) you want to ask questions about, then a chatbot will appear.
    """
)

with st.sidebar.expander('**My Other Apps**'):
    st.caption('[SpotOn](https://spoton.streamlit.app/)')
    st.caption('[Qdoc](https://qdocst.streamlit.app/)')
    st.caption('[CooPA](https://coopas.streamlit.app/)')

# Save the uploaded PDF file to the temporary directory
if 'temp_path' not in st.session_state:
    st.session_state.temp_path = None

uploaded_file = st.file_uploader("Upload a PDF file to activate the chatbot", type=["pdf"])

if uploaded_file is not None:
    if st.session_state.temp_path is None:
        st.session_state.temp_path = tempfile.mkdtemp()

    pdf_path = os.path.join(st.session_state.temp_path, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.success(f"{uploaded_file.name} successfully uploaded to temporary folder.")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if st.session_state.temp_path is not None:
    loader = DirectoryLoader(st.session_state.temp_path, glob="**/*.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""
    prompt = PromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if question := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        response = chain.invoke(question)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
