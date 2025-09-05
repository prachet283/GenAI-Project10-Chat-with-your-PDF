import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint,ChatHuggingFace
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough ,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import tempfile

st.title("📚 Chat with your PDF")

# --- Step 1: API Choice ---
api_choice = st.radio("Choose API Provider:", ["HuggingFace", "OpenAI"])

api_key = st.text_input(f"Enter your {api_choice} API key:", type="password")

# --- Step 2: Upload PDF ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and api_key:
    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load and split
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # Step 3: Setup embeddings
    if api_choice == "HuggingFace":
        embedding = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=api_key
        )
    else:
        embedding = OpenAIEmbeddings(openai_api_key=api_key)

    # Step 4: Vectorstore
    vector_store = FAISS.from_documents(chunks, embedding)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Step 5: LLM
    if api_choice == "HuggingFace":
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation",
            huggingfacehub_api_token=api_key
        )
        model = ChatHuggingFace(llm=llm)
    else:
        model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)

    # Prompt
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Use ONLY the following context to answer.
        If context is insufficient, say "I don't know".

        Context:
        {content}

        Question: {question}
        """,
        input_variables=["content", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel({
        "content": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | model | parser

    # --- Step 6: Chat UI ---
    st.subheader("💬 Chat")
    # if "history" not in st.session_state:
    #     st.session_state.history = []

    # user_input = st.text_input("Ask a question about your PDF:")

    # if user_input:
    #     response = main_chain.invoke(user_input)
    #     st.session_state.history.append(("You", user_input))
    #     st.session_state.history.append(("Bot", response))

    # # Display conversation
    # for role, text in st.session_state.history:
    #     st.write(f"**{role}:** {text}")


    # Keep chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask a question about your PDF..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # --- Call your RAG chain here ---
        response = main_chain.invoke(user_input)   # ⬅️ your chain from earlier

        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
