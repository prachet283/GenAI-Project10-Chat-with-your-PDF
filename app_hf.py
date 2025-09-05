from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.runnables import RunnableParallel , RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

#load apis
load_dotenv()

#load the pdfloader
loader = PyPDFLoader('/content/SAR_Image_Colorization_Using_Multidomain_Cycle-Consistency_Generative_Adversarial_Network.pdf')
docs = loader.load()

#text splits in chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 200,
)
chunks = text_splitter.split_documents(docs)

#loading the embedding model
embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

#creating the vector store
vector_store = FAISS.from_documents(chunks, embedding)

#retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

# Prompt
prompt = PromptTemplate(
    template = """
    You are a helpful assistant.
    Answer ONLY from the provided context.
    If the context is insufficient, just say you don't know.

    Context:
    {content}

    Question: {question}
    """,
    input_variables = ['content','question']
)

#LLM
llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.2',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)

#formatting the docs
def format_docs(retrieved_docs):
  content_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return content_text

#creating the chains
parallel_chain = RunnableParallel({
    'content': retriever | RunnableLambda(format_docs),
    'question':RunnablePassthrough()
})

#parser
parser = StrOutputParser()

#main chain
main_chain = parallel_chain | prompt | model | parser