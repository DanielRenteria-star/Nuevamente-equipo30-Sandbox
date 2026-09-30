from theKeys import GEMINI_API_KEY
from theModels import GEMINI_PRO, GEMINI_FLASH, GEMINI_EMBEDDINGS
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
    
def chunk_embeddings(docs):
    
    print("\n--- chunk_embeddings -- Muestra del texto limpio ---")
    print(docs[0].page_content[:300])

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs_splits = splitter.split_documents(docs)

    # Crear Embeddings
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    modelo_embeddings = GoogleGenerativeAIEmbeddings(
        model = GEMINI_EMBEDDINGS,
        google_api_key=GEMINI_API_KEY
    )

    # Generando el Vector Store
    from langchain_community.vectorstores import FAISS

    vectorstore = FAISS.from_documents(docs_splits, modelo_embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 4}
    )

    vectorstore.save_local("vectorstore")
    
    return retriever