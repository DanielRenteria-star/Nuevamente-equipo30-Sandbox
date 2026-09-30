import os
import sys

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_cohere import ChatCohere

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from theKeys import COHERE_API_KEY
from ChunkEmbeddings import chunk_embeddings
from ingestion import process_document
from state import AgentState

#Agregando el agente

llm = ChatCohere(
    api_key=COHERE_API_KEY,
    temperature = 0
)

Pregunta_usuario = "cuales son los componentes principales para la Arquitectura de Redes VCN en OCI"

sample_file = "data/documento_prueba.md"

prompt_Investigador_RAG = ChatPromptTemplate(
    [
        ("system",
         """
        Eres el especialista de una biblioteca de una empresa de tecnología.
        Tu objetivo es investigar y extraer información precisa basándote en los siguientes parámetros:
        - Perfil del usuario final: {perfil}
        - Formato esperado: {formato}
        - Nicho/Tema: {nicho}

        Responde siempre utilizando los conocimientos del contexto que se te otorga.
        Si no hay información referente al contexto, responde que NO tienes conocimiento al respecto.
        """),
        ("human", "Contexto: {context}. \nPregunta del empleado: {input}")
    ]
)

cadena_rag = create_stuff_documents_chain(llm, prompt_Investigador_RAG)

def Investigador_RAG(state: AgentState) -> AgentState:

    pregunta = state.get("pregunta_usuario","")
    formato = state.get("formato","texto_estandar")
    nicho = state.get("nicho","tecnologia")
    perfil = state.get("perfil","principiante")

    busqueda = f"{nicho}:{pregunta}" 

    docs = process_document(sample_file)

    retriever = chunk_embeddings(docs)

    documentos_relacionados = retriever.invoke(busqueda)

    if not documentos_relacionados:
        return { "respuesta": "No tengo información al respecto"}

    respuesta = cadena_rag.invoke({"input": Pregunta_usuario, "context": documentos_relacionados, "formato": formato, "perfil": perfil, "nicho": nicho})
    
    return { "datos_rag":respuesta}


