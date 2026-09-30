from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_cohere import ChatCohere
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from theKeys import COHERE_API_KEY
from ChunkEmbeddings import chunk_embeddings
from ingestion import process_document

#Agregando el agente

llm = ChatCohere(
    api_key=COHERE_API_KEY,
    temperature = 0
)


class Mens(TypedDict):

    perfil: str
    Pregunta_usuario: str
    formato: str
    nicho: str

    resultado_prueba: str
    datos_rag: str
    score: int
    aprobado: bool

Pregunta_usuario = "cuales son los componentes principales para la Arquitectura de Redes VCN en OCI"

sample_file = "data/documento_prueba.md"

docs = process_document(sample_file)

retriever = chunk_embeddings(docs)

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


def Investigador_RAG(Mens) -> Mens:

    pregunta = Mens.get("pregunta_usuario","")
    formato = Mens.get("formato","texto_estandar")
    nicho = Mens.get("nicho","tecnologia")
    perfil = Mens.get("perfil","principiante")

    busqueda = f"{nicho}:{pregunta}"

    documentos_relacionados = retriever.invoke(busqueda)

    if not documentos_relacionados:
        return { "respuesta": "No tengo información al respecto"}

    respuesta = cadena_rag.invoke({"input": Pregunta_usuario, "context": documentos_relacionados, "formato": formato, "perfil": perfil, "nicho": nicho})
    
    return { "datos_rag":respuesta}


prompt_redactor = ChatPromptTemplate(
    [
        ("system",
         """
        Eres un experto en redacción pedagógica y diseño instruccional.
        Tu objetivo es instruir a los distintos empleados de manera puntual y precisa de acuerdo a los siguientes parametros.
        - Perfil del usuario final: {perfil}
        - Formato esperado: {formato} (ejemplo: Flashcards, guía paso a paso, resumen ejecutivo)
        - Nicho/Tema: {nicho}

        Reglas:
        1. Adapta el tono, vocabulario y estructura únicamente según el perfil y formato requeridos.
        2. Mantén la fidelidad al contenido otorgado (no inventes datos ni alucines).
        """
        ),
        ("human", 
         """
        Por favor adapta el siguiente contenido tecnico al formato especificado:

        contenido a redactar: {texto_redactar}
        """)
    ]
)

cadena_redactor = prompt_redactor | llm

def Redactor_Pedagogico(Mens) -> Mens:

    respuesta_RAG = Mens.get("datos_rag","")
    formato = Mens.get("formato","texto_estandar")
    nicho = Mens.get("nicho","tecnologia")
    perfil = Mens.get("perfil","principiante")

    redaccion = cadena_redactor.invoke({"texto_redactar":respuesta_RAG, "formato": formato, "nicho": nicho, "perfil": perfil})

    return {"resultado_prueba": redaccion.content}

"""def Agente_Revisor(Mens) -> Mens:

    return"""

#Creación del g

agent_builder = StateGraph(Mens)

agent_builder.add_node("RAG",Investigador_RAG)
agent_builder.add_node("Redactor",Redactor_Pedagogico)
"""agent_builder.add_node("Revisor",Agente_Revisor)"""

agent_builder.add_edge(START,"RAG")
agent_builder.add_edge("RAG","Redactor")
#agent_builder.add_edge("Redactor","Revisor")

#Revisión
#agent_builder.add_conditional_edge("Revisor",Arista_Revisor["Devolver":"Redactor"])

agent_builder.add_edge("Redactor",END)
