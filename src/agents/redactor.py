import os
import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from theKeys import COHERE_API_KEY
from state import AgentState

llm = ChatCohere(
    model =COHERE_API_KEY,
    temperature= 0.4
)

Pregunta_usuario = "cuales son los componentes principales para la Arquitectura de Redes VCN en OCI"

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

def Redactor_Pedagogico(state:AgentState) -> dict:

    print("✍️  [Agente redactor] Redactando el contenido basandose en el perfil y formato del usuario...")

    respuesta_RAG = state.get("datos_rag","")
    formato = state.get("formato","texto_estandar")
    nicho = state.get("nicho","tecnologia")
    perfil = state.get("perfil","principiante")

    redaccion = cadena_redactor.invoke({"texto_redactar":respuesta_RAG, "formato": formato, "nicho": nicho, "perfil": perfil})

    print("✅ [Agente redactor] Redacción completada.")
    
    return {"resultado_prueba": redaccion.content}


if __name__ == "__main__":
    # 1. Crear un estado inicial simulado
    estado_inicial = {
        "pregunta": "cuales son los componentes principales para la Arquitectura de Redes VCN en OCI",
        "perfil": "Técnico",
        "formato": "Puntos clave"
    }
    
    # 2. Ejecutar el nodo directamente
    print("--- Ejecutando nodo investigador ---")
    resultado = Redactor_Pedagogico(estado_inicial)
    print("Resultado:", resultado)