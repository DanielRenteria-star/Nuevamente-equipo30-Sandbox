from typing import TypedDict

class AgentState(TypedDict):

    perfil: str
    Pregunta_usuario: str
    formato: str
    nicho: str

    resultado_prueba: str
    datos_rag: str
    score: int
    aprobado: bool