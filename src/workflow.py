import os
import sys


from langchain_cohere import ChatCohere
from langgraph.graph import START, END, StateGraph

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from state import AgentState
from agents.investigador import Investigador_RAG
from agents.redactor import Redactor_Pedagogico
from agents.revisor import reviewer_node

#Agregando el agente


def check_score(state: AgentState) -> str:
    """
    Conditional edge function.
    Decides whether the draft passes the review or needs to go back to the writer.
    """
    score = state.get("source_anchoring_score", 0.0)
    attempts = state.get("revision_attempts", 0)

    if score >= 0.8:
        print(f"🏁 [Graph Orchestrator] Draft Approved! Score: {score}")
        return "approved"
    elif attempts >= 3:
        print(
            f"⚠️ [Graph Orchestrator] Max attempts reached (3). Forcing approval. Score: {score}"
        )
        return "approved"
    else:
        print(
            f"🔄 [Graph Orchestrator] Draft Rejected (Score: {score}). Sending back to Writer..."
        )
        return "needs_revision"



#Creación del workflow
def construccion_grafo():
    agent_builder = StateGraph(AgentState)

    #Creación de los nodos
    agent_builder.add_node("RAG",Investigador_RAG)
    agent_builder.add_node("Redactor",Redactor_Pedagogico)
    agent_builder.add_node("Revisor",reviewer_node)

    #Conexiones
    agent_builder.set_entry_point("RAG")
    agent_builder.add_edge("RAG","Redactor")
    agent_builder.add_edge("Redactor","Revisor")

    agent_builder.add_conditional_edges(
        "Revisor",
            check_score,
            {
                "approved": END,
                "needs_revision": "Redactor",
            }
            )
    return agent_builder.compile()


