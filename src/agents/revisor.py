import os
import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, SecretStr

# Add src folder to path so it can find the_models and the_keys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from state import AgentState
from theKeys import GEMINI_API_KEY
from theModels import GEMINI_PRO


# Define the expected structured output using Pydantic (Hito 4)
class ReviewResult(BaseModel):
    score: float = Field(
        description="Fidelity score from 0.0 to 1.0. 1.0 means perfectly faithful to the context. Lower scores indicate hallucinations or errors."
    )
    feedback: str = Field(
        description="Detailed feedback explaining the score and instructing the writer on what needs to be fixed. If score is 1.0, write 'Approved'."
    )


def reviewer_node(state: AgentState) -> dict:
    """Node responsible for reviewing the draft against the original context."""
    print(
        "🕵️  [Reviewer Agent] Evaluating draft for hallucinations and pedagogical quality..."
    )

    # 1. Extract inputs from the state
    query = state.get("query", "")
    current_draft = state.get("current_draft", "")
    retrieved_docs = state.get("retrieved_docs", [])
    revision_attempts = state.get("revision_attempts", 0)

    # 2. Combine retrieved documents into a single context string
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 3. Initialize the LLM (Using temperature=0.0 for strict, deterministic evaluation)
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_PRO,
        api_key=SecretStr(GEMINI_API_KEY) if GEMINI_API_KEY else None,
        temperature=0.0,
    )

    # Force the LLM to return our Pydantic schema
    structured_llm = llm.with_structured_output(ReviewResult)

    # 4. Create the Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert Fact-Checker and Technical Reviewer.
Your task is to evaluate a drafted text against the original Technical Context.

INSTRUCTIONS:
1. Compare the DRAFT with the TECHNICAL CONTEXT.
2. Check for Hallucinations: Does the draft mention facts, features, or numbers not present in the context?
3. Output a structured evaluation with a 'score' (0.0 to 1.0) and 'feedback'.
4. If there are hallucinations or critical omissions, the score should be below 0.8.
5. If the draft is completely faithful and accurate, give a high score (0.8 - 1.0).

TECHNICAL CONTEXT:
{context}
""",
            ),
            ("human", "Topic: {query}\n\nDRAFT TO REVIEW:\n{draft}"),
        ]
    )

    # 5. Build and invoke the chain
    chain = prompt | structured_llm
    result = chain.invoke({"context": context, "query": query, "draft": current_draft})

    print(f"✅ [Reviewer Agent] Evaluation complete. Score: {result.score}")

    # 6. Return the updated state
    return {
        "source_anchoring_score": result.score,
        "review_feedback": result.feedback,
        "revision_attempts": revision_attempts + 1,
    }
