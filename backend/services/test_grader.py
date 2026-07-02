"""
test_grader.py

End-to-end test: retrieves real Graph RAG context from Neo4j for course 421,
then grades a simulated PARTIAL student answer against it using ExamGrader.

This is the first full pipeline run: Question + Mark Scheme -> GraphRetriever
-> retrieved context -> ExamGrader -> suggested mark + concepts + gaps.

Usage:
    python test_grader.py
"""

from dotenv import load_dotenv
from neo4j import GraphDatabase
import os
import json

from graph_retriever import GraphRetriever
from exam_grader import ExamGrader

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

COURSE_ID = 421


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    retriever = GraphRetriever(driver=driver, embedding_model="nomic-embed-text")
    grader = ExamGrader(model_name="gemini-2.5-flash-lite")

    question_text = "Explain the two-phase commit protocol and how it works."
    mark_scheme_text = (
        "Award marks out of 10 for: stating the purpose of the protocol - ensuring all "
        "participants commit or all roll back (2 marks); correctly identifying the coordinator "
        "and cohort/participant roles (2 marks); correctly describing the commit-request phase "
        "(coordinator asks for votes, participants reply YES/NO) (3 marks); correctly describing "
        "the commit phase (coordinator sends COMMIT or ABORT based on votes, participants "
        "acknowledge) (3 marks)."
    )

    # Deliberately partial/weak simulated answer:
    # - vague general idea, correct direction
    # - mentions "two phases" but doesn't name or describe them
    # - no mention of coordinator/cohort roles
    # - no mention of voting or COMMIT/ABORT outcomes
    student_answer_weak = (
        "The two-phase commit protocol is used in databases to make sure transactions are "
        "handled properly across the system. It has two phases that happen one after the other "
        "to make sure everything works correctly and the database stays consistent even if "
        "something goes wrong."
    )

    # Strong/complete simulated answer:
    # - correctly states the purpose
    # - names coordinator and cohorts explicitly
    # - describes both phases with correct mechanics
    student_answer_strong = (
        "The two-phase commit protocol ensures that all participants in a distributed transaction "
        "either commit or roll back together, keeping databases synchronized. One node acts as "
        "the coordinator, while the other nodes are called cohorts or participants. "
        "In the first phase, the commit-request phase, the coordinator sends a 'ready for commit' "
        "request to each cohort and waits for their votes. Each cohort votes YES if it is prepared "
        "to commit, or NO if it cannot commit due to a local failure. "
        "In the second phase, the commit phase, if the coordinator receives YES from all cohorts, "
        "it decides to commit and sends a COMMIT message to all participants. If it receives any "
        "NO response, or times out waiting, it decides to abort and sends an ABORT message instead. "
        "Participants then acknowledge by replying DONE."
    )

    print("=" * 70)
    print("STEP 1: Retrieving grounded context from knowledge graph...")
    print("=" * 70)

    retrieved_context = retriever.retrieve_context(
        course_id=COURSE_ID,
        question_text=question_text,
        mark_scheme_text=mark_scheme_text
    )
    print(retrieved_context[:500] + "...\n[truncated for display]\n")

    print("=" * 70)
    print("STEP 2: A/B TEST — same answer, WITH context vs WITHOUT context")
    print("=" * 70)
    print("If results differ meaningfully, the graph is contributing.")
    print("If results are nearly identical, the graph is NOT contributing.\n")

    # Use the strong answer for this test since it has the most surface area
    # for context to matter (multiple distinct claims to verify/ground).
    test_answer = student_answer_strong
    print(f"Question: {question_text}")
    print(f"Student answer:\n\"{test_answer}\"\n")

    print("-" * 70)
    print("RUN A — WITH retrieved context")
    print("-" * 70)
    result_with_context = grader.grade_answer(
        question_text=question_text,
        mark_scheme_text=mark_scheme_text,
        retrieved_context=retrieved_context,
        student_answer=test_answer
    )
    print(json.dumps(result_with_context, indent=2))

    print("\n" + "-" * 70)
    print("RUN B — WITHOUT context (forced empty string)")
    print("-" * 70)
    result_without_context = grader.grade_answer(
        question_text=question_text,
        mark_scheme_text=mark_scheme_text,
        retrieved_context="",
        student_answer=test_answer
    )
    print(json.dumps(result_without_context, indent=2))

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    mark_a = result_with_context.get("suggested_mark")
    mark_b = result_without_context.get("suggested_mark")
    print(f"Suggested mark WITH context:    {mark_a}/{result_with_context.get('total_marks')}")
    print(f"Suggested mark WITHOUT context: {mark_b}/{result_without_context.get('total_marks')}")
    print(f"Difference: {abs((mark_a or 0) - (mark_b or 0))} marks")

    gaps_a = set(result_with_context.get("gaps_identified", []))
    gaps_b = set(result_without_context.get("gaps_identified", []))
    print(f"\nGaps identified WITH context:    {gaps_a}")
    print(f"Gaps identified WITHOUT context: {gaps_b}")
    print(f"Same gaps identified? {'YES — context made no difference' if gaps_a == gaps_b else 'NO — context changed the outcome'}")

    driver.close()


if __name__ == "__main__":
    main()