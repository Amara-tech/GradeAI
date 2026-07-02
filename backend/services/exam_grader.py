"""
exam_grader.py

Layer 6 — The Outer Grading LLM.

Combines retrieved Graph RAG context, the exam question, the mark scheme,
and a student's answer into a single grounded grading request. This is the
component the lecturer/examiner directly interacts with — it produces a
suggested mark, concepts addressed, gaps identified, and a justification,
while the examiner retains final authority to review and override.
"""

import json
import re
import logging
import os
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger("ExamGrader")


GRADING_PROMPT_TEMPLATE = """
You are an expert academic grading assistant. You grade fairly and precisely — neither
overly generous nor overly harsh. You evaluate based on what the student actually
demonstrates understanding of, not on whether they used exact textbook terminology.

EXAM QUESTION:
{question_text}

MARK SCHEME:
{mark_scheme_text}

VERIFIED COURSE KNOWLEDGE GRAPH CONTEXT (ground truth — what was taught):
{retrieved_context}

STUDENT'S ANSWER:
{student_answer}

GRADING PRINCIPLES:
- A student does NOT need to use exact textbook terms to earn marks, but they DO need
  to demonstrate genuine understanding of the specific concept — not just gesture at
  the general topic.
- Distinguish between "correct but general/vague" (deserves PARTIAL credit on the
  definition criterion only) versus "demonstrates the specific underlying mechanism"
  (deserves credit on that specific criterion).
- Example of PARTIAL credit: if a student says normalization "makes the database work
  better and organizes data," that is a CORRECT but GENERAL statement — award partial
  marks (e.g. 1-2 of 3) for the definition criterion, since it shows basic awareness
  but lacks precision about WHY/HOW.
- Example of NO credit: do not award marks for "redundancy elimination," "data
  dependency," or "specific normal forms" UNLESS the student says something that
  reflects that specific concept — even in their own words. "Splitting tables" alone,
  with no further explanation of WHY tables are split, is NOT evidence of redundancy
  elimination or normal forms — it is just a vague action with no demonstrated reasoning.
  If the student explains splitting tables IN A WAY that shows they understand it
  reduces duplicate data, THEN credit redundancy elimination.
- Each criterion is scored independently. A vague general answer should score low
  across most criteria, but can still earn partial credit on the definition criterion
  specifically, since "correct but imprecise" is different from "absent."

Your task:
1. Identify which concepts the student meaningfully demonstrated, even in their own words.
2. Identify specific gaps — concepts not demonstrated, even vaguely.
3. Score each mark-scheme criterion individually (allowing partial credit per criterion
   where reasonable) and sum them for the total.
4. Justify each score by referencing the specific part of the student's answer it's based on.

Return ONLY a valid JSON object in this exact format, no other text:
{{
  "concepts_addressed": ["concept 1"],
  "gaps_identified": ["specific missing concept 1"],
  "criteria_breakdown": [
    {{"criterion": "...", "marks_available": 3, "marks_awarded": 1, "reason": "..."}}
  ],
  "suggested_mark": <sum of marks_awarded>,
  "total_marks": <number>,
  "justification": "Overall summary"
}}

IMPORTANT: suggested_mark MUST equal the sum of all marks_awarded values in criteria_breakdown.
"""


class ExamGrader:
    """
    Layer 6 Grading Engine. Assembles grounded grading prompts from retrieved
    Graph RAG context and produces a structured, examiner-reviewable grading
    suggestion. The examiner always retains final authority over the mark.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        logger.info(f"[ExamGrader] Connecting to Gemini API using model: {model_name}...")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY environment variable.")

        self.client = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0,
            max_tokens=4096,
        )

    def _clean_and_parse_json(self, raw_llm_output: str) -> Dict[str, Any]:
        """
        Strips markdown fences and isolates the outer JSON object, with a
        light auto-repair pass for common LLM JSON malformations before
        falling back to a safe error payload.
        """
        try:
            clean_text = raw_llm_output.strip()

            if clean_text.startswith("```"):
                lines = clean_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                clean_text = "\n".join(lines).strip()

            match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if not match:
                raise ValueError("No structural JSON object found in raw LLM output.")

            json_string = match.group(0)

            try:
                return json.loads(json_string)
            except json.JSONDecodeError:
                repaired = json_string
                repaired = re.sub(r',\s*}', '}', repaired)
                repaired = re.sub(r',\s*]', ']', repaired)
                repaired = re.sub(r'}\s*{', '},{', repaired)
                return json.loads(repaired)

        except Exception as e:
            logger.error(f"[Parser Error] Grading JSON recovery failed: {str(e)}")
            logger.error(f"Raw output snippet: {raw_llm_output[:300]}")
            return {
                "concepts_addressed": [],
                "gaps_identified": [],
                "suggested_mark": None,
                "total_marks": None,
                "justification": "Grading failed — could not parse model output. Examiner review required.",
                "grading_failed": True
            }

    def grade_answer(
        self,
        question_text: str,
        mark_scheme_text: str,
        retrieved_context: str,
        student_answer: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Produces a grounded grading suggestion for one student answer.

        This is the primary entry point the examiner-facing application calls.
        The returned dict is a SUGGESTION ONLY — the examiner reviews and
        finalises the actual recorded mark.
        """
        prompt = GRADING_PROMPT_TEMPLATE.format(
            question_text=question_text,
            mark_scheme_text=mark_scheme_text,
            retrieved_context=retrieved_context,
            student_answer=student_answer
        )

        lc_messages = [{"role": "user", "content": prompt}]

        for attempt in range(max_retries):
            try:
                response = self.client.invoke(lc_messages)
                raw_response = response.content
                parsed = self._clean_and_parse_json(raw_response)
                return parsed

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"[ExamGrader] Attempt {attempt + 1} failed: {e}. Retrying...")
                else:
                    logger.error(f"[ExamGrader] Failed after {attempt + 1} attempts: {e}")
                    return {
                        "concepts_addressed": [],
                        "gaps_identified": [],
                        "suggested_mark": None,
                        "total_marks": None,
                        "justification": f"Grading request failed: {str(e)}. Examiner review required.",
                        "grading_failed": True
                    }