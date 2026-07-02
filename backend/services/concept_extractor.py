import json
import time
import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
import os


class ConceptExtractor:
    """
    Extracts explicit academic concepts and contextualized relationships 
    using Google's Gemini 2.5 Flash Lite via the Gemini Developer API.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        print(f"[ConceptExtractor] Connecting to Gemini API using model: {model_name}...")
        # Pull your API key safely from environment variables
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY environment variable. Run 'export GOOGLE_API_KEY=...' in your shell.")
        
        # Initialize native ChatNVIDIA client using your parameter blueprint
        # We enforce temperature 0.0 here because graph entity parsing requires strict factual precision
        self.client = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0,
            max_tokens=8192,
            thinking_budget=0,  # disables extended reasoning — not needed for structured extraction
        )
    def _clean_and_parse_json(self, raw_llm_output: str) -> Dict[str, Any]:
        """
        Enhanced Path B Post-Processor: Strips markdown backticks and isolates 
        anything between the first '{' and the last '}' to completely prevent 
        conversational noise from crashing the parser.
        """
        try:
            # 1. Strip whitespace
            clean_text = raw_llm_output.strip()
            
            # 2. If wrapped in markdown blocks, strip them safely using clear string replaces
            if clean_text.startswith("```"):
                # Split by lines, discard the first line (the ```json) and the last line (the ```)
                lines = clean_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                clean_text = "\n".join(lines).strip()

            # 3. Locate absolute outer JSON boundaries using the DOTALL flag
            match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if not match:
                raise ValueError("No structural JSON object found in raw LLM output.")
                
            json_string = match.group(0)
            return json.loads(json_string)
            
        except Exception as e:
            print(f"[Parser Error] Regex structural recovery failed: {str(e)}")
            print(f"Raw output snippet for debugging: {raw_llm_output[:200]}")
            # Return a safe fallback payload so the pipeline loop doesn't crash
            return {"concepts": [], "relationships": []}

    def extract_from_chunk(self, chunk: Document, max_retries: int = 3) -> Dict[str, Any]:
        """
        Builds the prompt, calls MiniMax-M3 via ChatNVIDIA, post-processes 
        the output, and returns a clean graph translation dictionary.
        Retries automatically on rate-limit (429) errors with increasing backoff.
        """
        chunk_text = chunk.page_content
        source_file = chunk.metadata.get("source", "Unknown Source")
        page_num = chunk.metadata.get("page", 0)

        prompt = f"""
    You are an expert academic analysis system building a high-precision Knowledge Graph for a Computer Science examination grading system.
    Analyze the educational text chunk provided below.

    1. Extract the key academic concepts. For each concept, provide:
       - name: The standardized term (e.g., "Transaction Log"). Use full, standardized
       academic terminology — avoid abbreviations or acronyms unless the text never
       spells out the full term (e.g., write "Database Management System" not "DBMS"
       if the full form appears anywhere in the text).
       - type: Must be one of [CONCEPT, CODE_SNIPPET, DEFINITION].
       - description: A brief summary of what this entity means based on the text.

    2. Extract relationships between these concepts. For each relationship, provide:
       - from: Name of the source concept.
       - to: Name of the target concept.
       - relation: A concise description of the connection (e.g., "is_prerequisite_of", "implements", "violates").
       - source_context: The verbatim phrase or sentence from the text that proves this relationship exists.

    If the text contains no extractable academic concepts (e.g. it is a title page,
    table of contents, exam question, or unreadable OCR fragment), return exactly:
    {{"concepts": [], "relationships": []}}
    Return ONLY a single valid JSON object. Do not include introductory text, markdown backticks, or concluding explanations.

    Expected JSON format:
    {{
      "concepts": [
        {{"name": "Concept Name", "type": "CONCEPT", "description": "Description text"}}
      ],
      "relationships": [
        {{"from": "Concept A", "to": "Concept B", "relation": "relationship_type", "source_context": "literal phrase from text"}}
      ]
    }}

    Text to analyze:
    {chunk_text}
    """

        lc_messages = [{"role": "user", "content": prompt}]

        for attempt in range(max_retries):
            try:
                response = self.client.invoke(lc_messages)
                raw_response = response.content
                parsed_data = self._clean_and_parse_json(raw_response)
                parsed_data["source_metadata"] = {"file": source_file, "page": page_num}
                return parsed_data

            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 15 * (attempt + 1)  # 15s, 30s, 45s
                    print(f"[Rate Limit] Hit 429 on '{source_file}' page {page_num}. Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    print(f"[Error] Failed to extract from '{source_file}' page {page_num} after {attempt + 1} attempts: {e}")
                    return {
                        "concepts": [],
                        "relationships": [],
                        "source_metadata": {"file": source_file, "page": page_num},
                        "extraction_failed": True
                    }


    def extract_from_chunks(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """
    Loops sequentially through a collection of course document chunks,
    pacing requests to stay under the API rate limit, and accumulates
    independent extraction payloads.
        """
        extraction_results = []
        total_chunks = len(chunks)

        print(f"[ConceptExtractor] Starting cloud ChatNVIDIA extraction loop over {total_chunks} blocks...")

        for idx, chunk in enumerate(chunks):
            if idx % 10 == 0 or idx == total_chunks - 1:
                print(f" Processing chunk {idx + 1}/{total_chunks}...\n")

            result = self.extract_from_chunk(chunk)
            extraction_results.append(result)

            time.sleep(1.6)  # paces requests to stay safely under 40/min

        print(f"[ConceptExtractor] Completed extraction phase for all chunks.\n")
        return extraction_results