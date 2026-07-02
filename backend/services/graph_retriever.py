import requests
import logging
from typing import List, Dict, Any
from neo4j import GraphDatabase

logger = logging.getLogger("GraphRetriever")

class GraphRetriever:
    """
    Layer 5 Retrieval Engine. Executes type-isolated Vector Queries against Neo4j,
    performs one-hop graph neighbor expansions, and compiles grounding context.
    """
    
    def __init__(self, driver: GraphDatabase.driver, embedding_model: str = "nomic-embed-text"):
        self.driver = driver
        self.model_name = embedding_model
        self.ollama_endpoint = "http://localhost:11434/api/embed"
        
    def _get_query_embedding(self, text: str) -> List[float]:
        """Translates the grading criteria target text into a 768-dim nomic vector."""
        payload = {"model": self.model_name, "input": [text]}
        try:
            response = requests.post(self.ollama_endpoint, json=payload)
            response.raise_for_status()
            return response.json().get("embeddings", [[]])[0]
        except Exception as e:
            raise RuntimeError(f"[Retriever Error] Failed to compute query vector: {str(e)}")

    def _query_type_bucket(self, tx, course_id: int, embedding: List[float], node_type: str, limit: int) -> List[Dict[str, Any]]:
        """Executes vector search isolated to a specific node type and course profile,
        using Neo4j's in-index filtered SEARCH clause (2026.02+)."""
        cypher = """
        MATCH (candidate:Concept)
        WHERE candidate.course_id = $course_id AND candidate.type = $node_type
        SEARCH candidate IN (
            VECTOR INDEX concept_embeddings FOR $embedding LIMIT $limit
            )
        SCORE AS score

        OPTIONAL MATCH (candidate)-[r:RELATED]-(neighbor:Concept {course_id: $course_id})
        RETURN candidate.name AS anchor_name,
               candidate.description AS anchor_desc,
               candidate.type AS anchor_type,
               score,
               collect(DISTINCT {
                   name: neighbor.name,
                   relation: r.relation_type,
                   proofs: r.proof_contexts
                }) AS connections
        """
        result = tx.run(cypher, embedding=embedding, course_id=course_id, node_type=node_type, limit=limit)
        return [record.data() for record in result]
    
    def retrieve_context(self, course_id: int, question_text: str, mark_scheme_text: str) -> str:
        """
        Orchestrates end-to-end multi-type context retrieval. Compiles anchor nodes,
        relational neighbors, and historical proof sentences into a structured text context.
        """
        logger.info(f"Initiating graph RAG context retrieval for course {course_id}...")
        
        # 1. Combine prompt targets into a unified retrieval focus point
        search_focus = f"Question: {question_text} | Grading Criteria: {mark_scheme_text}"
        
        # 2. Extract vector coordinates
        query_vector = self._get_query_embedding(search_focus)
        
        # 3. Execute isolated searches to guarantee type balancing (3 Concept, 3 Code, 2 Definition)
        search_slots = [
            ("CONCEPT", 3),
            ("CODE_SNIPPET", 3),
            ("DEFINITION", 2)
        ]
        
        raw_matches = []
        with self.driver.session() as session:
            for node_type, limit in search_slots:
                type_records = session.execute_read(
                    self._query_type_bucket, course_id, query_vector, node_type, limit
                )
                raw_matches.extend(type_records)

        if not raw_matches:
            return "No matching background knowledge graph context found for this course tracking index."

        # 4. Serialize retrieved sub-graph into highly structural Markdown
        context_blocks = ["### VERIFIED BACKGROUND COURSE KNOWLEDGE CONTEXT"]
        
        for match in raw_matches:
            if not match["anchor_name"]:
                continue
                
            block = (
                f"📍 [{match['anchor_type']}] Concept: \"{match['anchor_name']}\"\n"
                f"   Vector Match Confidence: {match['score']:.4f}\n"
                f"   Description: {match['anchor_desc']}\n"
            )
            # Append neighborhood links if they exist
            valid_connections = [c for c in match["connections"] if c.get("name")]
            if valid_connections:
                block += "   Relational Connections:\n"
                for conn in valid_connections:
                    block += f"     - Linked to \"{conn['name']}\" via relationship ({conn['relation']})\n"
                    if conn["proofs"]:
                        block += f"       Lecturer Reference Context: \"{' | '.join(conn['proofs'])}\"\n"
            
            context_blocks.append(block)
            
        return "\n".join(context_blocks)