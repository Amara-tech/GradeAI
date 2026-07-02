import requests
import logging
from typing import List, Dict, Any
from neo4j import GraphDatabase

# Configure structured pipeline logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KnowledgeGraphBuilder")

class KnowledgeGraphBuilder:
    """
    Layer 4 Storage Engine for Neo4j. Connects to a Neo4j database instance to
    persist canonical concept nodes (enriched with nomic embeddings) and 
    structural relationship edges.
    """
    
    def __init__(self, uri: str = None, user: str = None, password: str = None,
            driver: GraphDatabase.driver = None, embedding_model: str = "nomic-embed-text"):
        if driver is not None:
            logger.info("Using shared Neo4j driver connection...")
            self.driver = driver
            self._owns_driver = False  # don't close a connection we didn't open
        else:
            if not all([uri, user, password]):
                raise ValueError("Must provide either a pre-built 'driver', or 'uri', 'user', and 'password'.")
            logger.info(f"Establishing singular connection to Neo4j at {uri}...")
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self._owns_driver = True  # this instance opened it, so it should close it

        self.model_name = embedding_model
        self.ollama_endpoint = "http://localhost:11434/api/embed"
        logger.info(f"Ollama vector connector configured to use model: {self.model_name}")
    def close(self) -> None:
        """Closes the underlying Neo4j driver connection safely."""
        if hasattr(self, 'driver') and self.driver:
            logger.info("Closing Neo4j driver connection pool...")
            self.driver.close()
            
    def _embed_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 1: Gathers all concept names and description strings, maps them 
        into managed chunks of 50 to prevent Ollama HTTP 400 buffer overflows,
        and attaches the 768-dimension vectors safely back onto the node payloads.
        """
        if not nodes:
            return []
            
        logger.info(f"Batch embedding {len(nodes)} nodes using local Ollama model '{self.model_name}'...")
        
        # 1. Standardize text representation inputs
        texts_to_embed = []
        for node in nodes:
            name = node.get("id") or node.get("name", "Unknown Concept")
            description = node.get("description", "No description provided.")
            texts_to_embed.append(f"{name}: {description}")
            
        # 2. Slice texts into controlled batch sizes of 50
        batch_size = 50
        all_embeddings = []
        
        try:
            for i in range(0, len(texts_to_embed), batch_size):
                batch_input = texts_to_embed[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                logger.info(f"[KnowledgeGraphBuilder] Vectorizing node batch {batch_num} ({len(batch_input)} elements)...")
                
                payload = {
                    "model": self.model_name,
                    "input": batch_input
                }
                
                response = requests.post(self.ollama_endpoint, json=payload)
                response.raise_for_status()
                batch_vectors = response.json().get("embeddings", [])
                all_embeddings.extend(batch_vectors)
                
            # 3. Map vectors sequentially back into original concept dictionary keys
            for idx, node in enumerate(nodes):
                node["name"] = node.get("id") or node.get("name")
                node["embedding"] = all_embeddings[idx]
                
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to fetch chunked vector embeddings from local Ollama endpoint: {str(e)}")
            logger.warning("Falling back to structural zero-vectors to prevent ingestion lock.")
            # Resilient fallback state
            for node in nodes:
                node["name"] = node.get("id") or node.get("name")
                node["embedding"] = [0.0] * 768
            return nodes
    
            
        

    def _write_nodes(self, course_id: int, embedded_nodes: List[Dict[str, Any]]) -> None:
        """
        Step 2: Runs a high-performance Cypher transaction with UNWIND to batch merge
        concept nodes. Re-running updates property values while maintaining course isolation.
        """
        logger.info(f"Batch-writing {len(embedded_nodes)} Concept nodes to Neo4j for course {course_id}...")
        
        cypher_query = """
        UNWIND $nodes AS node
        MERGE (n:Concept {name: node.name, course_id: $course_id})
        SET n.type = node.type,
            n.description = node.description,
            n.embedding = node.embedding
        """
        
        with self.driver.session() as session:
            session.run(cypher_query, nodes=embedded_nodes, course_id=course_id)

    def _write_edges(self, course_id: int, edges: List[Dict[str, Any]], node_names: List[str]) -> None:
        """
        Step 3: Maps relationships between existing Concepts. Evaluates endpoint integrity,
        warns on orphaned entities, and concatenates unique proof contexts over repeated runs.
        """
        logger.info(f"Processing {len(edges)} potential graph edges for writing...")
        
        # Verify node registration to catch orphaned concept paths before they silently bypass Cypher MATCH bounds
        validated_edges = []
        for edge in edges:
            source = edge.get("source") or edge.get("from")
            target = edge.get("target") or edge.get("to")
            relation = edge.get("relation", "associated_with")
            contexts = edge.get("contexts") or edge.get("proof_contexts", [])
            
            if source not in node_names or target not in node_names:
                logger.warning(
                    f"Skipping orphaned edge: '{source}' --[{relation}]--> '{target}'. "
                    f"One or both concepts did not exist in the resolved canonical nodes list."
                )
                continue
                
            validated_edges.append({
                "from": source,
                "to": target,
                "relation": relation,
                "proof_contexts": contexts
            })
            
        if not validated_edges:
            logger.info("No validated edges found. Skipping edge write step.")
            return

        logger.info(f"Batch-writing {len(validated_edges)} validated connections to Neo4j...")

        # Native Cypher array compression checks for existing context duplicates during re-runs
        cypher_query = """
        UNWIND $edges AS edge
        MATCH (a:Concept {name: edge.from, course_id: $course_id})
        MATCH (b:Concept {name: edge.to, course_id: $course_id})
        MERGE (a)-[r:RELATED {relation_type: edge.relation}]->(b)
        ON CREATE SET r.proof_contexts = edge.proof_contexts
        ON MATCH SET r.proof_contexts = r.proof_contexts + [ctx IN edge.proof_contexts WHERE NOT ctx IN r.proof_contexts]
        """
        
        with self.driver.session() as session:
            session.run(cypher_query, edges=validated_edges, course_id=course_id)

    def build_graph(self, course_id: int, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Public orchestrator entry-point. Receives deduplicated concept payloads from Layer 3,
        vectorizes properties, and structures relational graph dependencies safely.
        """
        logger.info(f"========== Initiating Neo4j Graph Builder Pipeline for Course ID: {course_id} ==========")
        
        # Step 1: Compute 768-dim embeddings
        embedded_nodes = self._embed_nodes(nodes)
        
        # Step 2: Write Concept nodes to Neo4j
        self._write_nodes(course_id, embedded_nodes)
        
        # Capture reference names list for parent tracking checks in Step 3
        canonical_names = [node["name"] for node in embedded_nodes]
        
        # Step 3: Map relationships across elements
        self._write_edges(course_id, edges, canonical_names)
        
        logger.info("========== Neo4j Graph Builder Integration Phase Complete! ==========\n")
        
        return {
            "course_id": course_id,
            "nodes_written": len(embedded_nodes),
            "edges_written": len(edges)
        }