# entity_resolver.py
import json
import requests
import numpy as np
from typing import List, Dict, Any

class EntityResolver:
    """
    Implements Layer 3 Entity Resolution. Groups near-duplicate academic concepts 
    using local vector embeddings and rewrites graph edges to maintain relational integrity.
    """
    
    def __init__(self, embedding_model: str = "nomic-embed-text", similarity_threshold: float = 0.85):
        self.model_name = embedding_model
        self.threshold = similarity_threshold
        self.ollama_endpoint = "http://localhost:11434/api/embed"

    def _get_batch_embeddings(self, texts: List[str], batch_size: int = 50) -> np.ndarray:
        """
    Calls local Ollama HTTP API to embed an array of concept strings.
    Splits into smaller batches to avoid exceeding Ollama's request size limits,
    then concatenates all results into one matrix.
    """
        if not texts:
            return np.empty((0, 0))

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {
            "model": self.model_name,
            "input": batch
            }

            try:
                response = requests.post(self.ollama_endpoint, json=payload)
                response.raise_for_status()
                response_data = response.json()
                embeddings = response_data.get("embeddings", [])
                all_embeddings.extend(embeddings)
                print(f"[EntityResolver] Embedded batch {i // batch_size + 1} ({len(batch)} concepts)...")

            except Exception as e:
                raise RuntimeError(f"[Ollama Error] Embedding batch retrieval failed at batch starting index {i}: {str(e)}")

        return np.array(all_embeddings, dtype=np.float32)

    def _calculate_cosine_similarity(self, vecA: np.ndarray, vecB: np.ndarray) -> float:
        """Calculates standard cosine similarity between two normal vectors."""
        dot_product = np.dot(vecA, vecB)
        normA = np.linalg.norm(vecA)
        normB = np.linalg.norm(vecB)
        if normA == 0 or normB == 0:
            return 0.0
        return float(dot_product / (normA * normB))

    def resolve(self, all_extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes resolution pipeline: collects entities, clusters them semantically 
        using rich name+description signals, selects canonical leaders, and rewrites 
        relation pointers with fallback key protections.
        """
        print(f"[EntityResolver] Starting local cluster resolution (Threshold: {self.threshold})...")
        
        # 1. Collect unique concepts across all chunk payloads
        raw_concepts = {}
        for payload in all_extractions:
            for concept in payload.get("concepts", []):
                # Defensive guard against empty or missing concept name properties
                name_raw = concept.get("name")
                if not name_raw:
                    continue
                name = str(name_raw).strip()
                
                if name not in raw_concepts:
                    raw_concepts[name] = {
                        "type": concept.get("type", "CONCEPT"),
                        "descriptions": [concept.get("description", "")]
                    }
                else:
                    raw_concepts[name]["descriptions"].append(concept.get("description", ""))

        unique_names = list(raw_concepts.keys())
        if not unique_names:
            return {"nodes": [], "edges": []}

        print(f"[EntityResolver] Found {len(unique_names)} unique literal string concepts. Vectorizing...")

        # 2. Build a richer embedding signal: "name: description" instead of bare name.
        # Uses the first available non-empty description per concept as a representative signal.
        embedding_inputs = []
        for name in unique_names:
            descriptions = raw_concepts[name]["descriptions"]
            first_desc = next((d for d in descriptions if d), "")
            embedding_text = f"{name}: {first_desc}" if first_desc else name
            embedding_inputs.append(embedding_text)

        embeddings_matrix = self._get_batch_embeddings(embedding_inputs)

        # 3 & 4. Centroid-based Cluster Grouping (Eliminates sequential chaining)
        clusters: List[List[int]] = []  # Holds lists of indices pointing to unique_names
        
        for i in range(len(unique_names)):
            vec_current = embeddings_matrix[i]
            placed = False
            
            for cluster in clusters:
                # Compare current vector against the cluster leader (the first entity in that group)
                vec_leader = embeddings_matrix[cluster[0]]
                similarity = self._calculate_cosine_similarity(vec_current, vec_leader)
                
                if similarity >= self.threshold:
                    cluster.append(i)
                    placed = True
                    break
            
            if not placed:
                # Create a new semantic cluster with this item as the leader
                clusters.append([i])

        # 5. Build translation mapping and compile unified node data structures
        translation_map: Dict[str, str] = {}
        unified_nodes: Dict[str, Dict[str, Any]] = {}

        for cluster in clusters:
            cluster_names = [unique_names[idx] for idx in cluster]
            
            # Select Canonical Leader: longest/most complete form
            canonical_name = max(cluster_names, key=len)
            
            # Map all cluster members to this single leader
            for name in cluster_names:
                translation_map[name] = canonical_name

            # Merge description strings into a single deduplicated string for the grader
            merged_descriptions = []
            node_type = "CONCEPT"
            for idx in cluster:
                name_item = unique_names[idx]
                node_type = raw_concepts[name_item]["type"]  # Maintain types
                merged_descriptions.extend(raw_concepts[name_item]["descriptions"])
            
            clean_desc = " | ".join(list(set([d for d in merged_descriptions if d])))

            unified_nodes[canonical_name] = {
                "id": canonical_name,
                "type": node_type,
                "description": clean_desc if clean_desc else "No contextual description provided."
            }

        print(f"[EntityResolver] Clustering complete: Consolidated {len(unique_names)} terms down to {len(unified_nodes)} canonical concepts.")

        # 6. Edge Rewriting and deduplication (Fortified with strict key fallback protection)
        unified_edges: Dict[str, Dict[str, Any]] = {}
        
        for payload in all_extractions:
            for rel in payload.get("relationships", []):
                # Native structural recovery: support both standard variants ('from'/'to' and 'source'/'target')
                raw_from = rel.get("from") or rel.get("source")
                raw_to = rel.get("to") or rel.get("target")
                
                # If an edge is completely missing its essential endpoints, drop it safely and keep running
                if not raw_from or not raw_to:
                    continue
                    
                raw_from = str(raw_from).strip()
                raw_to = str(raw_to).strip()
                
                # If an LLM extracted a link for a concept it failed to explicitly define, skip gracefully
                if raw_from not in translation_map or raw_to not in translation_map:
                    continue
                    
                # Rewrite keys to their matching canonical baseline clusters
                canonical_from = translation_map[raw_from]
                canonical_to = translation_map[raw_to]
                relation_type = rel.get("relation", "is_associated_with").lower().strip()
                
                # Prevent self-looping edges caused by a cluster consolidation
                if canonical_from == canonical_to:
                    continue
                    
                # Create a strict composite dictionary key to deduplicate identical relations across chunks
                edge_key = f"{canonical_from}--[{relation_type}]-->{canonical_to}"
                
                if edge_key not in unified_edges:
                    unified_edges[edge_key] = {
                        "source": canonical_from,
                        "target": canonical_to,
                        "relation": relation_type,
                        "contexts": [rel.get("source_context", "")]
                    }
                else:
                    # Append new source contexts safely so the grader retains all reference sentences
                    ctx_sentence = rel.get("source_context", "")
                    if ctx_sentence and ctx_sentence not in unified_edges[edge_key]["contexts"]:
                        unified_edges[edge_key]["contexts"].append(ctx_sentence)

        # Construct final Graph compilation structure
        master_graph = {
            "nodes": list(unified_nodes.values()),
            "edges": list(unified_edges.values())
        }
        
        print(f"[EntityResolver] Edge re-mapping complete. Final Graph contains {len(master_graph['edges'])} structured connections.\n")
        return master_graph