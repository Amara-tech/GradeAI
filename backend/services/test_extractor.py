# test_extractor.py
import os
import json
from dotenv import load_dotenv

# 1. Load keys before any dependencies read the environment registry
load_dotenv()

from initializer import GraphRAGInitializer
from concept_extractor import ConceptExtractor 
from entity_resolver import EntityResolver

if __name__ == "__main__":
    TEST_DIR = "../Test"
    
    try:
        # Step 1: Initialize Layer 1 Ingestion & Chunking
        print("\n--- [Layer 1] Initializing Course Directory Scan ---")
        initializer = GraphRAGInitializer()
        raw_docs = initializer.process_directory(TEST_DIR)
        all_chunks = initializer.chunk_documents(raw_docs)
        
        # Step 2: Isolate a tiny slice of chunks for stable pipeline tracking
        test_slice = all_chunks[0:4]
        print(f"--- [Layer 1] Isolated {len(test_slice)} chunks for extraction testing ---\n")
        
        # Step 3: Initialize Layer 2 Concept Extractor (NVIDIA Cloud NIM)
        extractor = ConceptExtractor()
        
        # Step 4: Execute the Cloud Extraction Loop
        extraction_payloads = extractor.extract_from_chunks(test_slice)
        
        # Step 5: Beautiful Inspection of Raw LLM Output Results
        print("\n==============================================")
        print("--- INSPECTION OF RAW EXTRACTOR PAYLOADS ---")
        print("==============================================\n")
        
        for idx, payload in enumerate(extraction_payloads):
            metadata = payload.get("source_metadata", {})
            print(f"📦 [Payload Fragment {idx + 1}] -> Source: {metadata.get('file')} | Page: {metadata.get('page')}")
            print(f"   Extracted Concepts: {len(payload.get('concepts', []))}")
            print(f"   Extracted Links:    {len(payload.get('relationships', []))}\n")
            
        # Step 6: Initialize and Execute Layer 3 Entity Resolution
        print("--- [Layer 3] Executing Vectorized Entity Resolution ---")
        resolver = EntityResolver(similarity_threshold=0.85)
        master_graph = resolver.resolve(extraction_payloads)
        
        # Step 7: Final Master Consolidated Inspection
        print("\n==============================================")
        print("--- INSPECTION OF MASTER CONSOLIDATED GRAPH ---")
        print("==============================================\n")
        
        print(f"Total Unique Resolved Nodes: {len(master_graph['nodes'])}")
        print(f"Total Unique Deduplicated Edges: {len(master_graph['edges'])}\n")
        
        print("--- Unified Nodes List ---")
        for node in master_graph["nodes"]:
            print(f" 🟢 [{node['type']}] Key: \"{node['id']}\"")
            print(f"    Desc: {node['description'][:120]}...")
            
        print("\n--- Canonical Unified Edges ---")
        for edge in master_graph["edges"]:
            print(f" ⛓️  \"{edge['source']}\" --({edge['relation']})--> \"{edge['target']}\"")
            print(f"    Proofs gathered: {len(edge['contexts'])} context sentence(s).")      

    except Exception as e:
        print(f"\n❌ Pipeline Execution Failed: {str(e)}")