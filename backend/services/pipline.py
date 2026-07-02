# run_pipeline_neo4j.py
import os
import logging
from dotenv import load_dotenv

# Initialize early system environment variables loading
load_dotenv()

# Configure logger output structure
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PipelineRunner")

from initializer import GraphRAGInitializer
from concept_extractor import ConceptExtractor 
from entity_resolver import EntityResolver
from knowledge_graph_builder import KnowledgeGraphBuilder

# Neo4j Active Database Connection Settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")  # Provide your Neo4j instance password

def run_production_pipeline(course_id: int, source_directory: str):
    logger.info("==========================================================")
    logger.info("🔥 STARTING GRADEAI PRODUCTION PIPELINE (NEO4J TARGET)")
    logger.info("==========================================================\n")
    
    # 1. Initialize Storage Builder
    builder = None
    try:
        builder = KnowledgeGraphBuilder(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD
        )
        
        # ---- LAYER 1: DIRECTORY SCANNING & CHUNKING ----
        logger.info("[Step 1/4] Running Layer 1 Document Parser and Chunking Engine...")
        initializer = GraphRAGInitializer()
        raw_documents = initializer.process_directory(source_directory)
        all_chunks = initializer.chunk_documents(raw_documents)
        
        # Track a production slice for testing (e.g. 10 files) or use full pipeline array
        production_slice = all_chunks
        logger.info(f"Loaded {len(all_chunks)} chunks total. Processing slice of size: {len(production_slice)}")
        
        # ---- LAYER 2: SYSTEM EXTRACTOR (NVIDIA MINIMAX-M3) ----
        logger.info("[Step 2/4] Running Layer 2 Information Extraction via GOOGLE CLOUD...")
        extractor = ConceptExtractor()
        raw_extractions = extractor.extract_from_chunks(production_slice)
        
        # ---- LAYER 3: ENTITY RESOLUTION ----
        logger.info("[Step 3/4] Running Layer 3 Entity Matching Engine...")
        resolver = EntityResolver(similarity_threshold=0.85)
        master_graph = resolver.resolve(raw_extractions)
        
        # ---- LAYER 4: NEO4J KNOWLEDGE GRAPH STORAGE ----
        logger.info("[Step 4/4] Writing Master Unified Graph to Neo4j Ecosystem...")
        summary = builder.build_graph(
            course_id=course_id,
            nodes=master_graph["nodes"],
            edges=master_graph["edges"]
        )
        
        logger.info("==========================================================")
        logger.info("🎉 SUCCESS: MULTIPLEX KNOWLEDGE GRAPH FULLY BUILT IN NEO4J!")
        logger.info(f"   Course Context: {summary['course_id']}")
        logger.info(f"   Canonical Nodes: {summary['nodes_written']}")
        logger.info(f"   Structural Edges: {summary['edges_written']}")
        logger.info("==========================================================\n")
        
    except Exception as e:
        logger.error(f"❌ Critical Pipeline Failure: {str(e)}", exc_info=True)
    finally:
        # Cleanly release the Neo4j connection pool
        if builder:
            builder.close()

if __name__ == "__main__":
    # Run the production integration targeting Course ID 421 (Database Design and Management)
    run_production_pipeline(course_id=421, source_directory="../Test")