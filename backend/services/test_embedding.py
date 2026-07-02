"""
test_embedding_batch.py

Standalone test for EntityResolver._get_batch_embeddings.
Run this BEFORE re-running the full pipeline to confirm the
Ollama batching fix actually works.

Usage:
    python test_embedding_batch.py
"""

from entity_resolver import EntityResolver

def main():
    resolver = EntityResolver(embedding_model="nomic-embed-text", similarity_threshold=0.85)

    # Simulate a realistic load: more than one batch_size's worth of names,
    # including a duplicate-ish pair to sanity-check downstream clustering later.
    test_names = [f"Test Concept {i}" for i in range(120)]

    # --- Comparison A: name only (the original, weaker signal) ---
    name_a_only = "Database Management System"
    name_b_only = "Database Management System (DBMS)"

    # --- Comparison B: name + description (the improved signal) ---
    name_a_full = (
        "Database Management System: Software that manages and stores data "
        "in a database, allowing creation, retrieval, update, and deletion of data."
    )
    name_b_full = (
        "Database Management System (DBMS): Software that manages and stores "
        "data in a database, allowing creation, retrieval, update, and deletion of data."
    )

    test_names[5] = name_a_only
    test_names[6] = name_b_only
    test_names[7] = name_a_full
    test_names[8] = name_b_full

    print(f"--- Testing _get_batch_embeddings with {len(test_names)} names ---\n")

    try:
        embeddings = resolver._get_batch_embeddings(test_names, batch_size=50)
        print(f"\n✅ SUCCESS")
        print(f"Returned matrix shape: {embeddings.shape}")
        print(f"Expected shape: ({len(test_names)}, <embedding_dim>)")

        if embeddings.shape[0] != len(test_names):
            print(f"⚠️  WARNING: row count mismatch! Got {embeddings.shape[0]}, expected {len(test_names)}")
        else:
            print("Row count matches input count. ✅")

        import numpy as np

        def cosine_sim(vec_a, vec_b):
            return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))

        # Comparison A: name only
        sim_name_only = cosine_sim(embeddings[5], embeddings[6])
        print(f"\n[Name only]      Similarity: {sim_name_only:.4f}")
        print(f"  '{name_a_only}'")
        print(f"  '{name_b_only}'")

        # Comparison B: name + description
        sim_name_desc = cosine_sim(embeddings[7], embeddings[8])
        print(f"\n[Name + desc]    Similarity: {sim_name_desc:.4f}")
        print(f"  '{name_a_full[:60]}...'")
        print(f"  '{name_b_full[:60]}...'")

        print(f"\n--- Comparison ---")
        print(f"Name only:      {sim_name_only:.4f}")
        print(f"Name + desc:    {sim_name_desc:.4f}")
        print(f"Improvement:    {sim_name_desc - sim_name_only:+.4f}")
        print(f"Threshold:      0.85")
        print(f"Name only clears threshold?   {'✅ YES' if sim_name_only >= 0.85 else '❌ NO'}")
        print(f"Name+desc clears threshold?   {'✅ YES' if sim_name_desc >= 0.85 else '❌ NO'}")

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        print("\nIf this still fails, try lowering batch_size further (e.g. 25 or 10)")
        print("in the test_names call above, or check that Ollama is running:")
        print("  ollama list   (should show nomic-embed-text)")
        print("  ollama serve  (if not already running)")


if __name__ == "__main__":
    main()