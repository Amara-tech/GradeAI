import hashlib
import psycopg2
from typing import Dict, Any


class MaterialDuplicateChecker:
    """
    Layer 0 — Pre-ingestion gatekeeper. Checks whether uploaded course material
    content already exists for a course BEFORE any expensive pipeline stages run.
    """

    def __init__(self, db_connection):
        self.conn = db_connection

    def compute_file_hash(self, file_path: str) -> str:
        """Computes a SHA-256 hash of a file's contents."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def check_duplicate(self, course_id: int, filename: str, file_path: str) -> Dict[str, Any]:
        """
        Returns {"status": "duplicate" | "filename_conflict" | "new", "message": str, "file_hash": str}
        """
        file_hash = self.compute_file_hash(file_path)
        cursor = self.conn.cursor()

        cursor.execute(
            "SELECT id, filename FROM course_materials WHERE course_id = %s AND file_hash = %s",
            (course_id, file_hash)
        )
        exact_match = cursor.fetchone()
        if exact_match:
            return {
                "status": "duplicate",
                "message": f"This exact file content already exists as '{exact_match[1]}'.",
                "file_hash": file_hash
            }

        cursor.execute(
            "SELECT id FROM course_materials WHERE course_id = %s AND filename = %s",
            (course_id, filename)
        )
        if cursor.fetchone():
            return {
                "status": "filename_conflict",
                "message": "Filename matches an existing file but content differs — likely an update.",
                "file_hash": file_hash
            }

        return {"status": "new", "message": "No duplicate detected.", "file_hash": file_hash}