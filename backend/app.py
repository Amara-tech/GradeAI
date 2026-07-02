"""
app.py

GradeAI Flask API — wires the existing Graph RAG pipeline classes
(GraphRAGInitializer, ConceptExtractor, EntityResolver, KnowledgeGraphBuilder,
GraphRetriever, ExamGrader, MaterialDuplicateChecker) into HTTP endpoints.

Architecture notes:
- Material uploads trigger the full pipeline ASYNCHRONOUSLY (threading) so the
  lecturer's request returns immediately. The examiner works against an
  already-built graph later — the two roles are decoupled in time.
- Grading happens ONE ANSWER AT A TIME (not batch), so the examiner can review
  OCR confidence and correct text before triggering each grading call.
- PostgreSQL holds all relational data. Neo4j holds the knowledge graph only.
"""

import os
import threading
import logging
import hashlib
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from neo4j import GraphDatabase
from dotenv import load_dotenv

# --- Your existing pipeline classes ---
from services.router import DocumentIngestionRouter
from services.initializer import GraphRAGInitializer
from services.concept_extractor import ConceptExtractor
from services.entity_resolver import EntityResolver
from services.knowledge_graph_builder import KnowledgeGraphBuilder
from services.graph_retriever import GraphRetriever
from services.exam_grader import ExamGrader

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GradeAI")

app = Flask(__name__)
CORS(app)

# --- Configuration ---
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
UPLOAD_BASE_DIR = os.getenv("UPLOAD_BASE_DIR", "./uploads")
ALLOWED_EXTENSIONS = {".pdf", ".pptx", ".ppt", ".png", ".jpg", ".jpeg", ".tiff", ".docx", ".txt"}

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Build from individual components if DATABASE_URL isn't set directly
    db_host = os.getenv("DB_HOST", "127.0.0.1")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    if all([db_name, db_user, db_password]):
        DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    else:
        raise ValueError("Missing database configuration. Set DATABASE_URL or DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD in .env")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# =====================================================================
# DATABASE HELPERS
# =====================================================================

def get_db():
    """Returns a request-scoped PostgreSQL connection."""
    if "db" not in g:
        g.db = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


# =====================================================================
# AUTH HELPERS
# =====================================================================

def generate_token(user_id: int, role: str) -> str:
    payload = {
        "user_id": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def token_required(f):
    """Decorator: validates JWT, attaches user_id and role to flask.g"""
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            g.user_id = payload["user_id"]
            g.role = payload["role"]
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        return f(*args, **kwargs)
    return decorated


def role_required(required_role: str):
    """Decorator: restricts a route to a specific role (lecturer or examiner)."""
    def decorator(f):
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            if g.get("role") != required_role:
                return jsonify({"error": f"This action requires the '{required_role}' role"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


# =====================================================================
# AUTH ROUTES
# =====================================================================

@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role")

    if not all([name, email, password, role]):
        return jsonify({"error": "name, email, password, and role are required"}), 400
    if role not in ("lecturer", "examiner"):
        return jsonify({"error": "role must be 'lecturer' or 'examiner'"}), 400

    db = get_db()
    cursor = db.cursor()

    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
    if cursor.fetchone():
        return jsonify({"error": "An account with this email already exists"}), 409

    password_hash = generate_password_hash(password)
    cursor.execute(
        "INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, %s) RETURNING id",
        (name, email, password_hash, role)
    )
    user_id = cursor.fetchone()["id"]
    db.commit()

    token = generate_token(user_id, role)
    return jsonify({"token": token, "user": {"id": user_id, "name": name, "role": role}}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not all([email, password]):
        return jsonify({"error": "email and password are required"}), 400

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, name, password_hash, role FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid email or password"}), 401

    token = generate_token(user["id"], user["role"])
    return jsonify({
        "token": token,
        "user": {"id": user["id"], "name": user["name"], "role": user["role"]}
    }), 200


# =====================================================================
# COURSE ROUTES
# =====================================================================

@app.route("/api/courses", methods=["POST"])
@token_required
@role_required("examiner")
def create_course():
    """Examiner creates a course. Lecturers self-assign to it later."""
    data = request.get_json()
    course_name = data.get("course_name")
    course_code = data.get("course_code")

    if not all([course_name, course_code]):
        return jsonify({"error": "course_name and course_code are required"}), 400

    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO courses (course_name, course_code, examiner_id) VALUES (%s, %s, %s) RETURNING id",
        (course_name, course_code, g.user_id)
    )
    course_id = cursor.fetchone()["id"]
    db.commit()

    return jsonify({"id": course_id, "course_name": course_name, "course_code": course_code}), 201


@app.route("/api/courses", methods=["GET"])
@token_required
def list_courses():
    """
    Lecturers see all courses (to self-assign).
    Examiners see only courses they created.
    """
    db = get_db()
    cursor = db.cursor()

    if g.role == "examiner":
        cursor.execute("SELECT * FROM courses WHERE examiner_id = %s ORDER BY course_name", (g.user_id,))
    else:
        cursor.execute("SELECT * FROM courses ORDER BY course_name")

    return jsonify(cursor.fetchall()), 200


@app.route("/api/courses/<int:course_id>/assign-lecturer", methods=["POST"])
@token_required
@role_required("lecturer")
def assign_lecturer_to_course(course_id):
    """Lecturer self-assigns to a course they teach."""
    db = get_db()
    cursor = db.cursor()

    cursor.execute("SELECT id FROM courses WHERE id = %s", (course_id,))
    if not cursor.fetchone():
        return jsonify({"error": "Course not found"}), 404

    cursor.execute(
        "SELECT id FROM course_lecturers WHERE course_id = %s AND lecturer_id = %s",
        (course_id, g.user_id)
    )
    if cursor.fetchone():
        return jsonify({"message": "Already assigned to this course"}), 200

    cursor.execute(
        "INSERT INTO course_lecturers (course_id, lecturer_id) VALUES (%s, %s)",
        (course_id, g.user_id)
    )
    db.commit()
    return jsonify({"message": "Successfully assigned to course"}), 201


# =====================================================================
# MATERIALS ROUTES — async pipeline trigger
# =====================================================================

def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_pipeline_in_background(material_id: int, course_id: int, file_path: str):
    """
    Runs the full Graph RAG ingestion pipeline in a background thread.
    Updates course_materials.status AND progress_percent/progress_stage
    throughout, so the lecturer's UI can show real progress.
    This function owns its OWN database connection since it runs outside
    the Flask request context (g is not available here).
    """
    db = psycopg2.connect(DATABASE_URL)
    cursor = db.cursor()

    def update_progress(stage: str, percent: int):
        cursor.execute(
            "UPDATE course_materials SET progress_stage = %s, progress_percent = %s WHERE id = %s",
            (stage, percent, material_id)
        )
        db.commit()

    try:
        logger.info(f"[Pipeline] Starting background processing for material {material_id} (course {course_id})")

        update_progress("loading_document", 5)
        initializer = GraphRAGInitializer()
        documents = initializer.process_material(file_path)

        update_progress("chunking", 10)
        chunks = initializer.chunk_documents(documents)
        total_chunks = len(chunks)

        # Extraction is the longest stage (one Gemini call per chunk) — it gets
        # the most progress "room", from 10% to 70%, scaled by chunk count.
        update_progress("extracting_concepts", 10)
        extractor = ConceptExtractor()

        raw_extractions = []
        for idx, chunk in enumerate(chunks):
            result = extractor.extract_from_chunk(chunk)
            raw_extractions.append(result)
            # Map chunk progress (0-100% of extraction) onto the 10-70% overall range
            extraction_fraction = (idx + 1) / total_chunks
            overall_percent = 10 + int(extraction_fraction * 60)
            update_progress("extracting_concepts", overall_percent)

        update_progress("resolving_entities", 75)
        resolver = EntityResolver()
        master_graph = resolver.resolve(raw_extractions)

        update_progress("building_graph", 90)
        builder = KnowledgeGraphBuilder(driver=neo4j_driver)
        builder.build_graph(course_id=course_id, nodes=master_graph["nodes"], edges=master_graph["edges"])

        cursor.execute(
            "UPDATE course_materials SET status = 'completed', progress_stage = 'completed', "
            "progress_percent = 100 WHERE id = %s",
            (material_id,)
        )
        db.commit()
        logger.info(f"[Pipeline] Completed material {material_id}: "
                    f"{len(master_graph['nodes'])} nodes, {len(master_graph['edges'])} edges")

    except Exception as e:
        logger.error(f"[Pipeline] Failed processing material {material_id}: {str(e)}")
        cursor.execute(
            "UPDATE course_materials SET status = 'failed', error_message = %s, progress_stage = 'failed' "
            "WHERE id = %s",
            (str(e), material_id)
        )
        db.commit()

    finally:
        cursor.close()
        db.close()


@app.route("/api/courses/<int:course_id>/materials", methods=["POST"])
@token_required
@role_required("lecturer")
def upload_material(course_id):
    """
    Uploads a course material file. Checks for duplicates BEFORE saving or
    processing. If new, saves the file and kicks off the pipeline as a
    background thread, returning immediately (202 Accepted).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename.lower())[1]

    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    course_dir = os.path.join(UPLOAD_BASE_DIR, f"course_{course_id}")
    os.makedirs(course_dir, exist_ok=True)

    temp_path = os.path.join(course_dir, f"_tmp_{filename}")
    file.save(temp_path)
    file_hash = compute_file_hash(temp_path)

    db = get_db()
    cursor = db.cursor()

    # Duplicate check — only blocks if a SUCCESSFUL or IN-PROGRESS attempt already
    # exists with this content. A previously FAILED attempt should not block a retry.
    cursor.execute(
        "SELECT id, filename, status FROM course_materials "
        "WHERE course_id = %s AND file_hash = %s AND status != 'failed'",
        (course_id, file_hash)
    )
    existing = cursor.fetchone()
    if existing:
        os.remove(temp_path)
        return jsonify({
            "error": "duplicate",
            "message": f"This exact file content already exists as '{existing['filename']}' (status: {existing['status']})."
        }), 409

    # Clean up any previous FAILED attempt for this same content before inserting the retry,
    # so we don't accumulate dead rows every time a lecturer retries a failed upload.
    cursor.execute(
        "DELETE FROM course_materials WHERE course_id = %s AND file_hash = %s AND status = 'failed'",
        (course_id, file_hash)
    )

    final_path = os.path.join(course_dir, filename)
    os.rename(temp_path, final_path)

    cursor.execute(
        """INSERT INTO course_materials (course_id, filename, file_path, file_hash, lecturer_id, status)
           VALUES (%s, %s, %s, %s, %s, 'processing') RETURNING id""",
        (course_id, filename, final_path, file_hash, g.user_id)
    )
    material_id = cursor.fetchone()["id"]
    db.commit()

    # Fire and forget — Flask returns immediately, pipeline runs in background
    thread = threading.Thread(
        target=run_pipeline_in_background,
        args=(material_id, course_id, final_path),
        daemon=True
    )
    thread.start()

    return jsonify({
        "id": material_id,
        "filename": filename,
        "status": "processing",
        "message": "Upload received. Knowledge graph is being built in the background."
    }), 202


@app.route("/api/courses/<int:course_id>/materials", methods=["GET"])
@token_required
def list_materials(course_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "SELECT id, filename, status, progress_stage, progress_percent, error_message, uploaded_at "
        "FROM course_materials WHERE course_id = %s ORDER BY uploaded_at DESC",
        (course_id,)
    )
    return jsonify(cursor.fetchall()), 200


# =====================================================================
# EXAM SESSION & QUESTION ROUTES
# =====================================================================

@app.route("/api/exams", methods=["POST"])
@token_required
@role_required("examiner")
def create_exam_session():
    data = request.get_json()
    course_id = data.get("course_id")
    title = data.get("title")

    if not all([course_id, title]):
        return jsonify({"error": "course_id and title are required"}), 400

    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        """INSERT INTO exam_sessions (course_id, examiner_id, title, status)
           VALUES (%s, %s, %s, 'draft') RETURNING id""",
        (course_id, g.user_id, title)
    )
    session_id = cursor.fetchone()["id"]
    db.commit()

    return jsonify({"id": session_id, "title": title, "status": "draft"}), 201


@app.route("/api/exams", methods=["GET"])
@token_required
@role_required("examiner")
def list_exam_sessions():
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "SELECT * FROM exam_sessions WHERE examiner_id = %s ORDER BY created_at DESC",
        (g.user_id,)
    )
    return jsonify(cursor.fetchall()), 200


@app.route("/api/exams/<int:session_id>/questions", methods=["POST"])
@token_required
@role_required("examiner")
def add_question(session_id):
    data = request.get_json()
    question_text = data.get("question_text")
    mark_scheme_text = data.get("mark_scheme_text")
    total_marks = data.get("total_marks")

    if not all([question_text, mark_scheme_text, total_marks]):
        return jsonify({"error": "question_text, mark_scheme_text, and total_marks are required"}), 400

    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        """INSERT INTO questions (exam_session_id, question_text, mark_scheme_text, total_marks)
           VALUES (%s, %s, %s, %s) RETURNING id""",
        (session_id, question_text, mark_scheme_text, total_marks)
    )
    question_id = cursor.fetchone()["id"]
    db.commit()

    return jsonify({"id": question_id, "question_text": question_text, "total_marks": total_marks}), 201


@app.route("/api/exams/<int:session_id>/questions", methods=["GET"])
@token_required
def list_questions(session_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM questions WHERE exam_session_id = %s", (session_id,))
    return jsonify(cursor.fetchall()), 200


# =====================================================================
# GRADING ROUTES — one answer at a time, examiner stays in control
# =====================================================================

@app.route("/api/answers", methods=["POST"])
@token_required
@role_required("examiner")
def submit_answer():
    """
    Submits ONE student's answer for ONE question. Grading is triggered
    separately via /grade — this route only records the answer (typed or
    OCR-extracted text), so the examiner can review/correct OCR output
    before grading runs.
    """
    data = request.get_json()
    question_id = data.get("question_id")
    student_name = data.get("student_name")
    matric_no = data.get("matric_no")
    answer_text = data.get("answer_text")
    input_type = data.get("input_type", "typed")

    if not all([question_id, student_name, matric_no, answer_text]):
        return jsonify({"error": "question_id, student_name, matric_no, and answer_text are required"}), 400

    db = get_db()
    cursor = db.cursor()

    cursor.execute("SELECT id FROM students WHERE matric_no = %s", (matric_no,))
    student = cursor.fetchone()
    if not student:
        cursor.execute(
            "INSERT INTO students (student_name, matric_no) VALUES (%s, %s) RETURNING id",
            (student_name, matric_no)
        )
        student_id = cursor.fetchone()["id"]
    else:
        student_id = student["id"]

    cursor.execute(
        """INSERT INTO student_answers (question_id, student_id, answer_text, input_type)
           VALUES (%s, %s, %s, %s) RETURNING id""",
        (question_id, student_id, answer_text, input_type)
    )
    answer_id = cursor.fetchone()["id"]
    db.commit()

    return jsonify({"id": answer_id, "status": "submitted"}), 201


@app.route("/api/answers/<int:answer_id>/grade", methods=["POST"])
@token_required
@role_required("examiner")
def grade_answer_route(answer_id):
    """
    Triggers grading for ONE student answer. Retrieves grounded context from
    the knowledge graph, then calls ExamGrader. This is called one answer at
    a time by design — the examiner reviews each result before moving on.
    """
    db = get_db()
    cursor = db.cursor()

    cursor.execute(
        """SELECT sa.answer_text, q.question_text, q.mark_scheme_text, q.exam_session_id, es.course_id,
                  s.student_name, s.matric_no
           FROM student_answers sa
           JOIN questions q ON sa.question_id = q.id
           JOIN exam_sessions es ON q.exam_session_id = es.id
           JOIN students s ON sa.student_id = s.id
           WHERE sa.id = %s""",
        (answer_id,)
    )
    row = cursor.fetchone()
    if not row:
        return jsonify({"error": "Answer not found"}), 404

    retriever = GraphRetriever(driver=neo4j_driver)
    retrieved_context = retriever.retrieve_context(
        course_id=row["course_id"],
        question_text=row["question_text"],
        mark_scheme_text=row["mark_scheme_text"]
    )

    grader = ExamGrader()
    result = grader.grade_answer(
        question_text=row["question_text"],
        mark_scheme_text=row["mark_scheme_text"],
        retrieved_context=retrieved_context,
        student_answer=row["answer_text"]
    )

    cursor.execute(
        """INSERT INTO grading_outputs
           (student_answer_id, llm_suggested_mark, concepts_addressed, gaps_identified, justification_text)
           VALUES (%s, %s, %s, %s, %s) RETURNING id""",
        (
            answer_id,
            result.get("suggested_mark"),
            psycopg2.extras.Json(result.get("concepts_addressed", [])),
            psycopg2.extras.Json(result.get("gaps_identified", [])),
            result.get("justification", "")
        )
    )
    grading_output_id = cursor.fetchone()["id"]
    db.commit()

    return jsonify({
        "grading_output_id": grading_output_id,
        "student_name": row["student_name"],
        "matric_no": row["matric_no"],
        "answer_text": row["answer_text"],
        **result
    }), 200


@app.route("/api/answers/<int:answer_id>/result", methods=["GET"])
@token_required
def get_grading_result(answer_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "SELECT * FROM grading_outputs WHERE student_answer_id = %s ORDER BY generated_at DESC LIMIT 1",
        (answer_id,)
    )
    result = cursor.fetchone()
    if not result:
        return jsonify({"error": "No grading result found for this answer yet"}), 404
    return jsonify(result), 200


@app.route("/api/answers/<int:answer_id>/finalize", methods=["POST"])
@token_required
@role_required("examiner")
def finalize_mark(answer_id):
    """
    Examiner reviews the AI suggestion and records the FINAL mark.
    This is always a deliberate human action — never auto-submitted.
    """
    data = request.get_json()
    final_mark = data.get("final_mark")
    examiner_note = data.get("examiner_note", "")
    grading_output_id = data.get("grading_output_id")

    if final_mark is None or grading_output_id is None:
        return jsonify({"error": "final_mark and grading_output_id are required"}), 400

    db = get_db()
    cursor = db.cursor()

    cursor.execute("SELECT student_id FROM student_answers WHERE id = %s", (answer_id,))
    answer = cursor.fetchone()
    if not answer:
        return jsonify({"error": "Answer not found"}), 404

    cursor.execute(
        """INSERT INTO final_marks (grading_output_id, student_id, final_mark, examiner_note)
           VALUES (%s, %s, %s, %s) RETURNING id""",
        (grading_output_id, answer["student_id"], final_mark, examiner_note)
    )
    final_mark_id = cursor.fetchone()["id"]
    db.commit()

    return jsonify({"id": final_mark_id, "final_mark": final_mark, "status": "finalized"}), 201


# =====================================================================
# HEALTH CHECK
# =====================================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)