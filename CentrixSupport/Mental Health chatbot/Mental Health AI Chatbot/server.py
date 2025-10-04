from flask import Flask, render_template, request, jsonify, send_from_directory
from groq import Groq, BadRequestError
import os
import sys
import time
import traceback
import json
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import logging

# Add current directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import core functions (ensure main.py defines these)
from main import (
    detect_emotion,
    detect_high_risk,
    get_rag_context,
    SYSTEM_PROMPT
)

# Import utilities (use aliases to avoid clashing with Flask 'app')
import app.self_care_plan as self_care_plan
import app.atreective as atreective
import app.task_manager as task_manager
import app.mental_exercises as mental_exercises

# Load environment
load_dotenv()

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'png', 'jpg', 'jpeg'}
CONVERSATION_FOLDER = os.path.join(BASE_DIR, 'conversations')
DEFAULT_CONVERSATION_FILE = os.path.join(CONVERSATION_FOLDER, 'conversation.json')

# Flask app
app = Flask(
    __name__,
    static_url_path='/static',
    static_folder=os.path.join(BASE_DIR, 'static'),
    template_folder=os.path.join(BASE_DIR, 'templates')
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CONVERSATION_FOLDER'] = CONVERSATION_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024  # 150 MB

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERSATION_FOLDER, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

# API key for Groq (from .env -> license)
api_key = os.getenv("license")
if not api_key:
    logging.error("❌ API key not found in .env (license)")
    raise EnvironmentError("API key not found. Please set 'license' in .env.")

MODEL_NAME = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
client = Groq(api_key=api_key)

# ===== Helpers =====
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def trim_history(messages, max_len=10):
    """Keep the system prompt (first) and last `max_len` messages."""
    if not messages:
        return [{"role": "system", "content": SYSTEM_PROMPT}]
    system_msg = messages[0]
    tail = messages[-max_len:] if len(messages) > max_len else messages[1:]
    return [system_msg] + tail


def get_conversation_filepath(session_name=None):
    if session_name:
        safe_name = secure_filename(session_name)
        if not safe_name.endswith('.json'):
            safe_name += '.json'
        return os.path.join(app.config['CONVERSATION_FOLDER'], safe_name)
    return DEFAULT_CONVERSATION_FILE


def load_conversation(session_name=None):
    filepath = get_conversation_filepath(session_name)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            logging.warning(f"Failed to load conversation {filepath}: {e}")
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def save_conversation(history, session_name=None):
    filepath = get_conversation_filepath(session_name)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save conversation {filepath}: {e}")


def remove_file_safely(filepath):
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            logging.info(f"Removed uploaded file: {filepath}")
    except Exception as e:
        logging.warning(f"Failed to remove file {filepath}: {e}")


# ===== Routes =====
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/chat')
def index():
    return render_template('index.html')


@app.route('/learn')
def learn_more():
    return render_template('learn_more.html')


@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/resource')
def resource():
    return render_template('resource.html')


@app.route('/contact')
def contact():
    return render_template('contect.html')


@app.route('/help')
def help():
    return render_template('help.html')


# ==== Upload endpoint ====
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Accepts one or more files, saves them, and pre-processes into RAG vector store
    so they're ready before any query is asked.
    """
    try:
        files = request.files.getlist('file')
        if not files or len(files) == 0:
            return jsonify({"success": False, "error": "No files uploaded."}), 400

        saved_files = []
        for file in files:
            if not file or file.filename == '':
                continue
            if not allowed_file(file.filename):
                logging.info(f"Skipped unsupported file type: {file.filename}")
                continue

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # avoid overwriting
            if os.path.exists(filepath):
                base, ext = os.path.splitext(filename)
                filename = f"{base}_{int(time.time())}{ext}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)
            logging.info(f"File saved: {filepath}")
            saved_files.append(filepath)

        if not saved_files:
            return jsonify({"success": False, "error": "No valid files uploaded."}), 400

        # === NEW: preprocess into embeddings right now ===
        try:
            # this builds embeddings in advance, so /search is instant
            _ = get_rag_context(saved_files, "init preprocessing")
            logging.info(f"Preprocessed files into RAG store: {saved_files}")
        except Exception as e:
            logging.error(f"Preprocessing failed: {traceback.format_exc()}")
            return jsonify({"success": False, "error": f"Preprocessing failed: {str(e)}"}), 500

        return jsonify({
            "success": True,
            "filepaths": saved_files,
            "filepath": saved_files[0],
            "file_path": saved_files[0]
        })

    except Exception as e:
        logging.error(f"Upload error: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


# ==== Search (RAG) endpoint ====
@app.route('/search', methods=['POST'])
def search():
    """
    Expects JSON:
    {
      "query": "text",
      "filepaths": ["/abs/path/uploads/a.pdf"] OR
      "filepath": "/abs/path/uploads/a.pdf" OR
      "file_path": "/abs/path/uploads/a.pdf",
      "session_name": "..."
    }
    """
    try:
        start_time = time.time()
        data = request.get_json(force=True)

        question = data.get('query') or data.get('question')
        raw_filepaths = data.get('filepaths') or data.get('filepath') or data.get('file_path') or []
        if isinstance(raw_filepaths, str):
            raw_filepaths = [raw_filepaths]

        session_name = data.get('session_name')

        if not question:
            return jsonify({"success": False, "error": "Query is required"}), 400

        logging.info(f"Received query: {question}")
        if raw_filepaths:
            logging.info(f"Files provided: {raw_filepaths}")

        # Load conversation history
        conversation_history = load_conversation(session_name)

        # Emotion detection (best-effort)
        emotion = None
        try:
            emotion = detect_emotion(question)
        except Exception as e:
            logging.warning(f"Emotion detection failed: {e}")

        # High-risk detection
        try:
            if detect_high_risk(question):
                return jsonify({
                    "success": True,
                    "emotion_detected": emotion,
                    "response": (
                        "🚨 Crisis detected. You're not alone.\n"
                        "📞 Please call a helpline: +91 9152987821 or 91-84229 84528.\n"
                        "You matter. Support is here for you."
                    )
                })
        except Exception as e:
            logging.warning(f"High-risk detection failed: {e}")

        # Build RAG context using all provided files at once (if any)
        rag_context = ""
        used_filepaths = []
        if raw_filepaths:
            # Convert relative paths (frontend may send e.g. "uploads/foo.pdf") to absolute server paths
            abs_filepaths = [
                fp if os.path.isabs(fp) else os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(fp))
                for fp in raw_filepaths
            ]
            # Keep only files that exist on disk
            abs_filepaths = [fp for fp in abs_filepaths if os.path.exists(fp)]
            if abs_filepaths:
                try:
                    # IMPORTANT: get_rag_context should accept list-of-paths OR single string (we pass list here)
                    rag_context = get_rag_context(abs_filepaths, question) or ""
                    used_filepaths.extend(abs_filepaths)
                except Exception as e:
                    logging.error(f"RAG context error: {traceback.format_exc()}")

        # Compose messages for LLM: keep existing conversation, add RAG context as system msg, then user query
        messages = conversation_history.copy()
        if rag_context and rag_context.strip():
            messages.append({
                "role": "system",
                "content": f"Use the following context to answer the question:\n\n---\n{rag_context}\n---"
            })
        messages.append({"role": "user", "content": question})

        trimmed_messages = trim_history(messages)

        # Call Groq LLM
        # Try the configured model first, then fall back to any configured fallback models
        tried_models = []
        last_error_msg = None
        fallback_env = os.getenv('GROQ_FALLBACK_MODELS', 'llama-3.1-8b-instant,llama3-8b-8192')
        fallback_models = [m.strip() for m in fallback_env.split(',') if m.strip()]
        candidate_models = [MODEL_NAME] + [m for m in fallback_models if m != MODEL_NAME]

        for model_candidate in candidate_models:
            try:
                logging.info(f"Sending request to Groq LLM with model={model_candidate}...")
                completion = client.chat.completions.create(
                    model=model_candidate,
                    messages=trimmed_messages,
                    stream=False
                )
                assistant_reply = completion.choices[0].message.content.strip()
                logging.info(f"Groq response received using model={model_candidate}")
                break
            except BadRequestError as be:
                # Specific model-related rejection; record and try next candidate
                logging.warning(f"Groq BadRequest for model {model_candidate}: {be}")
                msg = str(be)
                try:
                    resp = getattr(be, 'response', None)
                    if isinstance(resp, dict):
                        msg = resp.get('error', {}).get('message', msg)
                except Exception:
                    pass
                last_error_msg = msg
                tried_models.append((model_candidate, msg))
                # If the message doesn't indicate a model issue, stop retrying
                if 'model_not_found' not in msg and 'model' not in msg.lower():
                    break
            except Exception as e:
                logging.error(f"LLM request failed for model {model_candidate}: {traceback.format_exc()}")
                msg = str(e)
                try:
                    resp = getattr(e, 'response', None)
                    if isinstance(resp, dict):
                        msg = resp.get('error', {}).get('message', msg)
                except Exception:
                    pass
                last_error_msg = msg
                tried_models.append((model_candidate, msg))
                # For generic errors, don't try many fallbacks — stop after first failure
                break

        if 'assistant_reply' not in locals():
            # All attempts failed — return the most helpful message we collected
            logging.error(f"All Groq model attempts failed: {tried_models}")
            msg = last_error_msg or 'Unknown Groq error'
            return jsonify({"success": False, "response": f"Groq error: {msg}", "tried_models": tried_models}), 500

        # Persist conversation
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        save_conversation(conversation_history, session_name)

        # Cleanup (single-use uploaded files) — comment this block out if you want files to persist
        for fp in used_filepaths:
            try:
                remove_file_safely(fp)
            except Exception:
                logging.warning(f"Failed to remove used file: {fp}")

        total_time = round(time.time() - start_time, 1)
        return jsonify({
            "success": True,
            "emotion_detected": emotion,
            "response": f"{assistant_reply}\n\n⏳ Time: {total_time}s"
        })

    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({"success": False, "response": f"❌ Internal Server Error: {str(e)}"}), 500


# ==== Conversations endpoints ====
@app.route('/history', methods=['GET'])
def get_history():
    session_name = request.args.get('session_name')
    return jsonify(load_conversation(session_name))


@app.route('/sessions', methods=['GET'])
def list_sessions():
    files = [f for f in os.listdir(app.config['CONVERSATION_FOLDER']) if f.endswith('.json')]
    return jsonify({"sessions": files})


# ==== Utility runners ====
@app.route('/run/self-care')
def self_care():
    try:
        self_care_plan.generate_self_care_plan()
        return jsonify({"status": "✅ Self-care plan executed"})
    except Exception as e:
        logging.error(f"Self-care runner error: {e}")
        return jsonify({"status": "❌ Error", "error": str(e)}), 500


@app.route('/run/checkin')
def checkin():
    try:
        progress = atreective.load_progress()
        atreective.send_check_in(progress)
        return jsonify({"status": "✅ Check-in complete"})
    except Exception as e:
        logging.error(f"Check-in runner error: {e}")
        return jsonify({"status": "❌ Error", "error": str(e)}), 500


@app.route('/run/tasks')
def tasks():
    try:
        task_manager.todo_menu()
        return jsonify({"status": "✅ Task manager executed"})
    except Exception as e:
        logging.error(f"Tasks runner error: {e}")
        return jsonify({"status": "❌ Error", "error": str(e)}), 500


@app.route('/run/exercises')
def exercises():
    try:
        mental_exercises.main()
        return jsonify({"status": "✅ Mental exercises executed"})
    except Exception as e:
        logging.error(f"Exercises runner error: {e}")
        return jsonify({"status": "❌ Error", "error": str(e)}), 500


# ==== File serving ====
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)


# ==== Run ====
if __name__ == '__main__':
    logging.info("✅ Flask starting at http://localhost:5000 (debug reloader disabled)")
    logging.info(f"🔐 API Key Loaded: {'Yes' if api_key else 'No'}")
    try:
        # Turn off the reloader to avoid child-process crashes caused by native libs
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception:
        logging.error("Flask failed to start:\n" + traceback.format_exc())
