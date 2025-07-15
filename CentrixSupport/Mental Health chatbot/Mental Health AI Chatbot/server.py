from flask import Flask, render_template, request, jsonify, send_from_directory
from groq import Groq
import os
import sys
import time
import traceback
import json
import threading
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from main import (
    detect_emotion,
    detect_high_risk,
    generate_summary_conclusion_recommendations,
    get_rag_context,
    SYSTEM_PROMPT,
    recommend_resources,
)
import app.self_care_plan
import app.atreective
import app.task_manager
import app.mental_exercises

# ========== Configuration ==========
load_dotenv()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'png', 'jpg', 'jpeg'}
CONVERSATION_FOLDER = 'conversations'
DEFAULT_CONVERSATION_FILE = os.path.join(CONVERSATION_FOLDER, 'conversation.json')

app = Flask(
    __name__,
    static_url_path='/static',
    static_folder='static',
    template_folder='templates'
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CONVERSATION_FOLDER'] = CONVERSATION_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERSATION_FOLDER, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

api_key = os.getenv("license")
if not api_key:
    logging.error("❌ API key not found in .env file. Please set 'license' in your .env.")
    raise EnvironmentError("API key not found. Please set 'license' in your .env file.")

client = Groq(api_key=api_key)

# ========== Helper Functions ==========

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def trim_history(messages, max_len=10):
    if not messages:
        return [{"role": "system", "content": SYSTEM_PROMPT}]
    system_msg = messages[0]
    return [system_msg] + messages[-max_len:]

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
                return json.load(f)
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

def background_postprocessing(history, query):
    try:
        generate_summary_conclusion_recommendations(history)
        recommend_resources(query)
        logging.info("Summary and resources generated.")
    except Exception as e:
        logging.warning(f"Postprocessing failed: {e}")

def remove_file_safely(filepath):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logging.info(f"Removed uploaded file: {filepath}")
    except Exception as e:
        logging.warning(f"Failed to remove file {filepath}: {e}")

# ========== Routes ==========

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

@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files.getlist('file')
    if not files:
        return jsonify({"success": False, "error": "No files uploaded."}), 400

    saved_files = []

    for file in files:
        if not allowed_file(file.filename):
            logging.info(f"Skipped unsupported file type: {file.filename}")
            continue

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if os.path.exists(filepath):
            base, ext = os.path.splitext(filename)
            timestamp = int(time.time())
            filename = f"{base}_{timestamp}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
            saved_files.append(filepath)
            logging.info(f"File saved: {filepath}")
        except Exception as e:
            logging.error(f"Failed to save file: {e}")
            return jsonify({"success": False, "error": f"Failed to save file: {str(e)}"}), 500

    if not saved_files:
        return jsonify({"success": False, "error": "No valid files uploaded."}), 400

    return jsonify({"success": True, "filepaths": saved_files})

@app.route('/search', methods=['POST'])
def search():
    try:
        start_time = time.time()
        data = request.get_json(force=True)

        question = data.get('query')
        filepaths = data.get('filepaths', [])
        session_name = data.get('session_name')

        if not question:
            return jsonify({"success": False, "error": "Query is required"}), 400

        logging.info(f"Received query: {question}")
        if filepaths:
            logging.info(f"Files used: {filepaths}")

        conversation_history = load_conversation(session_name)
        emotion = detect_emotion(question)

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

        rag_context = ""
        for filepath in filepaths:
            if filepath and os.path.exists(filepath):
                try:
                    rag_context += get_rag_context(filepath, question) + "\n"
                except Exception as e:
                    logging.error(f"RAG context error for {filepath}: {e}")

        messages = conversation_history.copy()
        if rag_context.strip():
            messages.append({
                "role": "system",
                "content": f"Use the following context to answer the question:\n\n---\n{rag_context}\n---"
            })
        messages.append({"role": "user", "content": question})

        trimmed_messages = trim_history(messages)

        logging.info("Sending request to Groq LLM...")
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=trimmed_messages,
            stream=False
        )
        assistant_reply = completion.choices[0].message.content.strip()

        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        save_conversation(conversation_history, session_name)

        threading.Thread(target=background_postprocessing, args=(conversation_history.copy(), question), daemon=True).start()

        for filepath in filepaths:
            remove_file_safely(filepath)

        total_time = round(time.time() - start_time, 1)

        return jsonify({
            "success": True,
            "emotion_detected": emotion,
            "response": f"{assistant_reply}\n\n⏳ Time: {total_time}s"
        })

    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "emotion_detected": None,
            "response": f"❌ Internal Server Error: {str(e)}"
        }), 500

@app.route('/history', methods=['GET'])
def get_history():
    session_name = request.args.get('session_name')
    history = load_conversation(session_name)
    return jsonify(history)

@app.route('/sessions', methods=['GET'])
def list_sessions():
    files = [f for f in os.listdir(app.config['CONVERSATION_FOLDER']) if f.endswith('.json')]
    return jsonify({"sessions": files})

@app.route('/run/self-care', methods=['GET'])
def self_care():
    app.self_care_plan.generate_self_care_plan()
    return jsonify({"status": "✅ Self-care plan executed"})

@app.route('/run/checkin', methods=['GET'])
def checkin():
    progress = app.atreective.load_progress()
    app.atreective.send_check_in(progress)
    return jsonify({"status": "✅ Check-in complete"})

@app.route('/run/tasks', methods=['GET'])
def tasks():
    app.task_manager.todo_menu()
    return jsonify({"status": "✅ Task manager executed"})

@app.route('/run/exercises', methods=['GET'])
def exercises():
    app.mental_exercises.main()
    return jsonify({"status": "✅ Mental exercises executed"})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    logging.info("✅ Flask running at http://localhost:5000")
    logging.info(f"🔐 API Key Loaded: {'Yes' if api_key else 'No'}")
    app.run(debug=True)
