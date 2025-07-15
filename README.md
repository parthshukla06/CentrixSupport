#  Centrix Support: AI Mental Health Support Chatbot

 Centrix Support is an AI-powered mental health support chatbot built using **Flask** and **Groq's LLaMA 3** model. Designed for empathy, safety, and accessibility, it helps users with:

*  Emotional support & conversation
*  Context-aware answers via uploaded documents (RAG)
*  Crisis detection & helpline guidance
*  Self-care plans, check-ins, tasks, and mental exercises

---

## 🚀 Features

###  AI Chat Support

* Uses Groq's **LLaMA 3** (`llama3-8b-8192`) for intelligent conversation.
* Maintains **chat history** per session.

###  Emotion & Risk Detection

* Detects emotional tone in user inputs.
* Responds to high-risk queries with helpline support.

###  Document-Based Q\&A (RAG)

* Upload files (PDF, TXT, DOCX, JPG, PNG).
* Extracts relevant context to answer user queries.

###  Self-Care Toolkit

* Self-care plan generator
* Check-in with past progress
* Task management and mental exercises

---

## 🛠️ Tech Stack

| Layer      | Tech                                   |
| ---------- | -------------------------------------- |
| Backend    | Flask, Python                          |
| LLM        | [Groq](https://groq.com/) - LLaMA 3    |
| Frontend   | HTML, Tailwind CSS, JavaScript         |
| Embeddings | Custom text chunking + semantic search |
| Deployment | Render / Railway / VPS ready           |

---

## 📂 Project Structure

```
├── app.py                # Flask entry point
├── main.py               # Core logic (emotion, RAG, summaries)
├── templates/            # HTML templates
├── static/               # JS, CSS, assets
├── uploads/              # Uploaded user files
├── conversations/        # Session-based chat logs
├── .env                  # API keys (excluded from Git)
```

---

## ⚙️ Setup Instructions

### 1. Clone Repo

```bash
git clone https://github.com/yourusername/mindwell-chatbot.git
cd mindwell-chatbot
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Create `.env` File

```env
license=your_groq_api_key_here
recommend_resources_URl=https://www.headspace.com/meditation/anxiety
resources_URl=https://www.sleepfoundation.org/sleep-hygiene
FLASK_ENV=production
```

### 4. Run Locally

```bash
python app.py
```

Visit [http://localhost:5000](http://localhost:5000)

---

## 🧪 Example Commands

```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How can I manage stress?", "filepaths": [], "session_name": "demo-session"}'
```

---

## 📦 Deployment

* Production-ready with `gunicorn`:

```bash
gunicorn app:app --bind 0.0.0.0:5000
```

* Add Nginx or host on Render / Railway / Docker.

---

## 🛡️ Disclaimer

> This chatbot is **not a replacement for professional therapy**. It is intended to support, not diagnose or treat mental health conditions. For emergencies, always contact a certified medical or mental health professional.

---

## 👨‍💻 Author

**Priyanshu Shukla**
AI Developer | Mental Wellness Advocate

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

* [Groq](https://groq.com/) for blazing-fast inference
* [OpenAI](https://openai.com/) for inspiration in safe AI interfaces
* [Headspace](https://headspace.com/) and [Sleep Foundation](https://sleepfoundation.org/) for mental health resources

---
