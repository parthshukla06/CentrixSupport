from groq import Groq
import os
import time
from dotenv import load_dotenv
import random
from collections import defaultdict
from app.content_retrieval import get_rag_context
import app.atreective
import app.mental_exercises
import app.task_manager
import app.self_care_plan

# ========== Load Environment ==========
load_dotenv()
api_key = os.getenv('license')
if not api_key:
    raise EnvironmentError("❌ 'license' key (Groq API Key) not found in .env")

client = Groq(api_key=api_key)

# ========== Constants & Globals ==========
conversation_history = []

SYSTEM_PROMPT = (
    "You are a compassionate, professional, and licensed mental health counselor.\n\n"
    "🔒 STRICT BEHAVIOR:\n"
    "- If the user's message is **not related to mental health, emotional wellness, or medical concerns**, respond with **only this exact sentence**:\n"
    "'That’s an interesting question! My focus here is on mental and emotional well-being, so I may not be the best fit to guide you on that. But if there’s anything on your mind or heart you’d like to talk about, I’m here for you.'\n"
    "- ⛔️ Do **not** explain, analyze, or generate any content beyond this if the topic is unrelated.\n\n"
    "✅ If the input **is related to mental or emotional health**, respond with:\n"
    "- Warm, validating tone\n"
    "- Practical, evidence-based strategies\n"
    "- Non-judgmental and encouraging support\n"
    "- Tailor your emotional tone based on the user’s dominant emotional state (detected from keywords)\n"
    "🧠 EMOTION TONE GUIDANCE (internal only):\n"
    "- Use detected emotion-related words to **adjust your tone**, **not the content**.\n"
    "- Guide tone as follows:\n"
    "   • If 'overwhelmed', 'pressure', or 'burnout': speak gently and reassuringly, prioritize calming techniques.\n"
    "   • If 'sad', 'empty', 'lonely': use comforting and empathetic tone, focus on connection and self-kindness.\n"
    "   • If 'angry', 'rage', 'mad': validate frustration and suggest healthy emotional expression.\n"
    "   • If 'anxious', 'panic', 'scared': speak soothingly and provide grounding or calming suggestions.\n\n"
    "🚨 CRISIS RESPONSE:\n"
    "- If user mentions suicidal thoughts or self-harm:\n"
    "  1. Respond: 'I'm really sorry you're feeling this way, but you're not alone. There are people who care about you.'\n"
    "  2. Provide helpline: 'Please, don't go through this alone. You can reach out to the National Suicide Prevention Lifeline at +91 9152987821 or 91-84229 84528.'\n"
    "  3. Encourage immediate help and remind the user they deserve support.\n\n"
    "📌 RECOMMENDATION SYSTEM:\n"
    "- At the end of each mental health-related response, **if appropriate**, provide 1–3 gentle and relevant recommendations.\n"
    "- These may include: helpful exercises (e.g., breathing, journaling), mindfulness apps, helplines, books, videos, or support group suggestions.\n"
    "- Keep recommendations brief, supportive, and optional, introduced with: 'You might find this helpful:' or 'If you're open to it, here's something that may support you:'.\n\n"
    "⚠️ Never attempt to diagnose. Your role is to support, guide, and comfort using professional best practices."
    "If you feel recommendations are needed this time, prioritize suggestions for self-care, check-ins, exercises, or mental health support."
)

conversation_history.append({"role": "system", "content": SYSTEM_PROMPT})

# ========== Emotion Detection ==========
emotion_keywords = {
    "stress": ["overwhelmed", "pressure", "tired", "burnout", "anxious", "exhausted", "panic", "stress"],
    "sadness": ["sad", "empty", "lonely", "depressed", "cry", "hopeless", "heartbroken", "grief"],
    "anger": ["angry", "hate", "furious", "rage", "mad", "annoyed", "resentful"],
    "fear": ["scared", "afraid", "terrified", "worried", "nervous"],
    "guilt": ["guilty", "shame", "blame", "regret"],
    "worthlessness": ["worthless", "useless", "not", "enough", "failure"],
    "suicidal": ["want", "to", "die", "end", "my", "life", "kill", "myself", "suicidal", "can't", "go", "on"]
}

suicidal_keywords = [
    "suicidal", "kill myself", "want to die", "end it all", "hurt myself",
    "end my life", "not worth living", "i'm done", "life is meaningless"
]

# ========== Utility Functions ==========

def detect_emotion(text):
    text = text.lower()
    for emotion, keywords in emotion_keywords.items():
        if any(k in text for k in keywords):
            return emotion
    return None

def detect_high_risk(msg):
    msg = msg.lower()
    return any(k in msg for k in suicidal_keywords)

def recommend_resources(query):
    if "anxiety" in query.lower():
        return os.getenv("recommend_resources_URl", "")
    elif "depression" in query.lower():
        return os.getenv("resources_URl", "")
    return ""

def rephrase_negative_thought(negative_thought):
    return (
        "It's okay to feel this way. Here's a reframe: "
        "“I'm having a hard time, but I’m trying my best. I matter, and I’m not alone.”"
    )

def analyze_sentiment(text):
    negative_words = ["sad", "angry", "hopeless", "worthless", "hate", "useless"]
    score = sum(text.lower().count(w) for w in negative_words)
    if score >= 3:
        return "⚠️ You're using more negative words than usual. Try to focus on even small positives."
    elif score == 0:
        return "✨ Your tone today feels hopeful and resilient."
    else:
        return "Your tone shows mixed feelings. Keep expressing yourself—journaling helps."

def generate_summary_conclusion_recommendations(conversation):
    base_prompt = (
        "Based on the following conversation between a user and a compassionate mental health counselor, "
        "if necessary, provide 1–3 gentle recommendations. If the conversation is too short, skip it.\n\n"
    )
    for msg in conversation:
        base_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful, compassionate mental health assistant."},
            {"role": "user", "content": base_prompt}
        ],
        model="llama3-8b-8192",
        stream=False,
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

def respond_to_user(message, rag_context=None):
    try:
        messages = conversation_history.copy()
        if rag_context:
            messages.append({"role": "system", "content": f"[File Context]: {rag_context}"})
            print("📄 Using uploaded file for context-enhanced response.")
        messages.append({"role": "user", "content": message})

        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            stream=True,
            temperature=0.7,
            max_tokens=1000
        )

        response_tokens = ""
        for chunk in chat_completion:
            content = getattr(chunk.choices[0].delta, "content", "")
            if content:
                print(content, end="", flush=True)
                response_tokens += content
                time.sleep(0.02)
        print()

        conversation_history.append({"role": "user", "content": message})
        conversation_history.append({"role": "assistant", "content": response_tokens})

        if rag_context and conversation_history[-3]["content"].startswith("[File Context]:"):
            conversation_history.pop(-3)

        summary = generate_summary_conclusion_recommendations(conversation_history)
        print("\n" + summary + "\n")
        conversation_history.append({"role": "assistant", "content": summary})

    except Exception as e:
        print(f"❌ Error during chat response: {e}")


# ========== CLI Mode ==========

if __name__ == "__main__":
    file_path = input("📁 Upload a file for RAG (or press Enter to skip): ").strip()
    use_rag = os.path.isfile(file_path)

    progress = app.atreective.load_progress()
    if app.atreective.should_send_check_in(progress):
        app.atreective.send_check_in(progress)

    print("\n💬 Welcome to the Mental Health AI Chatbot. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("💚 Take care! You're doing better than you think.")
            break

        if user_input.lower() == "self-care":
            app.self_care_plan.generate_self_care_plan()
            continue

        if user_input.lower() == "checkin":
            app.atreective.send_check_in(progress)
            continue

        if user_input.lower() == "journal":
            entry = input("Write your journal entry: ")
            print("📝 Reflecting... You're making space for healing.")
            continue

        if user_input.lower() == "tasks":
            app.task_manager.todo_menu()
            continue

        if user_input.lower() == "exercises":
            app.mental_exercises.main()
            continue

        emotion = detect_emotion(user_input)
        if emotion:
            print(f"\n🔍 Detected Emotion: {emotion}")

        if detect_high_risk(user_input):
            print("\n🚨 Crisis detected. You're not alone.")
            print("📞 Call 91-9152987821 or 91-84229 84528 immediately.")
            continue

        rag_context = get_rag_context(file_path, user_input) if use_rag else None
        respond_to_user(user_input, rag_context)
