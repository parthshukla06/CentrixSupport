import json
import os
import random
from datetime import datetime, timedelta

# File to store user progress persistently
PROGRESS_FILE = "user_progress.json"

# ANSI color codes for terminal output (optional)
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

motivational_messages = [
    "🌟 Remember, every small step counts. You're doing great!",
    "💪 Keep going! Consistency is key to progress.",
    "🌈 It's okay to have setbacks — what's important is to keep trying.",
    "🎉 Celebrate your wins, no matter how small they seem.",
    "💚 Your mental health matters. Keep taking care of yourself!"
]

motivational_quotes = [
    "“You don’t have to control your thoughts. You just have to stop letting them control you.” — Dan Millman",
    "“Almost everything will work again if you unplug it for a few minutes, including you.” — Anne Lamott",
    "“Keep your face always toward the sunshine—and shadows will fall behind you.” — Walt Whitman",
    "“Self-care is how you take your power back.” — Lalah Delia",
    "“Sometimes the bravest and most important thing you can do is just show up.” — Brené Brown"
]

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    else:
        return {
            "days_checked_in": 0,
            "improvements": [],
            "setbacks": [],
            "last_check_in": None,
            "mood_history": []
        }

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=4)

def summarize_progress(progress):
    summary = f"{Colors.BOLD}{Colors.OKCYAN}📝 Your progress so far:{Colors.ENDC}\n"
    summary += f"📅 Days checked in: {Colors.OKGREEN}{progress['days_checked_in']}{Colors.ENDC}\n"
    if progress["improvements"]:
        summary += f"🎉 Improvements noticed: {Colors.OKGREEN}{', '.join(progress['improvements'])}{Colors.ENDC}\n"
    else:
        summary += f"🎉 Improvements noticed: {Colors.WARNING}None recorded yet{Colors.ENDC}\n"
    if progress["setbacks"]:
        summary += f"⚠️ Setbacks encountered: {Colors.FAIL}{', '.join(progress['setbacks'])}{Colors.ENDC}\n"
    else:
        summary += f"⚠️ Setbacks encountered: {Colors.OKGREEN}None{Colors.ENDC}\n"
    return summary

def get_encouragement(mood, progress):
    # Different encouragement based on mood and past history
    if mood == "good":
        return f"{Colors.OKGREEN}Awesome! Keep riding this positive momentum. 😊{Colors.ENDC}"
    elif mood == "okay":
        return (f"{Colors.OKBLUE}Thanks for checking in. Remember, ups and downs are normal — "
                "and you’re doing your best every day.{Colors.ENDC}")
    else:  # mood == bad
        if "feeling down" in progress.get("setbacks", []):
            return (f"{Colors.FAIL}I know it’s tough, but every new day is a fresh chance to heal. "
                    "You’re not alone. Reach out whenever you need support. 💙{Colors.ENDC}")
        else:
            return (f"{Colors.FAIL}I'm sorry you're feeling down. Remember, it’s okay to ask for help — "
                    "you deserve support and kindness. 💙{Colors.ENDC}")

def send_check_in(progress):
    print(f"\n{Colors.HEADER}{'='*50}{Colors.ENDC}")
    print(f"{Colors.BOLD}🧠 Time for your mental health check-in!{Colors.ENDC}\n")

    print(summarize_progress(progress))

    mood = ""
    valid_moods = ["good", "okay", "bad"]
    while mood not in valid_moods:
        mood = input(f"How are you feeling today? ({'/'.join(valid_moods)}): ").strip().lower()
        if mood not in valid_moods:
            print(f"{Colors.WARNING}Please enter one of: {', '.join(valid_moods)}{Colors.ENDC}")

    # Save mood to history
    progress["mood_history"].append({"date": datetime.now().isoformat(), "mood": mood})

    if mood == "good":
        if "feeling good" not in progress["improvements"]:
            progress["improvements"].append("feeling good")
    elif mood == "bad":
        if "feeling down" not in progress["setbacks"]:
            progress["setbacks"].append("feeling down")

    print("\n" + get_encouragement(mood, progress))
    print(f"\n💬 {random.choice(motivational_messages)}")

    # Motivational quote to end
    print(f"\n📝 Motivational Quote:\n{Colors.OKCYAN}{random.choice(motivational_quotes)}{Colors.ENDC}")

    # Update check-in stats
    progress["days_checked_in"] += 1
    progress["last_check_in"] = datetime.now().isoformat()

    save_progress(progress)
    print(f"\n{Colors.HEADER}{'='*50}{Colors.ENDC}\n")

def should_send_check_in(progress):
    if not progress["last_check_in"]:
        return True
    last = datetime.fromisoformat(progress["last_check_in"])
    now = datetime.now()
    return (now - last) > timedelta(days=1)

def main():
    print(f"{Colors.BOLD}{Colors.OKCYAN}🧠 Welcome to your Mental Health Check-In Chatbot!{Colors.ENDC}")

    progress = load_progress()

    if should_send_check_in(progress):
        send_check_in(progress)
    else:
        print(f"{Colors.OKGREEN}✅ You’ve already checked in recently. Keep up the good work!{Colors.ENDC}")

    while True:
        cmd = input(f"\nType {Colors.BOLD}'checkin'{Colors.ENDC} to do a progress check-in, or {Colors.BOLD}'quit'{Colors.ENDC} to exit: ").strip().lower()
        if cmd == "checkin":
            send_check_in(progress)
        elif cmd == "quit":
            print(f"\n{Colors.OKBLUE}Take care! Remember, your mental health is important. 💙{Colors.ENDC}")
            break
        else:
            print(f"{Colors.WARNING}Unknown command. Please type 'checkin' or 'quit'.{Colors.ENDC}")

if __name__ == "__main__":
    main()
