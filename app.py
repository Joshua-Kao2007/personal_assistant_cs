from flask import Flask, request, jsonify, render_template, Response
from openai import OpenAI
import os
import cv2
import base64
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

# --------------------------
# ✅ SETUP: OpenAI + MongoDB
# --------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MONGO_URI = os.getenv("MONGO_URI") 
mongo_client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
db = mongo_client["emotion_db"]
collection = db["emotion_entries"]
study_collection = db["study_sessions"]

# --------------------------
# ✅ Webcam Stream
# --------------------------

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --------------------------
# ✅ Routes
# --------------------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/emotion")
def emotion_page():
    return render_template("emotion.html")

@app.route("/study")
def study_page():
    return render_template("study.html")

@app.route("/study_api", methods=["POST"])
def study_api():
    data = request.json
    topic = data.get("topic")
    duration = data.get("duration")
    notes = data.get("notes")

    prompt = (
    f"Give a short, concise 1 to 2 sentence summary about the topic '{topic}' using these notes: {notes}.\n"
    "Keep it under 2 sentences. Focus on clarity and brevity."
)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        summary = response.choices[0].message.content.strip()
        entry = {
            "topic": topic,
            "notes": notes,
            "duration": duration,
            "summary": summary,
            "timestamp": datetime.utcnow()
        }
        study_collection.insert_one(entry)
        return jsonify({"summary": summary})
    except Exception as e:
        print("🔴 Study API Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/study_history")
def study_history():
    entries = study_collection.find().sort("timestamp", -1)
    result = [
        {
            "topic": entry["topic"],
            "duration": entry["duration"],
            "summary": entry["summary"],
            "timestamp": entry["timestamp"].strftime("%Y-%m-%d %H:%M")
        }
        for entry in entries
    ]
    return jsonify(result)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/chat_api", methods=["POST"])
def chat_api():
    user_input = request.json.get("message")
    messages = request.json.get("messages", [])
    messages.append({"role": "user", "content": user_input})
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply, "messages": messages})
    except Exception as e:
        print("🔴 OpenAI Error:", e)
        return jsonify({"error": str(e)}), 500

# --------------------------
# ✅ Emotion Capture + Save
# --------------------------
@app.route("/capture_frame", methods=["POST"])
def capture_frame():
    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture frame"}), 500

    _, buffer = cv2.imencode(".jpg", frame)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")

    try:
        vision_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "1. Give ONLY a single word in English that summarizes the person's mood/emotion.\n"
                                "2. Then on the next line, write ONE sentence that briefly describes the person's expression, pose, or vibe.\n"
                                "Format:\n"
                                "Emotion: Happy\n"
                                "Description: The person is smiling confidently with a relaxed demeanor."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{jpg_as_text}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=150
        )

        full_response = vision_response.choices[0].message.content.strip()

        # Default values
        emotion_word = "Unknown"
        description = "No description provided."

        # Parse response safely
        if "Emotion:" in full_response and "Description:" in full_response:
            parts = full_response.split("Description:")
            emotion_line = parts[0].replace("Emotion:", "").strip()
            description = parts[1].strip()
            emotion_word = emotion_line.capitalize()
        elif "Emotion:" in full_response:
            emotion_word = full_response.replace("Emotion:", "").strip().capitalize()

        # Save to MongoDB
        entry = {
            "image_base64": jpg_as_text,
            "emotion": emotion_word,
            "keyword": emotion_word,
            "description": description,
            "timestamp": datetime.utcnow()
        }
        collection.insert_one(entry)

        return jsonify({"emotion": emotion_word})

    except Exception as e:
        print("🔴 GPT Vision Error:", e)
        return jsonify({"error": str(e)}), 500


# --------------------------
# ✅ Return Saved Cards
# --------------------------

@app.route("/emotion_entries")
def get_emotion_entries():
    entries = collection.find().sort("timestamp", -1)
    result = [
        {
            "image": f"data:image/jpeg;base64,{entry['image_base64']}",
            "emotion": entry["emotion"],
            "keyword": entry.get("keyword", "Emotion"),
            "description": entry.get("description", ""),
            "timestamp": entry.get("timestamp").strftime("%Y-%m-%d %H:%M")
        }
        for entry in entries
    ]
    return jsonify(result)

# --------------------------
# ✅ Run the App
# --------------------------

if __name__ == "__main__":
    app.run(debug=True)
