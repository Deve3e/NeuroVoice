import whisper
import pyttsx3
import requests
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import winsound
import os
import json
from datetime import datetime

# ---------------- PROFILE FUNCTION ----------------
def extract_and_save_personal_info(user_text, ai_response):
    """
    Extracts personal information from user input and saves it to profile.txt
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        profile_path = os.path.join(base_dir, "profile.txt")
        
        # Create or load existing profile
        if not os.path.exists(profile_path):
            profile_data = {
                "name": None,
                "age": None,
                "location": None,
                "job": None,
                "preferences": [],
                "interests": [],
                "other_info": [],
                "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Try to parse as JSON if it exists
                    if content.strip().startswith("{"):
                        profile_data = json.loads(content)
                    else:
                        profile_data = {
                            "name": None,
                            "age": None,
                            "location": None,
                            "job": None,
                            "preferences": [],
                            "interests": [],
                            "other_info": [],
                            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
            except:
                profile_data = {
                    "name": None,
                    "age": None,
                    "location": None,
                    "job": None,
                    "preferences": [],
                    "interests": [],
                    "other_info": [],
                    "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        
        # Extract personal information using simple patterns
        text_lower = user_text.lower()
        
        # Name extraction
        if "my name is" in text_lower or "i'm" in text_lower or "i am" in text_lower:
            if "my name is" in text_lower:
                parts = text_lower.split("my name is")
                if len(parts) > 1:
                    name = parts[1].strip().split()[0].capitalize()
                    if name and len(name) > 1:
                        profile_data["name"] = name
        
        # Age extraction
        if any(word in text_lower for word in ["i'm ", "i am ", "i'm "]):
            words = user_text.split()
            for i, word in enumerate(words):
                if word.isdigit() and 1 <= int(word) <= 120:
                    if i > 0 and words[i-1].lower() in ["years", "year", "old"]:
                        profile_data["age"] = int(word)
                        break
        
        # Location extraction
        if "live in" in text_lower or "i'm from" in text_lower or "from" in text_lower:
            if "live in" in text_lower:
                parts = user_text.split("live in")
                if len(parts) > 1:
                    location = parts[1].strip().split(",")[0].strip()
                    if location:
                        profile_data["location"] = location
        
        # Job extraction
        if "work as" in text_lower or "i'm a" in text_lower or "i am a" in text_lower or "job is" in text_lower:
            if "work as" in text_lower:
                parts = user_text.split("work as")
                if len(parts) > 1:
                    job = parts[1].strip().split(".")[0].strip()
                    if job:
                        profile_data["job"] = job
            elif any(phrase in text_lower for phrase in ["i'm a", "i am a"]):
                for phrase in ["i'm a", "i am a"]:
                    if phrase in text_lower:
                        parts = user_text.split(phrase)
                        if len(parts) > 1:
                            job = parts[1].strip().split(".")[0].split(",")[0].strip()
                            if job:
                                profile_data["job"] = job
                            break
        
        # Preferences extraction (like/enjoy)
        if "like" in text_lower or "enjoy" in text_lower or "love" in text_lower:
            for keyword in ["like", "enjoy", "love"]:
                if keyword in text_lower:
                    pattern_idx = text_lower.find(keyword)
                    preference = user_text[pattern_idx:].split(".")[0].strip()
                    if preference and preference not in profile_data["preferences"]:
                        profile_data["preferences"].append(preference)
        
        # Interests extraction (interested in)
        if "interested in" in text_lower:
            parts = user_text.split("interested in")
            if len(parts) > 1:
                interest = parts[1].strip().split(".")[0].strip()
                if interest and interest not in profile_data["interests"]:
                    profile_data["interests"].append(interest)
        
        # Save updated profile
        with open(profile_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(profile_data, indent=2, ensure_ascii=False))
        
        # Print if new info was found
        if any([profile_data.get("name"), profile_data.get("age"), profile_data.get("location"), profile_data.get("job")]):
            print("✓ Profile updated with personal information")
            
    except Exception as e:
        print(f"Profile update error: {str(e)}")

# ---------------- AI FUNCTION ----------------
def get_ai_response(prompt):
    try:
        # Load your memory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        memory_path = os.path.join(base_dir, "memory.txt")
        profile_path = os.path.join(base_dir, "profile.txt")

        if not os.path.exists(memory_path):
            with open(memory_path, "w", encoding="utf-8") as f:
                f.write("User memory:\n")

        with open(memory_path, "r", encoding="utf-8") as f:
            memory = f.read()

        # Load profile if it exists
        profile_info = ""
        if os.path.exists(profile_path):
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    profile_content = f.read()
                    if profile_content.strip().startswith("{"):
                        profile_data = json.loads(profile_content)
                        profile_lines = []
                        if profile_data.get("name"):
                            profile_lines.append(f"Name: {profile_data['name']}")
                        if profile_data.get("age"):
                            profile_lines.append(f"Age: {profile_data['age']}")
                        if profile_data.get("location"):
                            profile_lines.append(f"Location: {profile_data['location']}")
                        if profile_data.get("job"):
                            profile_lines.append(f"Job: {profile_data['job']}")
                        if profile_data.get("preferences"):
                            profile_lines.append(f"Preferences: {', '.join(profile_data['preferences'])}")
                        if profile_data.get("interests"):
                            profile_lines.append(f"Interests: {', '.join(profile_data['interests'])}")
                        if profile_lines:
                            profile_info = "User Profile:\n" + "\n".join(profile_lines) + "\n\n"
            except:
                pass

        full_prompt = f"""
You are a highly intelligent personal AI assistant.

{profile_info}
Use the following conversation history to answer better:

{memory}

User said: {prompt}

Respond clearly, intelligently, and helpfully.
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi",
                "prompt": full_prompt,
                "stream": False
            },
            timeout=60
        )

        data = response.json()

        if "response" in data:
            return data["response"]
        elif "error" in data:
            return f"AI error: {data['error']}"
        else:
            return "Unexpected AI response"

    except Exception as e:
        return f"Connection error: {str(e)}"


# ---------------- WHISPER ----------------
model = whisper.load_model("tiny")

# ---------------- TTS (FIXED) ----------------
import subprocess
import os

def speak(text):
    text = text.replace('"', '')  # clean text

    output_path = os.path.abspath("output.wav")

    command = f'''
    Add-Type -AssemblyName System.Speech;
    $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;
    $speak.SetOutputToWaveFile("{output_path}");
    $speak.Speak("{text}");
    $speak.Dispose();
    '''

    subprocess.run(["powershell", "-Command", command])

    if os.path.exists(output_path):
        winsound.PlaySound(output_path, winsound.SND_FILENAME)
    else:
        print("❌ Audio file was not created")

print("Speak something... (Press Ctrl+C to stop)")

# ---------------- RECORD FUNCTION (VAD) ----------------
def record_audio(filename="input.wav", fs=16000, silence_threshold=0.005, silence_duration=3.0):
    """
    Records audio until there's no sound for `silence_duration` seconds.
    Uses energy-based voice activity detection.
    """
    print("Listening... (speak now)")
    
    chunk_duration = 0.01  # Process in 10ms chunks
    chunk_samples = int(fs * chunk_duration)
    max_duration = 10  # Max recording time to prevent infinite loops
    recording = []
    silence_counter = 0
    chunks_to_wait = int(silence_duration / chunk_duration)
    speech_started = False

    with sd.InputStream(samplerate=fs, channels=1, dtype='float32') as stream:
        # Calibrate ambient noise for the current microphone
        calibration_chunks = int(0.5 / chunk_duration)
        noise_levels = []
        for _ in range(calibration_chunks):
            chunk, overflow = stream.read(chunk_samples)
            if overflow:
                print("⚠️ Buffer overflow during calibration")
            noise_levels.append(np.sqrt(np.mean(chunk ** 2)))

        ambient_energy = float(np.mean(noise_levels)) if noise_levels else 0.0
        adaptive_threshold = max(silence_threshold, ambient_energy * 3, 0.001)
        print(f"Ambient energy={ambient_energy:.6f}, threshold={adaptive_threshold:.6f}")

        while True:
            chunk, overflow = stream.read(chunk_samples)
            if overflow:
                print("⚠️ Buffer overflow detected")
            recording.append(chunk.copy())

            energy = np.sqrt(np.mean(chunk ** 2))

            if not speech_started:
                if energy >= adaptive_threshold:
                    speech_started = True
                    print("Speech detected...")
                else:
                    continue

            if energy < adaptive_threshold:
                silence_counter += 1
            else:
                silence_counter = 0

            if silence_counter >= chunks_to_wait:
                break

            if not speech_started and len(recording) >= int(max_duration / chunk_duration):
                print("No speech detected within 10 seconds.")
                break

    if not speech_started:
        print("No speech detected. Please try again.")
        return False

    recording = np.concatenate(recording, axis=0)
    wav.write(filename, fs, recording)
    print("Recording saved.")
    return True


# ---------------- MAIN LOOP ----------------
while True:
    if not record_audio():
        continue

    # Speech → Text
    result = model.transcribe("input.wav", language="en")
    user_text = result["text"].strip()
    print("You:", user_text)

    if not user_text:
        print("Didn't catch anything, try again.")
        continue

    # Text → AI
    ai_text = get_ai_response(user_text)
    print("AI:", ai_text)

    # Extract and save personal information to profile
    extract_and_save_personal_info(user_text, ai_text)

    # Record conversation to memory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    memory_path = os.path.join(base_dir, "memory.txt")
    with open(memory_path, "a", encoding="utf-8") as f:
        f.write(f"User: {user_text}\nAI: {ai_text}\n\n")

    # Text → Speech
    speak(ai_text)
