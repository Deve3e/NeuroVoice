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
def generate_profile_summary(user_text):
    profile_prompt = f"""
Extract only important long-term facts about the user from this message.
Return short bullet points.
If there is nothing important, return NONE.

Message: {user_text}
"""
    return get_ai_response(profile_prompt)


def extract_and_save_personal_info(user_text, ai_response):
    """
    Extracts personal information from user input and saves it to profile.txt
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        profile_path = os.path.join(base_dir, "profile.txt")

        profile_summary = generate_profile_summary(user_text)
        if not profile_summary:
            return

        summary_text = profile_summary.strip()
        if summary_text.upper() == "NONE":
            return

        if os.path.exists(profile_path):
            with open(profile_path, "r", encoding="utf-8") as f:
                existing = f.read().strip()
        else:
            existing = ""

        if summary_text in existing:
            return

        updated_content = existing + "\n\n" + summary_text if existing else summary_text

        with open(profile_path, "w", encoding="utf-8") as f:
            f.write(updated_content)

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
                    profile_content = f.read().strip()
                    if profile_content:
                        if profile_content.startswith("{"):
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
                        else:
                            profile_info = "User Profile:\n" + profile_content + "\n\n"
            except:
                pass
        ASSISTANT_PERSONALITY = """
You are NeuraVoice, a personal AI assistant.
You are clear, supportive, intelligent, and slightly friendly.
You adapt to the user's goals, interests, and knowledge level.
Do not give generic answers if personal context is available.
"""
        full_prompt = f"""
{ASSISTANT_PERSONALITY}

Known user profile:
{profile_info}

Conversation memory:
{memory}

Your task:
1. Understand the user's current request.
2. Adapt your explanation to their profile.
3. Use their interests and goals when useful.
4. Be concise, practical, and personal where needed.

User: {prompt}

NeuraVoice:
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
