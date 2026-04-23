import whisper
import pyttsx3
import requests
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import winsound
# ---------------- AI FUNCTION ----------------
def get_ai_response(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi",
                "prompt": prompt,
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
def record_audio(filename="input.wav", fs=16000, silence_threshold=0.01, silence_duration=2.0):
    """
    Records audio until there's no sound for `silence_duration` seconds.
    Uses energy-based voice activity detection.
    """
    print("Recording... (speak now)")
    
    chunk_duration = 0.1  # Process in 100ms chunks
    chunk_samples = int(fs * chunk_duration)
    max_duration = 30  # Max recording time to prevent infinite loops
    
    recording = []
    silence_counter = 0
    chunks_to_wait = int(silence_duration / chunk_duration)  # 20 chunks = 2 seconds
    
    while True:
        # Record a chunk
        chunk = sd.rec(chunk_samples, samplerate=fs, channels=1)
        sd.wait()
        recording.append(chunk)
        
        # Calculate energy (RMS) of the chunk
        energy = np.sqrt(np.mean(chunk ** 2))
        
        # Check if silence
        if energy < silence_threshold:
            silence_counter += 1
        else:
            silence_counter = 0
        
        # Stop if silence for specified duration
        if silence_counter >= chunks_to_wait and len(recording) > 10:  # Min 10 chunks (1 sec)
            break
        
        # Safety limit
        if len(recording) >= int(max_duration / chunk_duration):
            print("Max recording time reached")
            break
    
    # Combine all chunks
    recording = np.concatenate(recording, axis=0)
    wav.write(filename, fs, recording)
    print("Recording saved.")


# ---------------- MAIN LOOP ----------------
while True:
    record_audio()

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

    # Text → Speech
    speak(ai_text)