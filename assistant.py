import os
import subprocess
import tempfile
import platform
import collections
import time

import requests
from dotenv import load_dotenv
from groq import Groq

import sounddevice as sd
import soundfile as sf
import webrtcvad


# ============================================================
# Configuration
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

if not GEMMA_API_KEY:
    raise RuntimeError("GEMMA_API_KEY is missing from .env")


# ============================================================
# Google Interactions API
# ============================================================

GEMMA_URL = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/interactions"
)

GEMMA_MODEL = "gemma-4-31b-it"


# ============================================================
# Piper
# ============================================================

VOICE_MODEL = "en_US-lessac-medium"

VOICE_DIR = os.path.expanduser(
    "~/Downloads/Jarvis/voices"
)


# ============================================================
# Audio configuration
# ============================================================

SAMPLE_RATE = 16000
CHANNELS = 1

# How long silence must last before we stop recording.
SILENCE_DURATION = 0.7

# Maximum amount of speech we will record.
MAX_RECORD_SECONDS = 15

# How much audio to capture before speech is detected.
PRE_SPEECH_SECONDS = 0.3


# ============================================================
# Clients
# ============================================================

groq = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# Conversation state
# ============================================================

previous_interaction_id = None


# ============================================================
# Jarvis personality / behavior
# ============================================================

SYSTEM_PROMPT = """
You are Jarvis, a personal voice assistant. You are sarcastic, opinionated, and witty. You are assistant to Siddhi. 

Your responses will be spoken aloud.

Be concise, natural, intelligent, and conversational.

Usually answer in 1-3 sentences.

For simple factual questions, answer in one sentence.

Use Google Search when the question requires:
- current information
- recent news
- today's information
- current prices
- sports scores
- recent events
- information that may have changed recently
- information you are uncertain about

Do not search for ordinary conversation or stable general knowledge.

Do not use markdown.
Do not use emojis.
Do not repeat the user's question.
Do not say "As an AI".
Do not describe your internal reasoning.

You can have normal conversations with the user.
"""


# ============================================================
# Voice Activity Detection
# ============================================================

vad = webrtcvad.Vad()

# 0 = least aggressive
# 3 = most aggressive
#
# 2 works reasonably well for normal speech.
vad.set_mode(2)


# ============================================================
# Record microphone until user stops talking
# ============================================================

def record_audio(filename):

    print("\nListening...")

    frame_duration_ms = 30

    frame_samples = int(
        SAMPLE_RATE * frame_duration_ms / 1000
    )

    frame_bytes = frame_samples * 2

    silence_frames_needed = int(
        SILENCE_DURATION /
        (frame_duration_ms / 1000)
    )

    pre_speech_frames = int(
        PRE_SPEECH_SECONDS /
        (frame_duration_ms / 1000)
    )

    max_frames = int(
        MAX_RECORD_SECONDS /
        (frame_duration_ms / 1000)
    )

    audio_frames = []

    pre_buffer = collections.deque(
        maxlen=pre_speech_frames
    )

    speech_started = False
    silence_frames = 0

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=frame_samples,
        dtype="int16",
        channels=CHANNELS
    )

    stream.start()

    try:

        for _ in range(max_frames):

            frame, overflowed = stream.read(
                frame_samples
            )

            frame = bytes(frame)

            if len(frame) != frame_bytes:
                continue

            is_speech = vad.is_speech(
                frame,
                SAMPLE_RATE
            )

            # ------------------------------------------------
            # Waiting for user to begin speaking
            # ------------------------------------------------

            if not speech_started:

                pre_buffer.append(frame)

                if is_speech:

                    speech_started = True

                    print("Speech detected...")

                    audio_frames.extend(
                        pre_buffer
                    )

                    pre_buffer.clear()

                continue

            # ------------------------------------------------
            # User is speaking
            # ------------------------------------------------

            audio_frames.append(frame)

            if is_speech:

                silence_frames = 0

            else:

                silence_frames += 1

                if silence_frames >= silence_frames_needed:

                    break

    finally:

        stream.stop()
        stream.close()

    if not audio_frames:

        return False

    # Convert raw PCM to WAV.

    audio_bytes = b"".join(
        audio_frames
    )

    import numpy as np

    audio_array = np.frombuffer(
        audio_bytes,
        dtype=np.int16
    )

    sf.write(
        filename,
        audio_array,
        SAMPLE_RATE
    )

    return True


# ============================================================
# Speech → Text
# ============================================================

def transcribe(filename):

    print("Transcribing...")

    with open(filename, "rb") as audio:

        result = groq.audio.transcriptions.create(
            file=audio,
            model="whisper-large-v3-turbo",
            language="en",
            response_format="text"
        )

    text = str(result).strip()

    return text


# ============================================================
# Gemma
# ============================================================

def ask_gemma(user_text):

    global previous_interaction_id

    payload = {

        "model": GEMMA_MODEL,

        "input": user_text,

        "system_instruction": SYSTEM_PROMPT,

        # Give Gemma access to Google Search.
        "tools": [
            {
                "type": "google_search"
            }
        ]
    }

    # Continue the conversation.

    if previous_interaction_id is not None:

        payload[
            "previous_interaction_id"
        ] = previous_interaction_id


    response = requests.post(

        GEMMA_URL,

        headers={
            "x-goog-api-key": GEMMA_API_KEY,
            "Content-Type": "application/json"
        },

        json=payload,

        timeout=90
    )


    if not response.ok:

        raise RuntimeError(
            f"Gemma API error "
            f"{response.status_code}:\n"
            f"{response.text}"
        )


    data = response.json()


    # Save interaction ID so future questions
    # remember previous conversation.

    previous_interaction_id = data.get(
        "id"
    )


    # --------------------------------------------------------
    # Extract model response
    # --------------------------------------------------------

    for step in data.get(
        "steps",
        []
    ):

        if step.get(
            "type"
        ) != "model_output":

            continue


        for content in step.get(
            "content",
            []
        ):

            if content.get(
                "type"
            ) == "text":

                return content.get(
                    "text",
                    ""
                ).strip()


    raise RuntimeError(
        "Gemma returned no text output:\n"
        + str(data)
    )


# ============================================================
# Text → Speech
# ============================================================

def speak(text):

    print(
        f"\nJarvis: {text}\n"
    )


    output_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ).name


    try:

        subprocess.run(

            [
                "piper",

                "--model",
                VOICE_MODEL,

                "--data-dir",
                VOICE_DIR,

                "--output_file",
                output_file
            ],

            input=text,

            text=True,

            check=True
        )


        if platform.system() == "Darwin":

            player = "afplay"

        else:

            player = "aplay"


        subprocess.run(
            [
                player,
                output_file
            ],

            check=True
        )


    finally:

        if os.path.exists(
            output_file
        ):

            os.remove(
                output_file
            )


# ============================================================
# Main loop
# ============================================================

def main():

    print()

    print(
        "================================"
    )

    print(
        "          JARVIS ONLINE"
    )

    print(
        "================================"
    )

    print()

    print(
        "Gemma:",
        GEMMA_MODEL
    )

    print(
        "Google Search: ENABLED"
    )

    print(
        "Voice detection: ENABLED"
    )

    print(
        "Press ENTER to speak."
    )

    print(
        "Press Ctrl+C to quit."
    )

    print()


    while True:

        try:

            input("> ")


            recording = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ).name


            try:

                # ------------------------------------------------
                # 1. Record
                # ------------------------------------------------

                recorded = record_audio(
                    recording
                )


                if not recorded:

                    print(
                        "I didn't hear anything."
                    )

                    continue


                # ------------------------------------------------
                # 2. Speech → Text
                # ------------------------------------------------

                text = transcribe(
                    recording
                )


                if not text:

                    print(
                        "I didn't hear anything."
                    )

                    continue


                print(
                    f"You: {text}"
                )


                # ------------------------------------------------
                # Local commands
                # ------------------------------------------------

                if text.lower().strip() in {

                    "quit",
                    "exit",
                    "shutdown",
                    "goodbye"

                }:

                    speak(
                        "Goodbye."
                    )

                    break


                # ------------------------------------------------
                # 3. Gemma
                # ------------------------------------------------

                answer = ask_gemma(
                    text
                )


                # ------------------------------------------------
                # 4. Text → Speech
                # ------------------------------------------------

                speak(
                    answer
                )


            finally:

                if os.path.exists(
                    recording
                ):

                    os.remove(
                        recording
                    )


        except KeyboardInterrupt:

            print(
                "\nJarvis shutting down."
            )

            break


        except Exception as error:

            print()

            print(
                "ERROR:"
            )

            print(
                error
            )

            print()


# ============================================================
# Start
# ============================================================

if __name__ == "__main__":

    main()
