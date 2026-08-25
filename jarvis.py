import os
import sys
import time
import queue
import asyncio
import platform

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from google import genai
from google.genai import types

from openwakeword.model import Model as WakeWordModel
import openwakeword.utils as oww_utils


# ============================================================
# WHAT'S DIFFERENT ABOUT THIS VERSION
#
# The previous jarvis.py used four separate stages:
#   mic -> Groq STT -> Gemini text -> Piper TTS -> speaker
#
# This version uses Gemini's Live API, which is a single
# persistent streaming connection that does all of that
# internally:
#   mic (streamed continuously) -> Gemini Live -> speaker
#     (streamed continuously)
#
# Benefits:
#   - No separate STT call, no separate TTS call. Both are
#     built into the model.
#   - The API has server-side voice activity detection, so
#     it decides when you've stopped talking. This replaces
#     the local webrtcvad logic entirely, including the bug
#     where it kept recording for the full 15s cap.
#   - Audio response starts playing as soon as the first
#     chunk arrives, not after the full reply is generated.
#     This is the "streams almost instantaneously" behavior.
#
# This version also replaces "press ENTER to talk" with a
# wake word: say "hey Jarvis" (or "Jarvis") and it activates
# on its own. This uses openWakeWord, which is free, fully
# offline, and ships a pretrained "hey jarvis" model — no
# training or API key needed for this part.
#
# The mic runs continuously. A single audio callback pushes
# raw frames into a queue. A state machine decides what to do
# with each frame:
#   - LISTENING state: frames go to the wake word model.
#   - ACTIVE state: frames get streamed to the Gemini Live
#     session instead.
# Saying the wake word flips LISTENING -> ACTIVE. Gemini
# reporting turn_complete flips ACTIVE -> LISTENING.
#
# Tradeoffs / things to verify on your machine, since I
# can't run audio hardware from here:
#   - Voice, sample rates, and exact model name availability
#     depend on your API access tier. Adjust MODEL_NAME below
#     if you get a "model not found" error — try the other
#     Live model names visible in your AI Studio quota page.
#   - Google Search grounding tool support inside Live API
#     sessions varies by model version; it's included below
#     but wrapped so the script still runs if your model
#     rejects it.
#   - openWakeWord's "hey jarvis" model may also fire on just
#     "Jarvis" but with a higher false-reject rate — say "hey
#     Jarvis" for the most reliable trigger.
#   - The wake word listener is paused while Jarvis is
#     speaking, since openWakeWord's own docs note that
#     simultaneous playback increases false detections
#     without echo cancellation. This means you currently
#     can't barge in with the wake word mid-response — you
#     have to wait for Jarvis to finish before saying it
#     again. Flag if you want barge-in and we can look at
#     adding acoustic echo cancellation.
#   - WAKE_THRESHOLD below controls sensitivity. Lower catches
#     more (but more false positives), higher requires a
#     clearer utterance.
# ============================================================


load_dotenv()

GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")

if not GEMMA_API_KEY:
    raise RuntimeError("GEMMA_API_KEY is missing from .env")


# ============================================================
# Model / voice configuration
# ============================================================

# Try this first. If you get a "model not found" or similar
# error, swap in another Live model name from your AI Studio
# quota page (e.g. "gemini-3.1-flash-live-preview").
MODEL_NAME = "gemini-2.5-flash-native-audio-preview-09-2025"

# Prebuilt Gemini voice. Other options include: Aoede, Charon,
# Fenrir, Kore, Puck. Try a few and see what fits Jarvis.
VOICE_NAME = "Fenrir"

# Live API expects 16kHz mono PCM16 input.
INPUT_SAMPLE_RATE = 16000

# Live API returns 24kHz mono PCM16 output audio.
OUTPUT_SAMPLE_RATE = 24000

CHANNELS = 1

# How many milliseconds of audio per chunk sent to the API.
CHUNK_MS = 20
CHUNK_SAMPLES = int(INPUT_SAMPLE_RATE * CHUNK_MS / 1000)


# ============================================================
# Wake word configuration
# ============================================================

WAKE_WORD_NAME = "hey_jarvis"

# Score threshold (0-1) needed to trigger activation.
# openWakeWord's own default guidance is 0.5 for most cases.
WAKE_THRESHOLD = 0.5

# openWakeWord's models expect 16kHz mono int16 audio in
# frames of this many samples (80ms).
WAKE_FRAME_SAMPLES = 1280


SYSTEM_PROMPT = """
You are Jarvis, a personal voice assistant. You are sarcastic, opinionated, and witty, in the dry British sense. You are assistant to Siddhi.

Speak in British English: British vocabulary and spelling conventions (e.g. "brilliant", "rubbish", "quite"), not American ones.

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
"""


client = genai.Client(api_key=GEMMA_API_KEY)


def build_config():

    kwargs = dict(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_PROMPT,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=VOICE_NAME
                )
            ),
        ),
        # Ask the API to also give us text transcripts of both
        # sides, purely so we can print "You:" / "Jarvis:" like
        # the old version did.
        input_audio_transcription={},
        output_audio_transcription={},
    )

    try:

        kwargs["tools"] = [{"google_search": {}}]

        return types.LiveConnectConfig(**kwargs)

    except Exception:

        # If this model/SDK version rejects the search tool in
        # a live session, fall back to no tools rather than
        # crashing the whole script.

        kwargs.pop("tools", None)

        return types.LiveConnectConfig(**kwargs)


# ============================================================
# Wake word listener
#
# Wraps openWakeWord. Downloads the pretrained models on first
# run (openWakeWord handles caching itself).
# ============================================================

class WakeWordListener:

    def __init__(self):

        # Ensures the pretrained model files are present.
        oww_utils.download_models()

        self._model = WakeWordModel(
            wakeword_models=[WAKE_WORD_NAME]
        )

        self._buffer = np.array([], dtype=np.int16)

    def feed(self, pcm_bytes):
        """
        Feed raw int16 PCM bytes in. Returns True the moment
        the wake word score crosses threshold.
        """

        new_samples = np.frombuffer(pcm_bytes, dtype=np.int16)

        self._buffer = np.concatenate([self._buffer, new_samples])

        triggered = False

        while len(self._buffer) >= WAKE_FRAME_SAMPLES:

            frame = self._buffer[:WAKE_FRAME_SAMPLES]
            self._buffer = self._buffer[WAKE_FRAME_SAMPLES:]

            predictions = self._model.predict(frame)

            score = predictions.get(WAKE_WORD_NAME, 0.0)

            if score >= WAKE_THRESHOLD:
                triggered = True

        return triggered

    def reset(self):

        # Clears internal model buffers so leftover audio
        # doesn't cause an immediate re-trigger next time we
        # start listening again.
        self._model.reset()
        self._buffer = np.array([], dtype=np.int16)


# ============================================================
# Continuous microphone capture
#
# sounddevice runs its callback on a separate audio thread.
# We push raw PCM16 bytes into a plain thread-safe queue.Queue
# that the main asyncio loop drains.
# ============================================================

class MicStreamer:

    def __init__(self):

        self._q = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):

        if status:
            print(f"[mic status] {status}", file=sys.stderr)

        self._q.put(bytes(indata))

    def start(self):

        self._stream = sd.RawInputStream(
            samplerate=INPUT_SAMPLE_RATE,
            blocksize=CHUNK_SAMPLES,
            dtype="int16",
            channels=CHANNELS,
            callback=self._callback,
        )

        self._stream.start()

    def stop(self):

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self.drain()

    def drain(self):

        chunks = []

        while not self._q.empty():
            chunks.append(self._q.get_nowait())

        return chunks

    async def get_async(self, timeout=0.1):
        """
        Blocks (off the event loop) for up to `timeout` seconds
        waiting for the next chunk. Returns None on timeout.
        """

        loop = asyncio.get_event_loop()

        try:
            return await loop.run_in_executor(
                None, self._q.get, True, timeout
            )
        except queue.Empty:
            return None


# ============================================================
# Speaker playback
#
# A simple streaming output. We write chunks as they arrive
# so playback begins before the full reply has been received.
# ============================================================

class SpeakerPlayer:

    def __init__(self):

        self._stream = sd.RawOutputStream(
            samplerate=OUTPUT_SAMPLE_RATE,
            dtype="int16",
            channels=CHANNELS,
        )

        self._stream.start()
        self._first_chunk_time = None

    def write(self, pcm_bytes):

        if self._first_chunk_time is None:
            self._first_chunk_time = time.time()

        self._stream.write(pcm_bytes)

    def time_to_first_audio(self, since):

        if self._first_chunk_time is None:
            return None

        return self._first_chunk_time - since

    def close(self):

        self._stream.stop()
        self._stream.close()


# ============================================================
# One conversational turn
#
# Called once the wake word has already fired. Streams mic
# audio to the session until the API's own VAD reports the
# turn is complete, then plays back the streamed audio
# response as it arrives.
# ============================================================

async def run_turn(session, mic):

    speaker = SpeakerPlayer()

    turn_start = time.time()

    print("Yes? Listening...")

    user_text_parts = []
    model_text_parts = []

    stop_sending = asyncio.Event()

    async def sender():

        try:

            while not stop_sending.is_set():

                chunk = await mic.get_async(timeout=0.05)

                if chunk is None:
                    continue

                await session.send_realtime_input(
                    audio=types.Blob(
                        data=chunk,
                        mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}",
                    )
                )

        except Exception as error:

            print(f"[sender error] {error}", file=sys.stderr)

    async def receiver():

        try:

            async for message in session.receive():

                server_content = getattr(
                    message, "server_content", None
                )

                if server_content is None:
                    continue

                if getattr(server_content, "interrupted", False):
                    break

                input_transcription = getattr(
                    server_content, "input_transcription", None
                )

                if input_transcription and input_transcription.text:
                    user_text_parts.append(input_transcription.text)

                output_transcription = getattr(
                    server_content, "output_transcription", None
                )

                if output_transcription and output_transcription.text:
                    model_text_parts.append(output_transcription.text)

                model_turn = getattr(server_content, "model_turn", None)

                if model_turn:

                    for part in model_turn.parts:

                        inline = getattr(part, "inline_data", None)

                        if inline and inline.data:
                            speaker.write(inline.data)

                if getattr(server_content, "turn_complete", False):
                    break

        except Exception as error:

            print(f"[receiver error] {error}", file=sys.stderr)

        finally:

            stop_sending.set()

    sender_task = asyncio.create_task(sender())
    receiver_task = asyncio.create_task(receiver())

    await receiver_task
    await sender_task

    speaker.close()

    ttfa = speaker.time_to_first_audio(turn_start)

    user_text = "".join(user_text_parts).strip()
    model_text = "".join(model_text_parts).strip()

    if user_text:
        print(f"You: {user_text}")

    if model_text:
        print(f"Jarvis: {model_text}")

    if ttfa is not None:
        print(f"  (time to first audio: {ttfa:.2f}s)")

    return user_text


# ============================================================
# Main loop
#
# Runs continuously. Feeds mic audio to the wake word listener
# until "hey Jarvis" is detected, then hands control over to
# run_turn() for that conversational exchange, then goes back
# to listening for the wake word.
# ============================================================

async def main():

    print()
    print("================================")
    print("       JARVIS (LIVE) ONLINE")
    print("================================")
    print()
    print("Model:", MODEL_NAME)
    print("Voice:", VOICE_NAME)
    print('Say "hey Jarvis" to activate. Ctrl+C to quit.')
    print()

    config = build_config()
    wake_word = WakeWordListener()
    mic = MicStreamer()

    mic.start()

    async with client.aio.live.connect(
        model=MODEL_NAME, config=config
    ) as session:

        try:

            while True:

                chunk = await mic.get_async(timeout=0.1)

                if chunk is None:
                    continue

                if wake_word.feed(chunk):

                    print('\n"Hey Jarvis" detected.')

                    # Drop anything queued during the wake
                    # phrase itself so it isn't replayed into
                    # the turn as leftover audio.
                    mic.drain()

                    try:
                        user_text = await run_turn(session, mic)
                    except Exception as error:
                        print()
                        print("ERROR:")
                        print(error)
                        print()
                        user_text = ""

                    if user_text.lower().strip() in {
                        "quit", "exit", "shutdown", "goodbye"
                    }:
                        break

                    wake_word.reset()

                    print('\nSay "hey Jarvis" to activate. Ctrl+C to quit.')

        except (EOFError, KeyboardInterrupt):

            pass

        finally:

            mic.stop()
            print("\nJarvis shutting down.")


if __name__ == "__main__":

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nJarvis shutting down.")
