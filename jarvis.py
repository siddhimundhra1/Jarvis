import os
import sys
import time
import json
import queue
import asyncio
import platform
import traceback

import websockets.exceptions as ws_exceptions

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
# NEW IN THIS VERSION: persistent memory.
#
#   - A local JSON file (MEMORY_FILE) stores a running list of
#     facts Jarvis has learned about you (and about itself).
#   - On startup, whatever's already saved gets folded into the
#     system prompt, so Jarvis "knows" it from the first turn.
#   - Jarvis is given a `remember` function-calling tool inside
#     the Live session. It decides on its own when something is
#     worth saving (a lasting preference, a correction, a fact
#     you told it) and calls the tool; we catch that tool call,
#     append it to the JSON file, and confirm back to the model.
#   - This is separate from the Google Search tool — the Live
#     API supports multiple tools (search + function calling)
#     in the same session.
#   - Memory is intentionally simple and file-based (no vector
#     DB, no embeddings). For a personal assistant with a
#     memory file that only grows to a few hundred entries,
#     dumping the whole list into the prompt is fine. If it
#     grows large, you'll eventually want to summarize/prune
#     old entries or retrieve only relevant ones instead of
#     sending everything every time.
#
# Benefits (existing):
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
#     rejects it. Same caveat applies to the `remember`
#     function-calling tool: some Live model versions are
#     pickier about combining tool types. If you get a config
#     error, try removing google_search and keeping just the
#     remember tool, or vice versa.
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


# ============================================================
# Memory configuration
# ============================================================

# Lives next to the script by default so it survives across
# runs regardless of your current working directory. Point
# this somewhere else (e.g. a synced folder) if you want it
# to persist across machines.
MEMORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "jarvis_memory.json"
)

# How many of the most recent memories to inject into the
# system prompt. Keeps the prompt from growing unbounded if
# the file gets large.
MAX_MEMORIES_IN_PROMPT = 60


SYSTEM_PROMPT_BASE = """
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

You have a `remember` tool. Call it whenever Siddhi shares something worth
keeping for future conversations: a lasting preference, a correction to
something you got wrong, a personal detail, a recurring task or routine, or
something about how you (Jarvis) should behave going forward. Write the fact
concisely, in third person, as a standalone sentence (e.g. "Siddhi takes her
coffee black" or "Siddhi's dog is named Biscuit"). Do not call it for
one-off trivia, small talk, or anything that won't matter next week.

IMPORTANT: never use the search tool and the remember tool in the same
turn. If a single request seems to call for both (e.g. Siddhi shares a fact
worth remembering AND asks something that needs a search), pick ONE tool
for this turn and handle the other conversationally without calling it:
  - If it's a fact worth remembering, prefer calling `remember`, then
    answer the search-requiring part from your own knowledge as best you
    can, noting briefly that you'll want to look that up properly.
  - If the search need is more important to answering well right now, do
    the search instead, and just acknowledge the fact in your reply
    without calling `remember` — Siddhi can ask you to remember it again
    on its own in a follow-up turn.
At most one tool call per turn, full stop.

Do not use markdown.
Do not use emojis.
Do not repeat the user's question.
Do not say "As an AI".
Do not describe your internal reasoning.
"""


client = genai.Client(api_key=GEMMA_API_KEY)


# ============================================================
# Memory storage
#
# Plain JSON file: {"memories": [{"fact": ..., "saved_at": ...}]}
# Deliberately simple — no dedup, no embeddings, no pruning.
# Good enough for a personal assistant's memory file; revisit
# if it grows into the hundreds of entries.
# ============================================================

def load_memories():

    if not os.path.exists(MEMORY_FILE):
        return []

    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        return data.get("memories", [])
    except (json.JSONDecodeError, OSError) as error:
        print(f"[memory load error] {error}", file=sys.stderr)
        return []


def save_memory(fact):

    fact = fact.strip()

    if not fact:
        return

    memories = load_memories()

    memories.append({
        "fact": fact,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump({"memories": memories}, f, indent=2)
    except OSError as error:
        print(f"[memory save error] {error}", file=sys.stderr)


def format_memories_for_prompt(memories):

    if not memories:
        return ""

    recent = memories[-MAX_MEMORIES_IN_PROMPT:]

    lines = "\n".join(f"- {m['fact']}" for m in recent)

    return (
        "\n\nThings you already know about Siddhi and yourself, from "
        "earlier conversations:\n" + lines + "\n"
    )


# ============================================================
# `remember` function-calling tool definition
# ============================================================

REMEMBER_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="remember",
            description=(
                "Save a fact worth remembering long-term, about Siddhi or "
                "about how Jarvis should behave. Use for lasting "
                "preferences, personal details, corrections, or standing "
                "instructions. Do not use for one-off or ephemeral facts."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "fact": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "The fact to remember, written concisely in "
                            "third person as a standalone sentence."
                        ),
                    ),
                },
                required=["fact"],
            ),
        )
    ]
)


# DIAGNOSTIC MODE: which tool combo to build the session with.
# We're isolating whether `remember` or `google_search` (or the
# combination) is what's causing the 1011 close. Change this by
# hand and re-run to bisect:
#   "both"    -> google_search + remember (current failing config)
#   "remember_only" -> just the remember tool
#   "search_only"    -> just google_search
#   "none"    -> no tools at all
TOOLS_MODE = "both"


def print_ws_error(where, error, tools_mode):
    """
    Print as much detail as we can get about a Live API session
    error, specifically so we can tell a plain WebSocket close
    (e.g. 1011 internal error, which is the server rejecting the
    session config — often a bad tool combo) apart from a normal
    Python exception in our own code.
    """

    print(file=sys.stderr)
    print(f"[{where} error] tools_mode={tools_mode!r}", file=sys.stderr)
    print(f"[{where} error] type: {type(error).__module__}.{type(error).__name__}", file=sys.stderr)
    print(f"[{where} error] str:  {error}", file=sys.stderr)

    # websockets' ConnectionClosed variants carry a numeric close
    # code and reason separately from the string repr, which is
    # what actually tells us "1011 internal error" vs. something
    # else. Surface those explicitly if present.
    code = getattr(error, "code", None)
    reason = getattr(error, "reason", None)

    if code is not None or reason is not None:
        print(f"[{where} error] close code: {code}", file=sys.stderr)
        print(f"[{where} error] close reason: {reason!r}", file=sys.stderr)

    if isinstance(error, ws_exceptions.ConnectionClosedError) and getattr(error, "rcvd", None):
        print(f"[{where} error] rcvd frame: {error.rcvd}", file=sys.stderr)

    if code == 1011:
        print(
            f"[{where} error] Code 1011 = server-side internal error. "
            f"This is almost always the server rejecting the session "
            f"config it was sent (bad/unsupported tool combo, schema "
            f"issue, or unsupported model config) rather than a network "
            f"blip. Current tools_mode was {tools_mode!r} — try changing "
            f"TOOLS_MODE and re-running to bisect which tool is at fault.",
            file=sys.stderr,
        )

    print(f"[{where} error] traceback:", file=sys.stderr)
    traceback.print_exc()
    print(file=sys.stderr)


def build_config(memories):

    system_instruction = SYSTEM_PROMPT_BASE + format_memories_for_prompt(memories)

    kwargs = dict(
        response_modalities=["AUDIO"],
        system_instruction=system_instruction,
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

    if TOOLS_MODE == "both":
        kwargs["tools"] = [{"google_search": {}}, REMEMBER_TOOL]
    elif TOOLS_MODE == "remember_only":
        kwargs["tools"] = [REMEMBER_TOOL]
    elif TOOLS_MODE == "search_only":
        kwargs["tools"] = [{"google_search": {}}]
    elif TOOLS_MODE == "none":
        pass
    else:
        raise ValueError(f"Unknown TOOLS_MODE: {TOOLS_MODE!r}")

    print(f"[config] TOOLS_MODE = {TOOLS_MODE!r}, tools = {kwargs.get('tools')}")

    # NOTE: building this object is purely client-side — the SDK
    # does not validate the tool config against the server here.
    # A bad tool combo will NOT raise in this function. It will
    # instead surface later, during session.receive(), as a
    # WebSocket close (e.g. code 1011). See the error handling
    # in run_turn()'s sender()/receiver() for where that actually
    # gets caught and printed.
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
    remembered_this_turn = []

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

            print_ws_error("sender", error, tools_mode=TOOLS_MODE)

    async def receiver():

        try:

            async for message in session.receive():

                # --- tool calls (e.g. `remember`) ---

                tool_call = getattr(message, "tool_call", None)

                if tool_call and getattr(tool_call, "function_calls", None):

                    function_responses = []

                    for fc in tool_call.function_calls:

                        if fc.name == "remember":

                            fact = (fc.args or {}).get("fact", "")
                            fact = fact.strip() if fact else ""

                            if fact:
                                save_memory(fact)
                                remembered_this_turn.append(fact)
                                print(f"  (remembered: {fact})")

                            result = {"status": "saved" if fact else "skipped"}

                        else:

                            result = {"status": "unknown_tool"}

                        function_responses.append(
                            types.FunctionResponse(
                                id=fc.id,
                                name=fc.name,
                                response=result,
                            )
                        )

                    if function_responses:
                        await session.send_tool_response(
                            function_responses=function_responses
                        )

                    continue

                # --- normal turn content ---

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

                            # The moment Jarvis starts talking, stop
                            # streaming mic audio to the session. Without
                            # this, the speaker's own output leaks into
                            # the mic (no AEC here) and gets sent right
                            # back to Gemini, which transcribes Jarvis's
                            # own voice as if it were new user speech —
                            # that's the "You: <Jarvis's own line>" bug.
                            # Since each wake word triggers exactly one
                            # exchange in this design, there's nothing
                            # useful to keep sending once a response has
                            # started anyway.
                            stop_sending.set()

                            speaker.write(inline.data)

                if getattr(server_content, "turn_complete", False):
                    break

        except Exception as error:

            print_ws_error("receiver", error, tools_mode=TOOLS_MODE)

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
    print("Memory file:", MEMORY_FILE)

    memories = load_memories()

    print(f"Loaded {len(memories)} saved memories.")
    print('Say "hey Jarvis" to activate. Ctrl+C to quit.')
    print()

    config = build_config(memories)
    wake_word = WakeWordListener()
    mic = MicStreamer()

    mic.start()

    try:
        live_connection = client.aio.live.connect(model=MODEL_NAME, config=config)
    except Exception as error:
        print_ws_error("connect", error, tools_mode=TOOLS_MODE)
        mic.stop()
        return

    async with live_connection as session:

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
