import traceback
import asyncio
import pyaudio
import subprocess

from google import genai
from dotenv import load_dotenv
from google.genai import types
import os

load_dotenv()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

MODEL = "gemini-2.0-flash-live-001"

client = genai.Client(http_options={"api_version": "v1alpha"})

CONFIG = genai.types.LiveConnectConfig(
    response_modalities=["TEXT"],   # ✅ TEXT only — no audio output
    proactivity={"proactive_audio": False},
)

pya = pyaudio.PyAudio()

GUARDIAN_INSTRUCTION = """
You are Aegis, the walk-home guardian.
Listen to the user's surroundings.
Never output audio or engage in small talk.
Respond ONLY with a clear threat analysis in this exact format:

THREAT: <SAFE / WARNING / DANGER>
REASON: <short reason, 1 sentence>

If there is no threat, always say "THREAT: SAFE" and a reason.
If there is suspicious activity, say "THREAT: WARNING".
If there is real danger, say "THREAT: DANGER".
"""

DANGER_KEYWORDS = ["danger", "warning", "d anger", "w arning", "da nger", "wa rning"]

def call_emergency_contact():
    phone_number = f"tel:{os.getenv("PHONE_NUMBER")}"
    print(f"📞 [TOOL] Calling trusted contact at {phone_number}...")
    try:
        subprocess.run(["open", phone_number])
    except Exception as e:
        print(f"[ERROR] Could not initiate call: {e}")

class SafeSpaceGuardian:
    def __init__(self):
        self.out_queue = None
        self.session = None

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        kwargs = {"exception_on_overflow": False}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            blob = genai.types.Blob(data=msg["data"], mime_type=msg["mime_type"])
            await self.session.send_realtime_input(audio=blob)

    async def receive_text(self):
        while True:
            turn = self.session.receive()
            full_text = ""

            async for response in turn:
                if response.text:
                    piece = response.text.strip()
                    print(piece, end="", flush=True)  # optional: stream partial
                    full_text += piece + " "

            print(f"\n\n[MODEL THREAT ASSESSMENT]:\n{full_text.strip()}\n")

            threat_lower = full_text.lower()
            if any(level in threat_lower for level in DANGER_KEYWORDS):
                print("[ALERT] 🚨 Elevated threat detected! Taking action...")
                call_emergency_contact()
            else:
                print("[INFO] ✅ All clear — safe environment.")

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.out_queue = asyncio.Queue(maxsize=5)

                # ✅ Structured threat-only instruction
                initial_content = types.Content(
                    role="user",
                    parts=[types.Part(text=GUARDIAN_INSTRUCTION.strip())]
                )

                await self.session.send_client_content(
                    turns=[initial_content],
                    turn_complete=True
                )

                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_text())

        except Exception as e:
            traceback.print_exc()
        finally:
            if hasattr(self, "audio_stream"):
                self.audio_stream.close()

if __name__ == "__main__":
    guardian = SafeSpaceGuardian()
    asyncio.run(guardian.run())
