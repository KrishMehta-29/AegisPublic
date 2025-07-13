import traceback
import asyncio
import pyaudio

from google import genai
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "gemini-2.0-flash-live-001"

client = genai.Client(http_options={"api_version": "v1alpha"})

CONFIG = genai.types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    proactivity={"proactive_audio": False},
)

pya = pyaudio.PyAudio()

GUARDIAN_INSTRUCTION = """
You are Aegis, the walk-home guardian.
Listen to the user's surroundings. Do NOT respond to normal conversation or ambient noise.
Only respond if you detect danger, suspicious activity, or if the user needs help.
Stay silent otherwise.
"""

class SafeSpaceGuardian:
    def __init__(self):
        self.audio_in_queue = None
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

    async def receive_audio(self):
        while True:
            turn = self.session.receive()
            async for response in turn:
                if response.data:
                    self.audio_in_queue.put_nowait((response.data, response.server_content))
                elif response.text:
                    print(f"[MODEL TEXT]: {response.text.strip()}")
                    if any(word in response.text.lower() for word in ["danger", "suspicious", "help"]):
                        print("[ALERT] Danger detected! Taking action...")
                        # TODO: Add your trusted contacts/911 trigger here
            # Clear leftover audio if turn completes
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream, _ = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # ✅ Correct: Use send_client_content with `turns=[Content]`
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
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

        except Exception as e:
            traceback.print_exc()
        finally:
            if hasattr(self, "audio_stream"):
                self.audio_stream.close()

if __name__ == "__main__":
    guardian = SafeSpaceGuardian()
    asyncio.run(guardian.run())
