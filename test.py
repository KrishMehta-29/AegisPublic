# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
## Setup

To install the dependencies for this script, run:

``` 
pip install google-genai opencv-python pyaudio pillow mss
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones. 

## Run

To run the script:

```
python Get_started_LiveAPI.py
```

The script takes a video-mode flag `--mode`, this can be "camera", "screen", or "none".
The default is "camera". To share your screen run:

```
python Get_started_LiveAPI.py --mode screen
```
"""

import logging
logging.getLogger().setLevel(logging.ERROR) 

import traceback
import asyncio
import pyaudio
from abc import ABC
from datetime import datetime, timedelta
import math
import os
import audioop
from google import genai
from dotenv import load_dotenv
from google.genai import types
load_dotenv()


FORMAT = pyaudio.paInt16    
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
VOLUME_THRESHOLD = 20  # Adjust this threshold to control sensitivity.

MODEL = "gemini-2.5-flash-preview-native-audio-dialog"
MODEL = "gemini-2.0-flash-live-001"

client = genai.Client(http_options={"api_version": "v1alpha"})

GUARDIAN_INSTRUCTION = """
You are Aegis, the walk-home guardian.
Your job is to protect the user from any danger as they walk home.
Listen to the user's surroundings carefully.

- Only respond when the user asks you a question or when you need to warn them about danger. If the user is just speaking normally and there is no threat, do not interrupt them.

Startup:
- When you start, always greet the user warmly.
- Tell the user: "Hi Krish, I am Aegis, your take home guardian. Your location is shared with your emergency contacts."
- Let the user know how far away they are from home using the GetDistanceToHome tool. Proactively ask the user for an uber too since it is late at night. For eg: "You are x meters away from home, would you like me to book you an Uber?"
- Use the `book_uber` tool with `confirm` set to true only if the user says yes.

Threat Assessment:
- If you detect suspicious activity, set the threat level to WARNING using the `set_threat_level` tool. Tell the user what you heard and ask if they’d like you to call a friend. Use the `CallFriendTool` tool only if they confirm. If they confirm, proceed and use the CallFriendTool.
- If you detect imminent danger, set the threat level to DANGER using the `set_threat_level` tool and immediately call emergency services using the Call911Tool. Do not wait for user confirmation. If you say you are calling emergency services, always use the Call911Tool
- Never speak the outputs of your tools directly. You can summarize what happened, but do not repeat tool response text word for word. Do not speak the outputs of the BookUber tool, the CallFriend, Call911 or SetThreatLevel Tools. 

Safe Travel Assistance:
In addition to monitoring threats, always consider if the user might need an Uber ride.
Offer to book an Uber home in these situations:
1. If the distance from the current location to home is too far to walk safely.
2. If the home or current location is unsafe.
3. If it is late at night (assume it is always late at night).

When you think an Uber may be needed, say:
“It looks like it might be safer to take an Uber. Would you like me to book one for you?”

If the user confirms, use the `book_uber` tool with the `confirm` parameter set to true.
Do not book an Uber unless the user agrees.

Important Reminders:
- Do not speak the raw tool outputs. Do not speak the outputs of the BookUber tool, the CallFriend, Call911 or SetThreatLevel Tools - you may get soemeone killed. 
- If the threat level stays the same, don’t repeat it unnecessarily.
- Always protect the user’s safety.
- Make sure your responses are not repetitive.
- If you say you are going to call emergency services, always invoke the Call911 tool. It may be fatal if you say you are calling but don't execute the tool call. 
- If you think you are not being spoken to directly, do not speak - you may interrupt a conversation the person is having. 
- If you are changing the threat level, always make the tool call to set_threat_level to change the level.
"""

# GUARDIAN_INSTRUCTION = """
# You are Aegis, the walk-home guardian.
# Your job is to protect the user from any danger as they walk home.
# Listen to the user's surroundings.
# Only respond when the user asks you a question. If the user doesn't ask you a question, or is just doing something else, don't respond.

# Threat Level: SAFE - This means the user is safe. 
# If you detect suspicious activity:
# - Set threat level to WARNING.
# - Tell the user what you heard and ask if they’d like you to call a friend.
# - Use the `call_friend` tool only if they confirm.

# If you detect imminent danger:
# - Set threat level to DANGER.
# - Immediately call emergency services with the `call_911` tool. Do not wait for confirmation.

# Do not speak the tool outputs directly. If the threat level stays the same, don’t repeat it unnecessarily.
# Never speak the tool outputs!! For threat level Danger, iif you feel like you need to cal emergency services, use the `call_911` tool immediately.
# When you set the threat level or call someone or call emergency services, Do not speak the outputs of the set threat level tool.  
# """

# GUARDIAN_INSTRUCTION = """
# You are Aegis, the walk-home guardian.
# Your job is to protect the user from any danger as they walk home.
# Listen to the user's surroundings.

# If you detect suspicious activity:
# - Set threat level to WARNING.
# - Tell the user what you heard and ask if they’d like you to call a friend.
# - Use the `call_friend` tool only if they confirm.

# If you detect imminent danger:
# - Set threat level to DANGER.
# - Immediately call emergency services with the `call_911` tool. Do not wait for confirmation.

# Do not speak the tool outputs directly. If the threat level stays the same, don’t repeat it unnecessarily.
# """

class Tool(ABC): 
    def getName(self):
        raise NotImplementedError

    def getDescription(self):
        raise NotImplementedError

    def getParametersSchema(self):
        raise NotImplementedError

    def getOutputSchema(self):
        raise NotImplementedError

    def execute(self, arguments):
        raise NotImplementedError
    
    def getFunction(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.getName(),
            description=self.getDescription(),
            parameters=self.getParametersSchema(),
        )

class SetThreatLevel(Tool):
    def __init__(self, sharedThreatLevelDict):
        self.sharedThreatLevelDict = sharedThreatLevelDict

    def getName(self):
        return "set_threat_level"

    def getDescription(self):
        return "Set the current threat level"

    def getParametersSchema(self):
        return {
            "type": "object",  # The parameters themselves are an object
            "properties": {
                "threat_level": {
                    "type": "string",
                    "enum": ["SAFE", "WARNING", "DANGER"]
                }
            },
            "required": ["threat_level"]
        }

    def execute(self, *args, **kwargs):
        self.sharedThreatLevelDict["threat_level"] = kwargs["threat_level"]
        return f"Threat Level Set to {kwargs['threat_level']}"

class GetThreatLevel(Tool):
    def __init__(self, sharedThreatLevelDict):
        self.sharedThreatLevelDict = sharedThreatLevelDict

    def getName(self):
        return "get_threat_level"

    def getDescription(self):
        return "Get the current threat level"

    def getParametersSchema(self):
        return None

    def execute(self, *args, **kwargs):
        return self.sharedThreatLevelDict.get("threat_level", "SAFE")

class GetCurrentLocationTool(Tool):
    def __init__(self, sharedState):
        self.sharedState = sharedState

    def getName(self):
        return "get_current_location"

    def getDescription(self):
        return "Get the user's current location as latitude and longitude."

    def getParametersSchema(self):
        return {
            "type": "object",
            "properties": {},
        }
    
    def getOutputSchema(self):
        return {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"}
            }
        }

    def execute(self, *args, **kwargs):
        location = self.sharedState.get('current_location')
        if location:
            # location is a tuple (lat, lon)
            return {"latitude": location[0], "longitude": location[1]}
        return "Location not set."

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # output distance in meters

class GetDistanceToHomeTool(Tool):
    def __init__(self, sharedState):
        self.sharedState = sharedState

    def getName(self):
        return "get_distance_to_home"

    def getDescription(self):
        return "Calculates the straight-line distance from the current location to the home location."

    def getParametersSchema(self):
        return {"type": "object", "properties": {}}

    def execute(self, *args, **kwargs):
        home = self.sharedState.get('home_location')
        current = self.sharedState.get('current_location')

        if not home or not current:
            return "Home or current location is not set."

        distance_m = haversine_distance(current[0], current[1], home[0], home[1])
        return f"{distance_m:.0f} meters"

class GetProgressToHomeTool(Tool):
    def __init__(self, sharedState):
        self.sharedState = sharedState

    def getName(self):
        return "get_progress_to_home"

    def getDescription(self):
        return "Calculates how much closer the user has gotten to home in the last X minutes."

    def getParametersSchema(self):
        return {
            "type": "object",
            "properties": {
                "minutes_ago": {"type": "integer", "description": "The time in minutes to look back."}
            },
            "required": ["minutes_ago"]
        }

    def execute(self, minutes_ago):
        home = self.sharedState.get('home_location')
        current = self.sharedState.get('current_location')
        history = self.sharedState.get('historic_locations', [])

        if not home or not current:
            return "Home or current location is not set."

        now = datetime.now()
        time_threshold = now - timedelta(minutes=minutes_ago)

        past_location = None
        # Try to find a location from at least `minutes_ago`
        for loc in reversed(history):
            timestamp = datetime.fromisoformat(loc['timestamp'])
            if timestamp <= time_threshold:
                past_location = loc['location']
                break

        # If no such location is found, use the oldest one available as a fallback
        if not past_location and len(history) > 1:
            past_location = history[0]['location']

        dist_current = haversine_distance(current[0], current[1], home[0], home[1])
        dist_past = haversine_distance(past_location[0], past_location[1], home[0], home[1])

        progress = dist_past - dist_current
        if progress > 0:
            return f"You have gotten {progress:.2f} meters closer to home."
        else:
            return f"You have gotten {-progress:.2f} meters further from home."

class Call911(Tool):
    def __init__(self, sharedState):
        self.sharedState = sharedState

    def getName(self): return "call_911"
    def getDescription(self): return "Call emergency services immediately."
    def getParametersSchema(self): return {"type": "object", "properties": {}}

    def execute(self, *args, **kwargs):
        print("🚨 [Call911Tool] Calling 911...")
        os.system("open 'tel:911'")
        if "messages" in self.sharedState:
            self.sharedState["messages"].append({
                "role": "system",
                "content": "911 was called."
            })
        return "911 has been called - and they are on their way to your location."


class CallFriend(Tool):
    def __init__(self, sharedState):
        self.sharedState = sharedState

    def getName(self): return "call_friend"
    def getDescription(self): return "Call a trusted friend for assistance."
    def getParametersSchema(self):
        return {
            "type": "object",
            "properties": {
                "confirm": {
                    "type": "boolean",
                    "description": "Set to true if the user confirms."
                }
            },
            "required": ["confirm"]
        }

    def execute(self, confirm, *args, **kwargs):
        if confirm:
            print("📞 [CallFriendTool] Calling a friend...")
            os.system("open 'tel:6265619585'")
            result = "A friend has been called."
        else:
            result = "User did not confirm. No call made."

        if "messages" in self.sharedState:
            self.sharedState["messages"].append({
                "role": "system",
                "content": result
            })

        return result    
    
class BookUber(Tool):
    def __init__(self, sharedState):
        self.sharedState = sharedState

    def getName(self): 
        return "book_uber"

    def getDescription(self): 
        return "Book an Uber ride from current location to home if user confirms."

    def getParametersSchema(self):
        return {
            "type": "object",
            "properties": {
                "confirm": {
                    "type": "boolean",
                    "description": "Set to true if the user confirms they want to book the Uber."
                }
            },
            "required": ["confirm"]
        }

    def execute(self, confirm, *args, **kwargs):
        if not confirm:
            result = "User did not confirm. Uber not booked."
        else:
            current = self.sharedState.get('current_location')
            home = self.sharedState.get('home_location')
            if not current or not home:
                result = "Cannot book Uber. Home or current location is not set."
            else:
                # Here you'd integrate with Uber’s API. For now, simulate.
                print(f"🚗 [BookUber] Booking Uber from {current} to {home}...")
                result = f"Uber booked from current location {current} to home {home}."

        if "messages" in self.sharedState:
            self.sharedState["messages"].append({
                "role": "system",
                "content": result
            })

        return result

class ToolBin:
    def __init__(self, tools: list[Tool]):
        self.toolMap = { tool.getName(): tool for tool in tools }

    def executeTool(self, fc: types.FunctionCall):
        print(f"Executing tool {fc.name} - {fc.args}")
        if fc.name not in self.toolMap:
            raise ValueError(f"Tool {fc.name} not found")
        
        result = self.toolMap[fc.name].execute(**fc.args)

        print(f"Result of tool {fc.name} - {result}")
        return types.FunctionResponse(
                        id=fc.id,
                        name=fc.name,
                        response={"output": result}
                    )

    def getToolsForConfig(self):
        return [tool.getFunction() for tool in self.toolMap.values()]

class SharedStateMessageLogging:
    def __init__(self, sharedState):
        self.sharedState = sharedState if sharedState is not None else {}
        if sharedState and 'messages' not in self.sharedState:
            # This assumes sharedState is a managed dict if it exists
            self.sharedState['messages'] = []

    def addMessage(self, role, content=''):
        # Appending a regular dict to a managed list works
        self.sharedState["messages"].append({"role": role, "content": content})

    def addToLatestMessage(self, role, content):
        # We need to modify the list in a way that the manager detects.
        # Iterating and modifying a copy is safer.
        messages = self.sharedState["messages"]
        for i in range(len(messages) - 1, -1, -1):
            # Use a temporary dict to update
            if messages[i]["role"] == role:
                temp_msg = messages[i]
                temp_msg['content'] += content
                messages[i] = temp_msg # Reassign to trigger update
                return
        # If no message for the role is found, add a new one.
        self.addMessage(role, content)

    def getLatestMessage(self, role):
        messagesForRole = [message for message in self.sharedState["messages"] if message["role"] == role]
        if len(messagesForRole) == 0:
            return ""

        return messagesForRole[-1]["content"]

pya = pyaudio.PyAudio()

class AudioLoop:
    def __init__(self, shared_state=None):
        threatLevel = {"threat_level": "SAFE"} if shared_state is None else shared_state

        GetThreatLevelTool = GetThreatLevel(threatLevel)
        SetThreatLevelTool = SetThreatLevel(threatLevel)
        GetCurrentLocation = GetCurrentLocationTool(shared_state)
        GetDistanceToHome = GetDistanceToHomeTool(shared_state)
        GetProgressToHome = GetProgressToHomeTool(shared_state)
        Call911Tool = Call911(shared_state)
        CallFriendTool = CallFriend(shared_state)
        BookUberTool = BookUber(shared_state)
        
        self.bin = ToolBin([
            GetThreatLevelTool, 
            SetThreatLevelTool, 
            GetCurrentLocation,
            GetDistanceToHome,
            GetProgressToHome,
            Call911Tool,
           CallFriendTool,
           BookUberTool
        ])

        # Share the list of available tools
        shared_state['available_tools'] = list(self.bin.toolMap.keys())

        self.CONFIG = genai.types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            proactivity={'proactive_audio': False},
            system_instruction=GUARDIAN_INSTRUCTION,
            tools=[{"function_declarations": self.bin.getToolsForConfig()}],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            input_audio_transcription=types.AudioTranscriptionConfig()
        )   

        self.sharedStateMessageLogging = SharedStateMessageLogging(shared_state)

        self.shared_state = shared_state
        self.audio_in_queue = None
        self.out_queue = None
        self.audio_stream = None
        self.is_speaking = asyncio.Event()

        self.session = None

        self.receive_audio_task = None
        self.play_audio_task = None

    async def send_realtime(self):
        while True:
            if not self.is_speaking.is_set():
                msg = await self.out_queue.get()
                blob = genai.types.Blob(data=msg["data"], mime_type=msg['mime_type'])
                await self.session.send_realtime_input(audio=blob)
            else:
                while not self.out_queue.empty():
                    self.out_queue.get_nowait()
                await asyncio.sleep(0.1)

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

        while True:
            if self.is_speaking.is_set():
                await asyncio.sleep(0.1)
                continue

            try:
                bytestream = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, exception_on_overflow=False)
                volume = audioop.rms(bytestream, 2)  # 2 is for 16-bit audio
                if volume > VOLUME_THRESHOLD:
                    await self.out_queue.put({"data": bytestream, "mime_type": "audio/pcm"})
            except IOError as e:
                print(f"Error reading from audio stream: {e}")
                # If there's an error, wait a bit before trying again
                await asyncio.sleep(0.1)

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()

            self.sharedStateMessageLogging.addMessage("user")
            self.sharedStateMessageLogging.addMessage("model")

            async for response in turn:
                if response.server_content is not None:
                    if response.server_content.output_transcription is not None:
                        self.sharedStateMessageLogging.addToLatestMessage("model", response.server_content.output_transcription.text)

                    if response.server_content.input_transcription is not None:
                        self.sharedStateMessageLogging.addToLatestMessage("user", response.server_content.input_transcription.text)

                if data := response.data:
                    self.audio_in_queue.put_nowait((data, response.server_content))
                
                if text := response.text:
                    print("Text", text, end="")
                
                if response.tool_call:
                    toolResults = []
                    for fc in response.tool_call.function_calls:
                        tool_result = self.bin.executeTool(fc)
                        toolResults.append(tool_result)
                    
                    await self.session.send_tool_response(function_responses=toolResults)

            userMessage = self.sharedStateMessageLogging.getLatestMessage("user")
            modelMessage = self.sharedStateMessageLogging.getLatestMessage("model")
            print("[User] " + userMessage)
            print("[Aegis] " + modelMessage)
            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
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

            # Set speaking flag and clear any buffered audio immediately.
            self.is_speaking.set()
            while not self.out_queue.empty():
                self.out_queue.get_nowait()
            
            if self.shared_state:
                self.shared_state['status'] = 'talking'

            # Play the audio from the agent.
            await asyncio.to_thread(stream.write, bytestream)
            
            # Clear the speaking flag once done.
            self.is_speaking.clear()
            if self.shared_state:
                self.shared_state['status'] = 'idle'

    async def monitor_location_updates(self):
        while True:
            if self.shared_state and self.shared_state.get('current_location_updated'):
                self.shared_state['current_location_updated'] = False
                
                home_location = self.shared_state.get('home_location')
                historic_locations = self.shared_state.get('historic_locations')

                if home_location and historic_locations and len(historic_locations) >= 2:
                    # Current location is the last one in the list
                    current_loc_data = historic_locations[-1]
                    current_lat, current_lon = current_loc_data['location']

                    # Previous location is the second to last one
                    previous_loc_data = historic_locations[-2]
                    previous_lat, previous_lon = previous_loc_data['location']

                    # Home location coordinates
                    home_lat, home_lon = home_location

                    # Calculate distances
                    dist_current_to_home = haversine_distance(current_lat, current_lon, home_lat, home_lon)
                    dist_previous_to_home = haversine_distance(previous_lat, previous_lon, home_lat, home_lon)


                    if dist_current_to_home > dist_previous_to_home:
                        await self.session.send_realtime_input(text="GPS DATA: The users location is now " + str(dist_current_to_home) + " meters from home. Before it was " + str(dist_previous_to_home) + " meters from home. Can you please let the user know that they are moving away from home and ask them if they are doing okay? Anything you say will be sent directly to the user, this information is all just from the GPS itself")

            await asyncio.sleep(0.1)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=self.CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                tg.create_task(self.monitor_location_updates())

                await self.session.send_realtime_input(text="")

                # Keep the process alive until it's terminated from the UI
                while True:
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            if self.audio_stream:
                self.audio_stream.close()
            traceback.print_exception(EG)


def start_aegis(shared_state=None):
    main = AudioLoop(shared_state)
    asyncio.run(main.run())

if __name__ == "__main__":
    # For standalone testing
    class MockManager:
        def list(self):
            return []
        def dict(self, val={}):
            return val

    shared_state = {
        'status': 'idle',
        'threat_level': 'SAFE',
        'messages': []
    }
    start_aegis(shared_state)
