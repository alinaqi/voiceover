import requests

# Replace this with your ElevenLabs API Key
ELEVENLABS_API_KEY = "<your key>"

def list_voices():
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        voices = response.json()["voices"]
        for voice in voices:
            print(f"Voice ID: {voice['voice_id']}")
            print(f"Name: {voice['name']}")
            print(f"Description: {voice['description']}")
            print(f"Category: {voice['category']}")
            print(f"Language: {voice.get('language', 'Unknown')}")
            print(f"Gender: {voice.get('gender', 'Unknown')}")
            print("=" * 40)
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Call the function
list_voices()
