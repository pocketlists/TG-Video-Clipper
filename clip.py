import os
import subprocess
import gdown
from pyrogram import Client

# Environment variables se data lena
BOT_TOKEN = os.environ.get("BOT_TOKEN")
API_ID = os.environ.get("API_ID")
API_HASH = os.environ.get("API_HASH")
CHAT_ID = int(os.environ.get("CHAT_ID"))

GDRIVE_LINK = os.environ.get("GDRIVE_LINK")
START_TIME = os.environ.get("START_TIME")
END_TIME = os.environ.get("END_TIME")

app = Client("my_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

def process_video():
    print("Downloading from GDrive...")
    input_file = "movie.mp4"
    output_file = "clip.mp4"
    
    # GDrive se download
    gdown.download(GDRIVE_LINK, input_file, fuzzy=True)

    print(f"Clipping video from {START_TIME} to {END_TIME}...")
    # FFmpeg se quality loss ke bina cut karna (-c copy)
    command = [
        "ffmpeg", "-i", input_file, 
        "-ss", START_TIME, "-to", END_TIME, 
        "-c", "copy", output_file
    ]
    subprocess.run(command)

    print("Uploading to Telegram...")
    with app:
        app.send_video(
            chat_id=CHAT_ID,
            video=output_file,
            caption=f"✂️ Clip from {START_TIME} to {END_TIME}",
            supports_streaming=True
        )

if __name__ == "__main__":
    process_video()
