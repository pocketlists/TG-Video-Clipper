import os
import subprocess
import gdown
from pyrogram import Client

# GitHub Secrets
BOT_TOKEN = os.environ.get("BOT_TOKEN")
API_ID = os.environ.get("API_ID")
API_HASH = os.environ.get("API_HASH")
CHAT_ID = int(os.environ.get("CHAT_ID"))

# Cloudflare Variables
GDRIVE_LINK = os.environ.get("GDRIVE_LINK")
TIMESTAMPS = os.environ.get("TIMESTAMPS")

app = Client("my_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

def process_video():
    # Error Fix: Agar link nahi mila toh crash nahi hoga
    if not GDRIVE_LINK or not TIMESTAMPS:
        print("❌ Error: Link ya Timestamps nahi mile! Kripya command Telegram se bhejein, GitHub se manually run na karein.")
        return

    print("📥 GDrive se movie download ho rahi hai...")
    input_file = "input.mp4"
    
    # Naya Smart GDrive Logic
    if "/d/" in GDRIVE_LINK:
        file_id = GDRIVE_LINK.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
    else:
        download_url = GDRIVE_LINK

    gdown.download(url=download_url, output=input_file, quiet=False)

    if not os.path.exists(input_file):
        print("❌ Download fail ho gaya! Link check karein.")
        return

    timestamps_list = TIMESTAMPS.split(",")
    
    with app:
        for idx, ts in enumerate(timestamps_list):
            if "-" not in ts:
                continue
            
            start_time, end_time = ts.split("-")
            output_file = f"clip_{idx+1}.mp4"
            
            print(f"✂️ Clipping {idx+1}: {start_time} to {end_time}...")
            command = [
                "ffmpeg", "-i", input_file, 
                "-ss", start_time, "-to", end_time, 
                "-c", "copy", output_file
            ]
            subprocess.run(command, check=True)

            print(f"📤 Uploading clip {idx+1} to Telegram...")
            app.send_video(
                chat_id=CHAT_ID,
                video=output_file,
                caption=f"✂️ Batch Clip {idx+1}: {start_time} to {end_time}",
                supports_streaming=True
            )
            
            if os.path.exists(output_file): 
                os.remove(output_file)
                
    if os.path.exists(input_file): 
        os.remove(input_file)
    print("✅ All processing completed!")

if __name__ == "__main__":
    process_video()
