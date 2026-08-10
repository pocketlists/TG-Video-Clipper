import os
import subprocess
import gdown
from pyrogram import Client, filters

# GitHub Secrets se variables import kar rahe hain
API_ID = int(os.environ.get("API_ID"))
API_HASH = os.environ.get("API_HASH")
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = int(os.environ.get("CHAT_ID"))

app = Client("clipper_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

# Ye bot sirf aapki CHAT_ID par respond karega
@app.on_message(filters.command("start") & filters.chat(CHAT_ID))
def start(client, message):
    message.reply_text(
        "✅ GitHub Action Bot Online Hai!\n\n"
        "Command format:\n"
        "`/cut [GDRIVE_LINK] [START_TIME] [END_TIME]`\n\n"
        "Example:\n"
        "`/cut https://drive.google.com/file/d/... 00:01:00 00:02:30`"
    )

@app.on_message(filters.command("cut") & filters.chat(CHAT_ID))
def cut_video(client, message):
    args = message.text.split()
    if len(args) != 4:
        message.reply_text("❌ Format: `/cut [LINK] [START] [END]`")
        return

    gdrive_link, start_time, end_time = args[1], args[2], args[3]
    msg = message.reply_text("📥 GDrive se movie download ho rahi hai...")
    
    input_file = "input.mp4"
    output_file = "output.mp4"
    
    # Purani files clear karna
    if os.path.exists(input_file): os.remove(input_file)
    if os.path.exists(output_file): os.remove(output_file)

    try:
        # Gdown se file fetch karna
        gdown.download(url=gdrive_link, output=input_file, quiet=False, fuzzy=True)
        msg.edit_text("✂️ Download complete! FFmpeg se clip cut ho rahi hai (Zero Quality Loss)...")

        # FFmpeg stream copy process
        cmd = ["ffmpeg", "-ss", start_time, "-to", end_time, "-i", input_file, "-c", "copy", output_file]
        subprocess.run(cmd, check=True)

        msg.edit_text("📤 Clip ready! Telegram par upload ho rahi hai...")
        
        # Telegram par final output bhejna
        client.send_video(chat_id=message.chat.id, video=output_file, supports_streaming=True)
        msg.delete()

    except Exception as e:
        msg.edit_text(f"❌ Error: {e}")
    finally:
        # Storage free karna
        if os.path.exists(input_file): os.remove(input_file)
        if os.path.exists(output_file): os.remove(output_file)

if __name__ == "__main__":
    print("Bot Started via GitHub Actions...")
    app.run()
