import os
import subprocess
import gdown
from pyrogram import Client, filters

# GitHub Secrets
API_ID = int(os.environ.get("API_ID"))
API_HASH = os.environ.get("API_HASH")
BOT_TOKEN = os.environ.get("BOT_TOKEN")

app = Client("clipper_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

@app.on_message(filters.command("start"))
def start(client, message):
    message.reply_text(
        "✅ Bot Online Hai!\n\n"
        "Format: `/cut [LINK] [START] [END]`"
    )

@app.on_message(filters.command("cut"))
def cut_video(client, message):
    args = message.text.split()
    if len(args) != 4:
        message.reply_text("❌ Format: `/cut [LINK] [START] [END]`")
        return

    gdrive_link, start_time, end_time = args[1], args[2], args[3]
    
    input_file = "input.mp4"
    output_file = "output.mp4"
    
    # Update: Yahan se input file delete karne wali line hata di gayi hai.
    # Ab naye command par sirf purani output (clip) delete hogi.
    if os.path.exists(output_file): os.remove(output_file)

    msg = message.reply_text("⏳ Processing shuru ho rahi hai...")

    try:
        # Smart Logic: Agar movie pehle se mojud nahi hai, tabhi download hogi
        if not os.path.exists(input_file):
            msg.edit_text("📥 GDrive se movie download ho rahi hai (Isme thoda time lag sakta hai)...")
            
            # GDrive link se exact ID nikal kar direct link banana
            if "/d/" in gdrive_link:
                file_id = gdrive_link.split("/d/")[1].split("/")[0]
                download_url = f"https://drive.google.com/uc?id={file_id}"
            else:
                download_url = gdrive_link

            gdown.download(url=download_url, output=input_file, quiet=False)
            
            if not os.path.exists(input_file):
                msg.edit_text("❌ Download fail ho gaya! Link check karein ya make sure public hai.")
                return
        else:
            msg.edit_text("✅ Movie pehle se system mein hai! Seedha clip cut kar rahe hain...")

        msg.edit_text("✂️ FFmpeg se clip cut ho rahi hai (Zero Quality Loss)...")

        # FFmpeg zero loss cut
        cmd = ["ffmpeg", "-ss", start_time, "-to", end_time, "-i", input_file, "-c", "copy", output_file]
        subprocess.run(cmd, check=True)

        msg.edit_text("📤 Clip ready! Telegram par upload ho rahi hai...")
        
        client.send_video(chat_id=message.chat.id, video=output_file, supports_streaming=True)
        msg.delete()

    except Exception as e:
        msg.edit_text(f"❌ Error: {e}")
    finally:
        # Storage safai - Update: Ab aakhiri mein sirf output clip delete hogi, main movie safe rahegi!
        if os.path.exists(output_file): os.remove(output_file)

if __name__ == "__main__":
    print("Bot Started...", flush=True)
    app.run()
