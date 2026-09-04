TG-Video-Clipper — One-Time Manga Recap Video Bot
Telegram bot jo images + audio (aapka khud ka voiceover) + optional image-prompts se ek manga/manhwa recap video ban deta hai, Ken Burns style zoom/pan effects aur auto-subtitles ke saath. GitHub Actions workflow_dispatch se chalta hai, ek video render karta hai, phir job khatam ho jaata hai (compute minutes bachane ke liye).
Kya naya/fix hua
Startup ping fix: pehle bot Telegram par proactively "main chal raha hoon" nahi bata paata tha, ya chat id sahi hone ke bawajood PEER_ID_INVALID-jaisi error deta tha — kyunki Pyrogram ka session har naye Actions run mein khaali hota hai. Ab ye seedha Telegram Bot HTTP API se bheja jaata hai, jisko session/peer-cache ki zaroorat nahi.
Kisi bhi order mein files bhejo: pehle ZIP hi pehle bhejna padta tha, warna baaki files silently ignore ho jaati thi. Ab har file (ZIP, image, audio, .txt) content dekh kar khud pehchaani jaati hai — order koi matter nahi karta, filename bhi random ho to chalega.
Blank-spot/silence fix: audio mein jo pause/silence hote the wo kisi bhi image clip mein cover nahi hote the (video audio se chhota ban jaata / desync ho jaata). Ab un gaps ko pichli line ki image tak extend kiya jaata hai taaki poora audio duration hamesha covered rahe.
Deprecated Gemini SDK migrate kiya: google-generativeai (purana, deprecated) ki jagah naya google-genai SDK use ho raha hai.
Video render ab background thread mein chalta hai taaki render ke doraan bhi bot /status jaise commands ka jawab de sake.
Kaise use karein
Bot ko ek baar /start bhejo — reply mein aapka chat ID milega.
Us chat ID ko OWNER_CHAT_ID naam se GitHub secret mein daal do (isse workflow start hote hi aapko turant Telegram par ping milega).
workflow_dispatch se workflow run karo.
Bot ko files bhejo, kisi bhi order mein, jaise kisi friend ko bhej rahe ho:
ZIP — sirf images, folder ho ya na ho, dono chalega
Audio — aapka khud ka generate kiya hua voiceover
.txt (optional) — image-prompts / description; bot khud (heuristic + Gemini) decide karta hai ki ye "prompts" hai ya "script" (agar audio nahi bheja to script se hi TTS voice banegi)
Doosra audio (optional) — pehla voiceover maana jaata hai, dusra background music
Zaroori files milte hi ~15 second baad render khud shuru ho jaata hai. Turant shuru karne ke liye /render bhejo. /status se progress dekho, /reset se sab clear karo.
Required GitHub Secrets
Secret
Zaroori?
Kaam
API_ID, API_HASH
Haan
Telegram (my.telegram.org se)
BOT_TOKEN
Haan
@BotFather se
OPENAI_API_KEY
Haan
TTS + transcription fallback (agar local Whisper fail ho)
GEMINI_API_KEY
Haan
Image-sync + text classification + transcription (last-resort fallback)
OWNER_CHAT_ID
Recommended
Startup ping ke liye
SARVAM_API_KEY
Optional
Hindi TTS (na ho to OpenAI TTS fallback)
Note
Audio transcription (timings/subtitles + image-sync): ab priority order hai —
Local open-source Whisper (openai-whisper pip package, poora model GitHub Actions runner par hi local chalta hai) — na koi API key, na rate-limit, na cost. Model size WHISPER_LOCAL_MODEL env var se badla ja sakta hai (default base; options: tiny, small, medium, ...). Workflow mein ffmpeg pehle se installed hai jo isko chahiye.
OpenAI Whisper API (whisper-1) — fallback agar local model kisi wajah se fail ho jaaye.
Gemini — last-resort fallback.
Pehle Gemini primary tha lekin uski bhi API rate-limit lag jaati thi, isliye ab fully local/open-source engine primary hai — koi bhi API limit is step ko block nahi kar sakti.
Gemini model names env vars se override ho sakte hain: GEMINI_SYNC_MODEL (default gemini-3.5-flash-lite), GEMINI_SCRIPT_MODEL (default gemini-3.5-flash), aur GEMINI_TRANSCRIBE_MODEL (default = GEMINI_SCRIPT_MODEL wahi, sirf last-resort fallback ke liye use hota hai).
