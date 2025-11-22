import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
# import wave
from scipy.io import wavfile
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

def record_smart(
    save_dir: Path,
    samplerate=16000,
    channels=1,
    dtype="int16",
    start_threshold=300,
    silence_threshold=500,
    silence_duration=0.7,
):
    """智慧錄音"""
    import os
    from scipy.io import wavfile
    
    frames = []
    recording = False
    silence_start = None
    
    voice_input_path = os.path.join(str(save_dir), "VoiceChatRoomDemo_input.wav")
    print(f"錄音檔路徑: {voice_input_path}")

    print("請開始說話...")

    with sd.InputStream(samplerate=samplerate, channels=channels, dtype=dtype) as stream:
        while True:
            data, _ = stream.read(int(samplerate * 0.1))
            volume = np.abs(data).mean()

            if not recording and volume > start_threshold:
                recording = True
                frames.append(data)
                print("🔴 錄音中...")
                continue

            if recording:
                frames.append(data)

                if volume < silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_duration:
                        print("錄音結束！")
                        break
                else:
                    silence_start = None

    audio_data = np.concatenate(frames, axis=0)

    # ✅ 用 scipy 寫入，不用 wave 模組
    wavfile.write(voice_input_path, samplerate, audio_data)

    return Path(voice_input_path)

def speech_to_text(client: OpenAI, audio_path: Path):
    """使用GPT模型辨識語音內容，並轉成文字(speech-to-text)"""
    transcription = client.audio.transcriptions.create(
        # transcribe模型，提供speech-to-text功能
        model="gpt-4o-mini-transcribe",
        # 載入語音檔案
        file=audio_path.open("rb")
    )
    text = transcription.text.strip()
    return text


def gpt_response(client: OpenAI, model: str, user_input: str):
    """GPT模型回覆"""
    response = client.responses.create(
        model=model,
        input=user_input
    )
    reply_text = response.output_text
    return reply_text


def text_to_speech(client: OpenAI, text: str, save_dir: Path):
    """GPT模型將文字回覆轉成語音回覆"""
    # 語音輸出檔案路徑
    output_path = save_dir/"VoiceChatRoomDemo_output.mp3"
    # tts模型，提供text-to-speech功能
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        # 聲音種類，參看 https://www.openai.fm/
        voice="alloy",
        input=text
    ) as speech:
        speech.stream_to_file(output_path)

    # 將 mp3 解碼為音訊資料
    data, samplerate = sf.read(output_path, dtype="float32")
    sd.play(data, samplerate)
    # 等待播放結束
    sd.wait()


def api_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key不存在，請檢查.env檔案")

    return api_key


def main():

            # 強制切換工作目錄
    import os
    os.chdir("C:/Users/selina/Documents/candy_store_project")

    # 語音存檔目錄
    save_dir = Path("C:/Users/selina/Documents/candy_store_project/saved")

    # 印出確認
    print(f"工作目錄: {os.getcwd()}")
    print(f"save_dir: {save_dir}")
    print(f"save_dir 存在: {save_dir.exists()}")



    client = OpenAI(api_key=api_key())
    model = "gpt-4.1-nano"
    # 語音存檔目錄
    current_dir = Path(__file__).resolve().parent
    save_dir = current_dir/"saved"
    save_dir.mkdir(parents=True, exist_ok=True)
    print("歡迎進入語音聊天室（說「結束對話」或「離開聊天室」可結束）")

    # 儲存聊天上下文
    messages = [
        {"role": "system", "content": "你是一個友善的語音助理，請簡潔回答使用者的問題"}
    ]

    while True:
        print("請開始說話...")
        voice_input_path = record_smart(save_dir)
        user_text = speech_to_text(client, voice_input_path)
        print(f"你說: {user_text}")

        if any(kw in user_text for kw in ["結束對話", "離開聊天室"]):
            print("已結束語音聊天室！")
            text_to_speech(client, "好的，期待下次再聊。", save_dir)
            break

        messages.append({"role": "user", "content": user_text})
        reply_text = gpt_response(client, model, messages)
        print(f"GPT: {reply_text}")
        text_to_speech(client, reply_text, save_dir)
        messages.append({"role": "assistant", "content": reply_text})




if __name__ == "__main__":
    print("=" * 30)
    main()
    print("=" * 30)
