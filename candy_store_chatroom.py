"""無人糖果商店 - 模擬店員互動"""
"""    TTS + RAG文件索引    """
"""追求真實有情商的購買體驗"""

import os
import time
import cv2
import random
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

from ultralytics import YOLO

from rag_database import load_candy_database, load_store_faq, build_system_prompt


def record_smart(save_dir, samplerate=16000, start_threshold=300, silence_threshold=500, silence_duration=0.6):
    frames, recording, silence_start = [], False, None
    voice_path = os.path.join(str(save_dir), "input.wav")

    with sd.InputStream(samplerate=samplerate, channels=1, dtype="int16") as stream:
        while True:
            data, _ = stream.read(int(samplerate * 0.1))
            volume = np.abs(data).mean()

            if not recording and volume > start_threshold:
                recording = True
                frames.append(data)
                print("[錄音中...]")
            elif recording:
                frames.append(data)
                if volume < silence_threshold:
                    silence_start = silence_start or time.time()
                    if time.time() - silence_start > silence_duration:
                        break
                else:
                    silence_start = None

    wavfile.write(voice_path, samplerate, np.concatenate(frames))
    return Path(voice_path)


def speech_to_text(client, audio_path):
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_path.open("rb")
    )
    return transcription.text.strip()


def gpt_response(client, messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=150
    )
    return response.choices[0].message.content


def text_to_speech(client, text, save_dir):
    output_path = os.path.join(str(save_dir), "output.mp3")
    response = client.audio.speech.create(model="gpt-4o-mini-tts", voice="coral", input=text)
    response.write_to_file(output_path)
    data, sr = sf.read(output_path, dtype="float32")
    sd.play(data, sr)
    sd.wait()

def scan_candy_with_yolo(model_path="yolo11_canday.pt", scan_duration=5):
    """開啟鏡頭掃描糖果，回傳辨識結果"""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    
    class_names = model.names if hasattr(model, 'names') else {}
    detected_items = []
    
    def get_color(cls_id):
        random.seed(cls_id)
        return tuple([int(x) for x in random.choices(range(50, 256), k=3)])

    print(f"[掃描中... {scan_duration}秒]")
    start_time = time.time()
    
    while time.time() - start_time < scan_duration:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = class_names.get(cls, str(cls))
                color = get_color(cls)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if conf > 0.5 and label not in detected_items:
                    detected_items.append(label)

        cv2.imshow('Candy Scanner', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return detected_items



def main():
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 路徑設定
    current_dir = Path("C:/Users/selina/Desktop/candy_store_project")
    data_dir, save_dir = current_dir / "data", current_dir / "saved"
    os.makedirs(save_dir, exist_ok=True)

    # 載入 RAG
    candy_db = load_candy_database(data_dir)
    faq_db = load_store_faq(data_dir)
    system_prompt = build_system_prompt(candy_db, faq_db)
    print(f"載入 {len(candy_db['candies'])} 種糖果, {len(faq_db['faq'])} 條 FAQ")

    print("\n=== 糖果商店語音助理 ===")
    print("說「結帳」掃描 / 說「離開」結束\n")

    # 歡迎
    welcome = "歡迎光臨！請問想要什麼糖果？"
    print(f"小糖: {welcome}")
    text_to_speech(client, welcome, save_dir)

    messages = [{"role": "system", "content": system_prompt}]

    while True:
        print("\n請說話...")
        user_text = speech_to_text(client, record_smart(save_dir))
        print(f"顧客: {user_text}")

        if any(kw in user_text for kw in ["離開", "掰掰", "再見", "結束"]):
            print("小糖: 感謝光臨！")
            text_to_speech(client, "感謝光臨，歡迎下次再來！", save_dir)
            break

        if any(kw in user_text for kw in ["結帳", "付款", "買單"]):
            # 1.掃描糖果
            print("小糖: 請把糖果放鏡頭前")
            text_to_speech(client, "請把糖果放在鏡頭前", save_dir)
            
            candy_items = scan_candy_with_yolo(model_path="models\yolo11_canday.pt", scan_duration=5)
            
            if not candy_items:
                text_to_speech(client, "沒偵測到糖果，請再試一次", save_dir)
                continue
            
            # 計算金額 (需要從 candy_db 查價格)
            total = 0
            for item in candy_items:
                for candy in candy_db["candies"]:
                    if candy["name"].lower() == item.lower() or candy["name_zh"] == item:
                        total += candy["price"]
                        break
            
            price_text = f"掃到 {', '.join(candy_items)}，總共 {total} 元，請投幣"
            print(f"小糖: {price_text}")
            text_to_speech(client, price_text, save_dir)
            
            # 2. 掃描硬幣
            coins = scan_candy_with_yolo(model_path="yolo11_NTD.pt", scan_duration=8)
            
            # 計算投幣金額
            coin_values = {"coin_1": 1, "coin_5": 5, "coin_10": 10, "coin_50": 50}
            paid = sum(coin_values.get(c, 0) for c in coins)
            
            if paid >= total:
                change = paid - total
                if change > 0:
                    result_text = f"收到 {paid} 元，找零 {change} 元，謝謝光臨！"
                else:
                    result_text = f"收到 {paid} 元，謝謝光臨！"
            else:
                result_text = f"金額不足，還差 {total - paid} 元"
            
            print(f"小糖: {result_text}")
            text_to_speech(client, result_text, save_dir)
            continue

        messages.append({"role": "user", "content": user_text})
        reply = gpt_response(client, messages)
        print(f"小糖: {reply}")
        text_to_speech(client, reply, save_dir)
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()