#  無人糖果商店

結合 YOLO 物件偵測、RAG 推薦系統、語音互動的無人商店專案。

## 功能

- **YOLO 糖果辨識**：即時辨識 11 種糖果商品
- **YOLO 硬幣辨識**：辨識台幣硬幣面額（1, 5, 10, 50 元）
- **RAG 推薦系統**：根據顧客需求推薦適合的糖果
- **語音互動**：Whisper 語音辨識 + GPT 回覆 + TTS 語音播報

## 安裝

```bash
# 1. 安裝套件
pip install -r requirements.txt

```

## 使用

### 語音聊天室模式
```bash
python candy_store_chatroom.py
```

### YOLO 辨識測試
```bash
python 03-YOLOv11_candy.py
```

## 專案結構

```
無人糖果商店/
├── models/                 # YOLO 模型
│   ├── yolo11_candy.pt
│   └── yolo11_coin.pt
├── data/                   # RAG 資料庫
│   ├── candy_database.json
│   └── store_faq.json
├── audio/prompts/          # 預錄語音
├── candy_store_chatroom.py # 主程式
└── requirements.txt

```

## 可辨識糖果（11 種）

| 糖果 | 價格 |
|------|------|
| Skittles 彩虹糖 | $45 |
| Snickers 士力架 | $35 |
| Airheads 軟糖 | $30 |
| M&M's 花生 | $40 |
| M&M's 原味 | $38 |
| Gummy Worms 蟲蟲軟糖 | $32 |
| Milky Way | $40 |
| Nerds 書呆子糖 | $28 |
| Starburst 星爆軟糖 | $35 |
| Three Musketeers | $38 |
| Twizzlers 扭扭糖 | $33 |

## 技術

- **物件偵測**：YOLOv11 (Ultralytics)
- **語音辨識**：OpenAI Whisper (gpt-4o-mini-transcribe)
- **語言模型**：GPT-4o-mini
- **語音合成**：OpenAI TTS (gpt-4o-mini-tts)

## 訓練成果

### 糖果模型
- mAP50: 0.995
- Precision: 0.992
- Recall: 0.999

### 硬幣模型
- 待重新訓練（更換攝影機後信心值下降）
