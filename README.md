#  無人糖果商店

本專案為無人糖果商店的智能互動系統，整合 YOLO 物件偵測、RAG 資料檢索、大型語言模型(LLM)與語音合成(TTS)技術，模擬真人收銀員提供完整的購物體驗。

<img width="319" height="739" alt="螢幕擷取畫面 2025-12-02 232714" src="https://github.com/user-attachments/assets/2d8b1381-2fee-47f6-97d7-acda579c3e1a" />


## 功能

- **YOLO 糖果辨識**：即時辨識 11 種糖果商品
- **YOLO 硬幣辨識**：辨識台幣硬幣面額（1, 5, 10, 50 元）
- **RAG 推薦系統**：根據顧客需求推薦適合的糖果
- **語音互動**：Whisper 語音辨識 + GPT 回覆 + TTS 語音播報
  
  ```
顧客進入商店
    ↓
[語音互動] 顧客詢問商品或尋求推薦
    ↓
[RAG 檢索] 從資料庫檢索商品資訊
    ↓
[LLM 生成] 產生個性化推薦與回應
    ↓
[TTS 語音] 播放語音回應給顧客
    ↓
顧客選定商品並放置結帳區
    ↓
[YOLO 辨識] 辨識商品種類與數量
    ↓
系統語音告知總金額
    ↓
顧客投入硬幣
    ↓
[YOLO 辨識] 辨識並計算硬幣金額
    ↓
確認付款完成 → 結帳成功！
```

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
candy_store_project/
├── data/
│   ├── candy_database.json       # 糖果商品資料庫
│   └── store_faq.json            # 商店常見問題資料庫
├── models/
│   ├── yolo11_canday.pt          # 糖果辨識模型
│   └── yolo11_NTD.pt             # 硬幣辨識模型
├── saved/                        # 語音檔案暫存目錄
├── test/
│   └── 03-YOLOv11_candy.py       # YOLO 測試腳本
├── training/
│   ├── 2025T105train_yolov11_models.py  # 模型訓練腳本
│   └── my_model.zip              # 訓練資料集
├── audio_prompts/                # 語音提示檔案
├── rag_database.py               # RAG 資料庫模組
├── candy_store_chatroom.py       # 主程式（命令列版本）
├── .env                          # 環境變數設定（需自行建立）
├── LICENSE                       # 授權條款
├── README.md                     # 專案說明文件
└── requirements.txt              # Python 套件需求
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
- **影像處理**: OpenCV

## 模型訓練

### 訓練糖果辨識模型
```bash
cd training
python 2025T105train_yolov11_models.py
```

訓練完成後將模型檔案移至 `models/` 目錄。

### 資料集準備

1. 收集糖果圖片並標註
2. 使用 Roboflow 或 LabelImg 進行標註
3. 匯出為 YOLO 格式
4. 放置於 `training/my_model.zip`

## 測試腳本

### 測試 YOLO 辨識功能
```bash
cd test
python 03-YOLOv11_candy.py
```

<img width="318" height="205" alt="image" src="https://github.com/user-attachments/assets/1bdacb05-a506-43ee-9731-9af3a9ea658c" /></br>
<img width="600" height="356" alt="image" src="https://github.com/user-attachments/assets/c502bf62-2ffb-4602-962d-2d265f25dda6" />
<img width="600" height="356" alt="image" src="https://github.com/user-attachments/assets/0947bee8-7a8d-4c1b-b3c3-a6fc874ebf2a" />


## 訓練成果


### 糖果模型
- mAP50: 0.995
- Precision: 0.992
- Recall: 0.999

### 硬幣模型
- 待重新訓練（更換攝影機後信心值下降）
