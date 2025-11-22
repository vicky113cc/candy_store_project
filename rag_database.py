"""RAG 資料庫模組"""

import json
from pathlib import Path


def load_candy_database(data_dir: Path) -> dict:
    db_path = data_dir / "candy_database.json"
    with open(db_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_store_faq(data_dir: Path) -> dict:
    faq_path = data_dir / "store_faq.json"
    with open(faq_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_candy_info(candy_db: dict) -> str:
    lines = []
    for candy in candy_db["candies"]:
        lines.append(f"{candy['id']}. {candy['name']} {candy['name_zh']} - ${candy['price']}")
        lines.append(f"   口味: {candy['flavor']} / 特色: {candy['description']}")
    return "\n".join(lines)


def build_system_prompt(candy_db: dict, faq_db: dict) -> str:
    candy_info = format_candy_info(candy_db)
    faq_examples = "\n".join([
        f"Q: {item['question']}\nA: {item['answer']}"
        for item in faq_db["faq"][:3]
    ])
    
    return f"""你是糖果店 AI 店員「小糖」，親切簡短回答（2-3句）。

商品:
{candy_info}

範例:
{faq_examples}
"""