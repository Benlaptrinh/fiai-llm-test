"""
Synthetic data generator for FI-AI F&B Multi-Agent LLM test.
Generates 4000 structured samples across 4 intents with:
- Balanced class distribution
- 10% ambiguous / hard samples
- Vietnamese + English queries
- Full checkpoint/resume support
"""

from __future__ import annotations

import csv
import random
import re
import time
from pathlib import Path
from typing import Optional

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ─── Intent distribution ───────────────────────────────────────────────────
TARGET_TOTAL = 4000
INTENTS = ["order", "consultant", "faq", "ignore"]
# order=40%, consultant=30%, faq=20%, ignore=10%  (real-world skew)
INTENT_RATIOS = {"order": 0.40, "consultant": 0.30, "faq": 0.20, "ignore": 0.10}
HARD_RATIO = 0.10  # 10% ambiguous queries

# ─── Menu items (subset for template generation) ───────────────────────────
COFFEE_ITEMS = [
    "Phin Sữa Đá", "Phin Đen Đá", "Bạc Xỉu", "Latte", "Americano",
    "Cappuccino", "Espresso", "Cold Brew", "Mocha", "Caramel Macchiato",
    "Cà Phê Muối", "Cà Phê Dừa", "Vietnamese Latte", "Hazelnut Latte",
    "Vanilla Latte", "Flat White", "Iced Coffee", "Double Espresso",
    "Cà Phê Sữa Nóng", "Cà Phê Đen Nóng",
]
TEA_ITEMS = [
    "Trà Sen Vàng", "Trà Đào Cam Sả", "Trà Vải", "Trà Xoài Nhiệt Đới",
    "Trà Chanh Mật Ong", "Trà Oolong Sữa", "Trà Nhài", "Trà Dâu",
    "Trà Cam Quế", "Trà Táo Bạc Hà", "Matcha Latte", "Hồng Trà Sữa",
    "Trà Sữa Trân Châu", "Trà Lài Mật Ong", "Trà Atiso Đỏ",
    "Trà Đào Ít Ngọt", "Trà Thanh Đào", "Trà Kiwi", "Trà Việt Quất",
    "Trà Gừng Mật Ong",
]
FREEZE_ITEMS = [
    "Freeze Trà Xanh", "Freeze Chocolate", "Freeze Caramel", "Freeze Cookies",
    "Freeze Việt Quất", "Freeze Dâu", "Freeze Matcha Đậu Đỏ", "Freeze Cà Phê",
    "Freeze Bạc Xỉu", "Freeze Chanh Tuyết", "Freeze Xoài", "Freeze Đào",
    "Freeze Sữa Chua Dâu", "Freeze Sữa Chua Việt Quất", "Freeze Socola Bạc Hà",
]
FOOD_ITEMS = [
    "Bánh Mì Que", "Croissant", "Tiramisu", "Bánh Chuối", "Bánh Phô Mai",
    "Muffin Chocolate", "Sandwich Gà", "Sandwich Cá Ngừ", "Bánh Táo",
    "Brownie", "Bánh Su Kem", "Bánh Mì Pate", "Bánh Quy Bơ", "Donut Chocolate",
    "Bánh Mặn Phô Mai",
]
ALL_ITEMS = COFFEE_ITEMS + TEA_ITEMS + FREEZE_ITEMS + FOOD_ITEMS

SIZES = ["S", "M", "L"]
QUANTITIES = ["1", "2", "3", "một", "hai", "ba"]
MODIFIERS = {
    "size S": "size S", "size M": "size M", "size L": "size L",
    "ít đường": "ít đường", "ít ngọt": "ít ngọt", "không đường": "không đường",
    "nóng": "nóng", "lạnh": "lạnh", "đá": "đá",
    "nhiều đá": "nhiều đá", "ít đá": "ít đá",
}

# ─── Intent templates ─────────────────────────────────────────────────────
ORDER_TEMPLATES_VN = [
    "cho anh {qty} ly {item} {size}",
    "em lấy {qty} {item} {size}",
    "mình muốn order {qty} {item}",
    "cho a {qty} {item}",
    "thêm cho tôi {qty} {item} {size}",
    "gọi {qty} ly {item} {mod}",
    "tính cho tôi {qty} {item} {size}",
    "mình order {qty} {item} {mod}",
    "đặt cho {qty} {item} {size}",
    "ship {qty} {item} {size}",
    "mang cho em {qty} {item}",
    "{qty} ly {item} {size} nha",
    "cho {qty} {item} {size} em",
    # More diverse openings
    "cho {qty} {item} {size} đi",
    "lấy {qty} {item} {size} nhé",
    "a ơi em gọi {qty} {item} {size}",
    "em muốn order {qty} {item} {size}",
    "anh ơi cho em {qty} {item}",
    "đặt {qty} ly {item} {size} nha",
    "mình lấy {qty} {item} {size}",
    "cho em {qty} {item} {size} em",
    "a tính cho em {qty} {item} đi",
    "em order {qty} {item} {size} nha",
    "gọi cho em {qty} {item} {size}",
    "tôi muốn mua {qty} {item} {size}",
    "lấy cho {qty} ly {item} {size}",
    "anh ơi {qty} {item} {size} nha",
    "em cần {qty} {item} {size}",
    "ship em {qty} {item} {size} nha",
    "cho {qty} {item} {mod} đi",
    "order {qty} {item} {size} nha",
    "đặt bàn và lấy {qty} {item} {size}",
    "a ơi thêm {qty} {item} đi",
    "em muốn thêm {qty} {item} {size}",
    "a lấy cho em {qty} {item}",
    "cho {qty} cốc {item} {size}",
    "em gọi {qty} {item} {size} nha",
    "mang ra {qty} {item} {size} nhé",
    "bên em có {qty} {item} {size} không",
    "có bán {qty} {item} {size} không",
    "quán có {qty} {item} {size} không",
    "cần {qty} {item} {size} nhé",
]
ORDER_TEMPLATES_EN = [
    "I'd like {qty} {item} {size} please",
    "can I get {qty} {item}",
    "order {qty} {item} {size}",
    "one {item} {size}",
    "two {item} please",
    "I want {qty} {item}",
    "get me {qty} {item} {size}",
    "can I have {qty} {item} {size}",
    "I'd like to order {qty} {item}",
    "please give me {qty} {item} {size}",
]

CONSULTANT_TEMPLATES_VN = [
    "có gì ngon không em",
    "gợi ý cho anh món ít ngọt",
    "anh không uống caffeine thì nên chọn gì",
    "tư vấn cho em món mát mát",
    "món nào bán chạy nhất",
    "em thích trà trái cây thì uống gì",
    "anh thích cà phê đậm thì nên uống gì",
    "có món nào nhẹ nhẹ không",
    "đồ uống nào hợp với bánh ngọt",
    "gợi ý món dưới 45k",
    "món nào không ngọt lắm",
    "anh thích trà hoa quả, recommend gì",
    "có gì uống mát không",
    "món gì cho người không uống cà phê",
    "tư vấn cho em món ngon giá rẻ",
    "nên uống gì vào buổi sáng",
    "món nào phù hợp cho buổi chiều",
    "anh thích vị nhẹ, gợi ý gì",
    "có món gì cho người ăn kiêng không",
    "món nào ít calories",
    "gợi ý đồ uống không đường",
    "có món nào giống latte nhưng không cà phê",
    "anh không thích ngọt, tư vấn gì",
    "món nào hợp với khẩu vị trẻ em",
    "có đồ uống healthy không",
    "recommend something refreshing please",
    "what's your best seller today",
    "suggest a drink for someone who hates coffee",
    "what would you recommend for hot weather",
    "gợi ý món ngon cho ngày nóng",
    "tư vấn đồ uống cho người mới uống cà phê",
    "món nào thay thế được cà phê",
    "có trà nào ngon không",
    "gợi ý cho khách không ăn đường",
    "anh thích đồ uống chua, có gì",
]
CONSULTANT_TEMPLATES_EN = [
    "what do you recommend",
    "suggest something sweet but not too sweet",
    "what's good here",
    "recommend a drink without caffeine",
    "what's the most popular item",
    "I prefer tea-based drinks, what should I get",
    "suggest a light refreshing drink",
    "what's good for hot weather",
    "recommend something under 50000",
]

FAQ_TEMPLATES_VN = [
    "wifi tên gì vậy",
    "quán mấy giờ mở cửa",
    "mấy giờ đóng cửa",
    "có thanh toán thẻ không",
    "có thanh toán qr không",
    "có giao hàng không",
    "có xuất hóa đơn không",
    "có chỗ ngồi làm việc không",
    "có ổ cắm điện không",
    "có chỗ đậu xe không",
    "có đặt bàn trước không",
    "có size lớn không",
    "quán có nhận đặt số lượng lớn không",
    "có giảm giá sinh viên không",
    "có nhận thanh toán công ty không",
    "có chương trình thành viên không",
    "có phòng họp không",
    "có giới hạn thời gian ngồi không",
    "có món cho trẻ em không",
    "có món chay không",
    "wifi password là gì",
    "quán mở cửa thứ mấy",
    "có hỗ trợ sinh nhật không",
    "có hoàn tiền không",
    "có đổi món sau khi đặt không",
    "có nhận đặt tiệc không",
    "có món nóng không",
    "có món lạnh không",
    "có bánh gì ăn kèm",
    "có combo không",
    "có khuyến mãi gì hôm nay không",
    "có nhận đặt qua điện thoại không",
    "có chỗ hút thuốc không",
    "quán có nhận thẻ membership không",
    "có xuất hóa đơn VAT không",
    "có giao tận nơi không",
]
FAQ_TEMPLATES_EN = [
    "what is the wifi name",
    "what time do you open",
    "do you deliver",
    "can I pay by card",
    "do you have qr payment",
    "what are your opening hours",
    "is there parking available",
    "do you have vegan options",
    "is there a student discount",
    "can I pay by cash",
    "do you accept credit cards",
    "what time do you close",
    "is there a private room",
    "can I make a reservation",
    "do you have vegetarian options",
    "do you do delivery",
    "what's your return policy",
    "do you offer birthday discounts",
    "is there a kid's menu",
]

IGNORE_TEMPLATES_VN = [
    "hello", "hi", "haha", "alo", "ừm", "ờ", "à à",
    "ok", "okay", "hmm", "uhhh", "yes", "no",
    "tạm biệt", "cảm ơn", "thanks", "bye", "chào",
    "nghe không", "alo alo", "hú hú", "hjc",
    "không", "vâng", "dạ", "được", "ừ",
    "haha thú vị", "ok cảm ơn", "bye bye",
    "cảm ơn nha", "thanks nhé", "chào nhé",
    "huhu", "hichic", "uhhhhh", "hjx",
]
IGNORE_TEMPLATES_EN = [
    "hello there", "hey", "yo", "what's up",
    "test", "abc xyz", "123", "cool", "nice",
    "good morning", "good afternoon", "good evening",
    "ty", "tnx", "thx", "cheers",
]


def _apply_hard_mutation(text: str, intent: str) -> str:
    """Add noise / ambiguity to simulate hard samples."""
    noise = random.choice([
        "???", "...", "  ", "  ???", ".",
        "", " à", " nhỉ", " vậy", " ạ",
    ])
    if intent == "order":
        # Ambiguous: could be order or consultant
        if random.random() < 0.5:
            text = text.replace("cho anh", random.choice(["cho tôi xem", "cho mình", ""]))
    elif intent == "consultant":
        if random.random() < 0.5:
            text += " nhỉ"
    elif intent == "faq":
        if random.random() < 0.5:
            text = text.replace("có", random.choice(["cho hỏi có", "cho mình hỏi", ""]))
    return text + noise


def _fill_template(template: str) -> str:
    """Fill a template with random menu items and modifiers."""
    item = random.choice(ALL_ITEMS)
    qty = random.choice(QUANTITIES)
    size = random.choice(SIZES)
    mod = random.choice(list(MODIFIERS.values()))

    result = template.format(
        item=item, qty=qty, size=size, mod=mod
    )
    # Remove double spaces
    result = re.sub(r"\s+", " ", result).strip()
    return result


def _generate_intent_queries(intent: str, count: int) -> list[dict]:
    """
    Generate queries for a specific intent using deterministic combinations
    to avoid memory issues from large while-loops.
    """
    rows = []

    if intent == "order":
        templates = ORDER_TEMPLATES_VN + ORDER_TEMPLATES_EN
        # Build large pool via deterministic combinations
        combos = []
        seen: set[str] = set()
        for template in templates:
            for item in ALL_ITEMS:
                for qty in QUANTITIES:
                    for size in SIZES:
                        for mod_val in list(MODIFIERS.values()):
                            text = template.format(
                                item=item, qty=qty, size=size, mod=mod_val
                            )
                            text = re.sub(r"\s+", " ", text).strip()
                            if text not in seen:
                                seen.add(text)
                                combos.append(text)
        random.shuffle(combos)
        for text in combos[:count]:
            is_hard = random.random() < HARD_RATIO
            if is_hard:
                text = _apply_hard_mutation(text, intent)
            rows.append({
                "text": text,
                "intent": intent,
                "is_noise": False,
                "is_hard": is_hard,
                "language": "vi" if any(c >= '\u0080' for c in text) else "en",
            })

    elif intent == "consultant":
        # Deterministic: template + item + preference combinations
        base_templates = CONSULTANT_TEMPLATES_VN + CONSULTANT_TEMPLATES_EN
        combos: list[str] = []
        seen: set[str] = set()
        for template in base_templates:
            for item in ALL_ITEMS:
                for pref in ["ít ngọt", "không caffeine", "mát", "đá"]:
                    if "{item}" in template:
                        text = template.format(item=item)
                    else:
                        text = f"{template} {item} {pref}"
                    text = re.sub(r"\s+", " ", text).strip()
                    if text not in seen:
                        seen.add(text)
                        combos.append(text)
                    # Also standalone
                    if template not in seen:
                        seen.add(template)
                        combos.append(template)
        random.shuffle(combos)
        for text in combos[:count]:
            is_hard = random.random() < HARD_RATIO
            if is_hard:
                text = _apply_hard_mutation(text, intent)
            rows.append({
                "text": text,
                "intent": intent,
                "is_noise": False,
                "is_hard": is_hard,
                "language": "vi" if any(c >= '\u0080' for c in text) else "en",
            })

    elif intent == "faq":
        # Deterministic: base question + prefix/suffix paraphrasing
        base_qs = FAQ_TEMPLATES_VN + FAQ_TEMPLATES_EN
        combos: list[str] = []
        seen: set[str] = set()
        prefixes = ["cho em hỏi", "cho mình hỏi", "hỏi chút", "cho hỏi", ""]
        suffixes = ["?", " vậy", " ạ", " nhỉ", " không", " nào"]
        for base in base_qs:
            for prefix in prefixes:
                for suffix in suffixes:
                    text = f"{prefix} {base}{suffix}".strip()
                    text = re.sub(r"\s+", " ", text).strip()
                    if text not in seen:
                        seen.add(text)
                        combos.append(text)
        random.shuffle(combos)
        for text in combos[:count]:
            is_hard = random.random() < HARD_RATIO
            if is_hard:
                text = _apply_hard_mutation(text, intent)
            rows.append({
                "text": text,
                "intent": intent,
                "is_noise": False,
                "is_hard": is_hard,
                "language": "vi" if any(c >= '\u0080' for c in text) else "en",
            })

    else:  # ignore
        templates = IGNORE_TEMPLATES_VN + IGNORE_TEMPLATES_EN
        combos: list[str] = []
        seen: set[str] = set()
        noises = ["", "...", "???", "  ", ".", " ạ", " nhé"]
        for base in templates:
            for noise in noises:
                text = (base + noise).strip()
                text = re.sub(r"\s+", " ", text)
                if text not in seen:
                    seen.add(text)
                    combos.append(text)
        random.shuffle(combos)
        for text in combos[:count]:
            rows.append({
                "text": text,
                "intent": intent,
                "is_noise": True,
                "is_hard": False,
                "language": "vi" if any(c >= '\u0080' for c in text) else "en",
            })

    return rows


def generate_synthetic_queries(
    output_path: Path,
    target: int = TARGET_TOTAL,
    seed: int = 42,
    checkpoint_path: Optional[Path] = None,
) -> list[dict]:
    """
    Generate TARGET_TOTAL balanced synthetic queries.

    Features:
    - Checkpoint/resume: saves intermediate CSV and resumes from it
    - Deduplication: removes exact duplicate texts
    - Progress logging every 10%
    - Hard sample augmentation (10% ambiguous queries)
    - Bilingual: Vietnamese + English
    """
    random.seed(seed)

    checkpoint_path = checkpoint_path or output_path.with_suffix(".checkpoint.csv")

    # Resume from checkpoint if exists
    existing_texts: set[str] = set()
    existing_rows: list[dict] = []
    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_texts.add(row["text"])
                existing_rows.append(row)
        print(f"[generate_data] Resuming from checkpoint: {len(existing_rows)} rows")

    # Build intent counts from existing
    current_counts = {i: 0 for i in INTENTS}
    for row in existing_rows:
        current_counts[row["intent"]] = current_counts.get(row["intent"], 0) + 1

    # Target per intent
    intent_targets = {i: int(target * INTENT_RATIOS[i]) for i in INTENTS}

    all_rows = list(existing_rows)
    all_texts = set(existing_texts)

    start = time.time()
    for intent in INTENTS:
        needed = intent_targets[intent] - current_counts[intent]
        if needed <= 0:
            print(f"[{intent}] already at target ({current_counts[intent]}), skipping")
            continue

        print(f"[{intent}] generating {needed} samples...")
        new_rows = _generate_intent_queries(intent, needed)

        added = 0
        for row in new_rows:
            if row["text"] not in all_texts:
                all_rows.append(row)
                all_texts.add(row["text"])
                added += 1

                # Checkpoint every 500 rows
                if added % 500 == 0:
                    with checkpoint_path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["text", "intent", "is_noise", "is_hard", "language"])
                        writer.writeheader()
                        writer.writerows(all_rows)
                    elapsed = time.time() - start
                    print(f"  [{intent}] checkpoint: {added}/{needed} unique, {elapsed:.1f}s")

        print(f"[{intent}] added {added} unique (needed {needed}, {added/needed*100:.1f}%)")

    # Final shuffle
    random.shuffle(all_rows)

    # Save final CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "intent", "is_noise", "is_hard", "language"])
        writer.writeheader()
        writer.writerows(all_rows)

    elapsed = time.time() - start
    print(f"\n[generate_data] Done: {len(all_rows)} samples in {elapsed:.1f}s")
    print(f"  Intent distribution:")
    for intent in INTENTS:
        count = sum(1 for r in all_rows if r["intent"] == intent)
        print(f"    {intent}: {count} ({count/len(all_rows)*100:.1f}%)")

    # Cleanup checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return all_rows


def generate_menu() -> None:
    """Generate ~100 structured menu rows."""
    rows: list[dict] = []
    item_id = 1

    all_groups = [
        ("Coffee", COFFEE_ITEMS),
        ("Tea", TEA_ITEMS),
        ("Freeze", FREEZE_ITEMS),
        ("Food", FOOD_ITEMS),
    ]

    def price_for_category(category: str, size: str) -> int:
        base = {
            "Coffee": random.choice([35000, 39000, 45000, 49000, 55000]),
            "Tea": random.choice([39000, 45000, 49000, 55000, 59000]),
            "Freeze": random.choice([55000, 59000, 65000, 69000]),
            "Food": random.choice([19000, 29000, 35000, 45000, 55000]),
        }[category]
        if size == "M":
            return base + 5000
        if size == "L":
            return base + 10000
        return base

    for category, items in all_groups:
        for name in items:
            sizes = ["Regular"] if category == "Food" else SIZES
            for size in sizes:
                if len(rows) >= 100:
                    break

                if category == "Coffee":
                    caffeine = random.choice(["medium", "high"])
                    sweetness = random.choice(["low", "medium", "high"])
                    tags = "coffee,caffeine,strong,popular"
                    ingredients = "cà phê, sữa, đá" if "Sữa" in name or "Latte" in name else "cà phê, nước, đá"
                    description = f"{name} size {size}, phù hợp khách thích cà phê và hương vị đậm."
                elif category == "Tea":
                    caffeine = random.choice(["low", "medium"])
                    sweetness = random.choice(["low", "medium", "high"])
                    tags = "tea,fruit,refreshing,low-caffeine"
                    ingredients = "trà, trái cây, syrup, đá"
                    description = f"{name} size {size}, vị thanh mát, phù hợp khách thích đồ uống nhẹ."
                elif category == "Freeze":
                    caffeine = random.choice(["none", "low", "medium"])
                    sweetness = random.choice(["medium", "high"])
                    tags = "freeze,cold,sweet,ice-blended"
                    ingredients = "đá xay, sữa, kem, syrup"
                    description = f"{name} size {size}, đồ uống đá xay mát lạnh, phù hợp khách thích ngọt."
                else:
                    caffeine = "none"
                    sweetness = random.choice(["low", "medium"])
                    tags = "food,snack,bakery"
                    ingredients = "bánh, bơ, nhân mặn hoặc ngọt"
                    description = f"{name}, món ăn nhẹ dùng kèm đồ uống."

                rows.append({
                    "id": item_id,
                    "name": name,
                    "category": category,
                    "size": size,
                    "price": price_for_category(category, size),
                    "caffeine": caffeine,
                    "sweetness": sweetness,
                    "tags": tags,
                    "ingredients": ingredients,
                    "description": description,
                })
                item_id += 1

            if len(rows) >= 100:
                break

    with (DATA_DIR / "menu.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["id", "name", "category", "size", "price",
                        "caffeine", "sweetness", "tags", "ingredients", "description"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated menu.csv: {len(rows)} rows")


def generate_faq() -> None:
    """Generate ~30 FAQ rows."""
    faqs = [
        ("Quán mở cửa mấy giờ?", "Quán mở cửa từ 7 giờ sáng đến 10 giờ tối mỗi ngày."),
        ("Quán đóng cửa lúc mấy giờ?", "Quán đóng cửa lúc 10 giờ tối."),
        ("Wifi tên gì?", "Wifi của quán là Highlands_Guest, mật khẩu vui lòng hỏi nhân viên."),
        ("Có thanh toán bằng thẻ không?", "Quán hỗ trợ tiền mặt, thẻ ngân hàng và ví điện tử."),
        ("Có thanh toán QR không?", "Quán có hỗ trợ thanh toán QR qua một số ví điện tử."),
        ("Có giao hàng không?", "Quán hỗ trợ giao hàng qua các ứng dụng đối tác."),
        ("Có giao hàng nội bộ không?", "Quán thường giao hàng qua đối tác giao nhận."),
        ("Có chỗ ngồi làm việc không?", "Một số chi nhánh có khu vực ngồi làm việc và ổ cắm điện."),
        ("Có ổ cắm điện không?", "Một số chi nhánh có ổ cắm điện gần khu vực ngồi làm việc."),
        ("Có xuất hóa đơn không?", "Quán có hỗ trợ xuất hóa đơn theo yêu cầu."),
        ("Có món ít ngọt không?", "Khách có thể yêu cầu giảm đường hoặc chọn món ít ngọt."),
        ("Có món không caffeine không?", "Có thể chọn trà trái cây, freeze hoặc một số món không cà phê."),
        ("Có đồ ăn nhẹ không?", "Quán có bánh mì, croissant, tiramisu và một số món ăn nhẹ."),
        ("Có giảm giá sinh viên không?", "Khuyến mãi phụ thuộc từng thời điểm và từng chi nhánh."),
        ("Có chỗ đậu xe không?", "Một số chi nhánh có chỗ đậu xe, vui lòng hỏi nhân viên tại quầy."),
        ("Có phòng họp không?", "Một số chi nhánh có không gian phù hợp họp nhóm nhỏ."),
        ("Có đặt bàn trước không?", "Khách có thể liên hệ chi nhánh để hỏi về đặt bàn trước."),
        ("Có đổi món sau khi thanh toán không?", "Việc đổi món sau thanh toán phụ thuộc tình trạng xử lý đơn hàng."),
        ("Có hoàn tiền không?", "Chính sách hoàn tiền phụ thuộc từng trường hợp cụ thể."),
        ("Có size lớn không?", "Nhiều món có size S, M và L tùy loại đồ uống."),
        ("Có món nóng không?", "Một số món cà phê và trà có thể phục vụ nóng."),
        ("Có món lạnh không?", "Hầu hết đồ uống đều có phiên bản lạnh hoặc đá."),
        ("Có món cho trẻ em không?", "Khách có thể chọn một số món ít caffeine hoặc không caffeine."),
        ("Có món chay không?", "Một số món ăn nhẹ có thể phù hợp, vui lòng hỏi nhân viên để xác nhận."),
        ("Có nhận đặt số lượng lớn không?", "Quán có thể hỗ trợ đơn số lượng lớn tùy chi nhánh."),
        ("Có hỗ trợ sinh nhật không?", "Chính sách hỗ trợ sinh nhật phụ thuộc từng chương trình."),
        ("Có giới hạn thời gian ngồi không?", "Thông thường không giới hạn nếu quán không quá đông."),
        ("Có nhận thanh toán công ty không?", "Quán có hỗ trợ hóa đơn theo thông tin khách cung cấp."),
        ("Có chương trình thành viên không?", "Chương trình thành viên phụ thuộc chính sách hiện hành."),
        ("Có nhận góp ý dịch vụ không?", "Khách có thể góp ý trực tiếp với nhân viên hoặc qua kênh hỗ trợ."),
    ]

    with (DATA_DIR / "faq.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "question", "answer"])
        writer.writeheader()
        for idx, (question, answer) in enumerate(faqs, start=1):
            writer.writerow({"id": idx, "question": question, "answer": answer})

    print(f"Generated faq.csv: {len(faqs)} rows")


def generate_docs() -> None:
    """Generate policy and system guideline docs (~30 chunks)."""
    docs = [
        "Chính sách đặt hàng:\nKhách có thể đặt một hoặc nhiều món. Nếu thiếu số lượng, hệ thống mặc định là 1. Nếu thiếu size, hệ thống cần hỏi lại size trước khi xác nhận.",
        "Chính sách xác nhận đơn:\nOrder Agent phải xác nhận lại tên món, số lượng, size và giá nếu thông tin đã đủ. Không được tự thêm món ngoài menu.",
        "Chính sách tư vấn ít ngọt:\nNếu khách nói thích ít ngọt, ưu tiên gợi ý trà trái cây, Americano, Phin Đen Đá hoặc món có sweetness low.",
        "Chính sách tư vấn không caffeine:\nNếu khách yêu cầu không caffeine, không gợi ý cà phê. Ưu tiên Freeze, trà trái cây hoặc món có caffeine none.",
        "Chính sách tư vấn cà phê đậm:\nNếu khách thích vị đậm, ưu tiên Phin Đen Đá, Phin Sữa Đá, Cold Brew hoặc Espresso.",
        "Chính sách tư vấn đồ uống mát:\nNếu khách muốn món mát, ưu tiên trà trái cây, freeze hoặc đồ uống đá.",
        "Chính sách tư vấn đồ uống phổ biến:\nNếu khách hỏi món bán chạy, có thể gợi ý Phin Sữa Đá, Trà Sen Vàng, Trà Đào Cam Sả hoặc Freeze Trà Xanh.",
        "Chính sách upsell:\nSau khi khách đặt đồ uống, có thể gợi ý thêm bánh mì que, croissant hoặc tiramisu một cách lịch sự.",
        "Chính sách chống hallucination:\nAgent không được bịa giá, bịa món, bịa size hoặc bịa chính sách. Nếu không có dữ liệu, phải nói chưa có thông tin xác nhận.",
        "Chính sách FAQ:\nFAQ Agent chỉ trả lời dựa trên FAQ hoặc tài liệu nội bộ. Nếu context không liên quan, trả lời rằng hiện chưa có dữ liệu.",
        "Chính sách session:\nHệ thống cần giữ các lượt hội thoại gần nhất để hiểu câu tiếp theo của khách, ví dụ khách nói 'thêm 1 ly nữa'.",
        "Chính sách cache:\nCác câu hỏi lặp lại như wifi, giờ mở cửa, thanh toán nên được cache để giảm latency.",
        "Chính sách xử lý câu mơ hồ:\nNếu khách nói 'có gì ngon không', hệ thống nên hỏi thêm khẩu vị hoặc gợi ý món phổ biến.",
        "Chính sách xử lý câu noise:\nCác câu như hello, haha, ừm, alo nên được phân loại ignore và phản hồi nhẹ nhàng.",
        "Chính sách đa ngôn ngữ:\nNếu khách hỏi bằng tiếng Anh, hệ thống có thể trả lời bằng tiếng Anh ở mức cơ bản.",
        "Chính sách TTS:\nCâu trả lời nên ngắn, tự nhiên, dễ đọc thành tiếng, tránh câu quá dài.",
        "Chính sách giá tiền:\nKhi đọc giá tiền, có thể diễn đạt 39000 VND là 39 nghìn đồng hoặc 39k.",
        "Chính sách size:\nNếu món có nhiều size, hỏi lại size khi khách chưa nói. Nếu món chỉ có Regular, không cần hỏi size.",
        "Chính sách món ăn:\nFood Agent hoặc Order Agent có thể xử lý bánh và đồ ăn nhẹ như một loại menu item.",
        "Chính sách gợi ý theo ngân sách:\nNếu khách nói ngân sách thấp, ưu tiên món giá dưới 45000 VND.",
        "Chính sách gợi ý theo nhóm:\nNếu khách đặt nhiều món, có thể gợi ý combo đồ uống và bánh nhưng không tự tạo combo nếu menu không có.",
        "Chính sách dữ liệu menu:\nMenuItem cần có tên, category, size, price, caffeine, sweetness, tags và description để hỗ trợ retrieval.",
        "Chính sách retrieval:\nRAG nên ưu tiên menu cho order và consultant, ưu tiên FAQ/document cho câu hỏi thông tin chung.",
        "Chính sách reranking:\nProduction có thể thêm reranker để cải thiện độ chính xác top-k.",
        "Chính sách graph RAG:\nProduction có thể thay ChromaDB bằng Neo4j để lưu MenuItem, Chunk, Entity và relationships.",
        "Chính sách lỗi hệ thống:\nNếu LLM quá tải hoặc lỗi, hệ thống cần trả lời graceful thay vì crash.",
        "Chính sách bảo trì dữ liệu:\nKhi menu hoặc FAQ thay đổi, cần chạy lại ingestion để cập nhật embedding.",
        "Chính sách benchmark:\nCần đo router accuracy, retrieval coverage, average latency, p95 latency và cache-hit latency.",
        "Chính sách bảo mật:\nKhông lưu thông tin nhạy cảm của khách trong session history.",
        "Chính sách production:\nMVP local dùng Ollama và ChromaDB, production có thể dùng vLLM/SGLang và Neo4j trên GPU server.",
    ]

    (DATA_DIR / "docs.txt").write_text("\n\n".join(docs), encoding="utf-8")
    print(f"Generated docs.txt: {len(docs)} chunks")


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    random.seed(seed)

    print("Generating menu.csv, faq.csv, docs.txt...")
    generate_menu()
    generate_faq()
    generate_docs()

    print("\nGenerating synthetic_queries.csv (4000 samples)...")
    rows = generate_synthetic_queries(
        output_path=DATA_DIR / "synthetic_queries.csv",
        target=TARGET_TOTAL,
        seed=seed,
    )
    print(f"\nTotal queries: {len(rows)}")
    print("Synthetic dataset generation completed.")
