"""
Data generation script for C2.1 Intent Extraction SFT.

Generates ~3000 train + ~800 test samples with structured labels:
  {subject, action, context}

This task is DIFFERENT from router classification (A1):
- Router: classifies into 4 intents (order/consultant/faq/ignore)
- C2.1: extracts structured components from a query

Output format: JSONL with chat template format
Each sample has: query text + structured label JSON
"""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "intent_extraction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = OUT_DIR / "train.jsonl"
TEST_OUT = OUT_DIR / "test.jsonl"
TRAIN_COUNT = 3000
TEST_COUNT = 800
SEED = 42

# ── Subject patterns ────────────────────────────────────────────────────────
SUBJECT_MAP = {
    "anh": ["cho anh", "a ơi", "anh ", "a "],
    "em": ["cho em", "em ", "e "],
    "tôi": ["tôi ", "tui "],
    "mình": ["mình ", "bọn mình "],
}

# ── Menu & context items ────────────────────────────────────────────────────
COFFEE_ITEMS = [
    "Phin Sữa Đá", "Phin Đen Đá", "Bạc Xỉu", "Latte", "Americano",
    "Cappuccino", "Espresso", "Cold Brew", "Mocha", "Caramel Macchiato",
    "Cà Phê Muối", "Cà Phê Dừa", "Vietnamese Latte", "Hazelnut Latte",
    "Vanilla Latte", "Flat White", "Iced Coffee",
]
TEA_ITEMS = [
    "Trà Sen Vàng", "Trà Đào Cam Sả", "Trà Vải", "Trà Xoài Nhiệt Đới",
    "Trà Chanh Mật Ong", "Trà Oolong Sữa", "Trà Nhài", "Trà Dâu",
    "Trà Cam Quế", "Trà Táo Bạc Hà", "Matcha Latte", "Hồng Trà Sữa",
    "Trà Sữa Trân Châu", "Trà Lài Mật Ong",
]
FREEZE_ITEMS = [
    "Freeze Trà Xanh", "Freeze Chocolate", "Freeze Caramel", "Freeze Cookies",
    "Freeze Việt Quất", "Freeze Dâu", "Freeze Matcha Đậu Đỏ",
    "Freeze Bạc Xỉu", "Freeze Xoài", "Freeze Socola Bạc Hà",
]
FOOD_ITEMS = [
    "Bánh Mì Que", "Croissant", "Tiramisu", "Bánh Chuối", "Bánh Phô Mai",
    "Muffin Chocolate", "Sandwich Gà", "Bánh Táo", "Brownie",
]
ALL_ITEMS = COFFEE_ITEMS + TEA_ITEMS + FREEZE_ITEMS + FOOD_ITEMS
SIZES = ["S", "M", "L"]
QUANTITIES = ["1", "2", "3", "một", "hai", "ba"]

# ── Action taxonomy for C2.1 (coarser than intent) ─────────────────────────
# Actions are used as CACHE KEYS — keep them short and semantic
# "đặt bàn" appears in both order and faq — add order prefix for clarity
ACTION_TAXONOMY = {
    "order": [
        ("đặt món", ["cho anh", "cho em", "cho tôi", "mình", "lấy", "gọi", "order", "mua", "thêm", "ship", "mang ra"]),
        ("tính tiền", ["tính tiền", "thanh toán", "bao nhiêu tiền", "giá bao nhiêu", "hết bao nhiêu"]),
        ("đặt bàn_order", ["đặt bàn", "book bàn", "giữ bàn", "reserve", "đặt chỗ"]),
        ("hủy đơn", ["hủy", "bỏ", "xóa đơn", "bỏ món"]),
        ("sửa đơn", ["sửa", "đổi", "thay đổi", "cập nhật đơn"]),
        ("xem đơn", ["xem đơn", "đơn hàng", "order hiện tại", "đã gọi", "what did I order"]),
    ],
    "consultant": [
        ("tư vấn theo khẩu vị", ["ngon", "ít ngọt", "không ngọt", "bớt ngọt", "đắng", "nhạt", "ngọt"]),
        ("tư vấn không caffeine", ["không caffeine", "không cà phê", "decaf", "ít caffeine", "caffeine"]),
        ("tư vấn theo thời tiết", ["nóng", "mát", "lạnh", "mùa hè", "buổi sáng", "buổi chiều"]),
        ("tư vấn theo giá", ["dưới", "rẻ", "mắc", "giá", "ngân sách", "budget", "rẻ nhất"]),
        ("tư vấn theo sở thích", ["thích", "muốn thử", "recommend", "suggest", "gợi ý", "nên uống"]),
        ("hỏi món bán chạy", ["bán chạy", "best seller", "hot", "phổ biến", "nhiều người gọi"]),
    ],
    "faq": [
        ("hỏi giờ mở cửa", ["mấy giờ mở", "giờ mở cửa", "open", "bắt đầu mở", "what time open"]),
        ("hỏi giờ đóng cửa", ["mấy giờ đóng", "giờ đóng cửa", "close", "đóng cửa", "what time close"]),
        ("hỏi wifi", ["wifi", "password", "mật khẩu", "pass wifi", "internet"]),
        ("hỏi thanh toán", ["thanh toán", "thẻ", "qr", "tiền mặt", "visa", "cash", "pay", "credit"]),
        ("hỏi giao hàng", ["giao hàng", "delivery", "ship", "giao tận", "deliver"]),
        ("hỏi chỗ ngồi", ["chỗ ngồi", "ổ cắm", "đậu xe", "parking", "seating", "làm việc"]),
        ("hỏi đặt bàn_faq", ["đặt bàn", "reservation", "book", "đặt chỗ"]),
        ("hỏi hóa đơn", ["hóa đơn", "xuất hóa", "VAT", "invoice", "biên lai"]),
        ("hỏi sinh nhật", ["sinh nhật", "birthday", "giảm sinh", "ưu đãi sinh"]),
        ("hỏi hoàn tiền", ["hoàn tiền", "refund", "đổi tiền"]),
        ("hỏi thành viên", ["thành viên", "membership", "thẻ thành viên", "tích điểm"]),
        ("hỏi khuyến mãi", ["khuyến mãi", "giảm giá", "sale", "voucher", "coupon", "discount"]),
        ("hỏi đồ ăn", ["đồ ăn", "bánh", "món ăn", "snack", "food", "có bánh gì"]),
        ("hỏi món chay", ["chay", "vegan", "vegetarian", "thuần chay"]),
    ],
    "ignore": [
        ("chào hỏi", ["hello", "hi", "chào", "hey", "alo", "haha", "ok", "okay"]),
        ("tạm biệt", ["bye", "tạm biệt", "chào nhé", "tks", "thanks", "cảm ơn"]),
        ("không rõ", ["ừm", "ờ", "hmm", "uhhh", "abc", "xyz"]),
    ],
}

# ── Context modifiers per action (keys must match action names above) ──────────
CONTEXT_TEMPLATES = {
    "đặt món": [
        "{qty} ly {item} size {size}",
        "{qty} cốc {item}",
        "{item} size {size}",
        "{qty} {item}",
        "thêm {item}",
    ],
    "tính tiền": ["tổng cộng bao nhiêu"],
    "đặt bàn_order": ["{qty} người", "vào {time}", "ngày {date}"],
    "hủy đơn": ["đơn hiện tại", "toàn bộ đơn"],
    "sửa đơn": ["sửa thành {item}", "đổi thành {item}"],
    "xem đơn": [],
    "tư vấn theo khẩu vị": ["ít ngọt", "không ngọt", "bớt ngọt", "vị nhẹ"],
    "tư vấn không caffeine": ["không cà phê", "không caffeine", "decaf"],
    "tư vấn theo thời tiết": ["ngày nóng", "trời nóng", "buổi sáng"],
    "tư vấn theo giá": ["dưới {price}k", "giá {price}k"],
    "tư vấn theo sở thích": ["thích cà phê", "thích trà", "thử món mới"],
    "hỏi món bán chạy": [],
    "hỏi giờ mở cửa": [],
    "hỏi giờ đóng cửa": [],
    "hỏi wifi": [],
    "hỏi thanh toán": [],
    "hỏi giao hàng": [],
    "hỏi chỗ ngồi": [],
    "hỏi đặt bàn_faq": ["{qty} người"],
    "hỏi hóa đơn": [],
    "hỏi sinh nhật": [],
    "hỏi hoàn tiền": [],
    "hỏi thành viên": [],
    "hỏi khuyến mãi": [],
    "hỏi đồ ăn": [],
    "hỏi món chay": [],
    "chào hỏi": [],
    "tạm biệt": [],
    "không rõ": [],
}

TIMES = ["7h", "8h", "9h", "10h", "11h", "12h", "13h", "14h", "15h", "17h", "18h", "19h", "20h"]
DATES = ["mai", "ngày mai", "chiều nay", "tối nay", "sáng nay", "chủ nhật", "thứ 7"]
PRICES = ["30", "35", "40", "45", "50", "55"]


def _detect_subject(text: str) -> Optional[str]:
    """Detect subject from query text."""
    text_lower = text.lower()
    for subj, patterns in SUBJECT_MAP.items():
        for pat in patterns:
            if pat in text_lower:
                return subj
    return None


def _build_query_and_label(intent: str, action: str, action_phrase: str) -> Dict:
    """
    Build a query + label pair for a given (intent, action).

    The action_phrase is the trigger word used in the query.
    """
    # ── Determine subject & subject pronouns ───────────────────────────────
    subject = random.choice(["anh", "em", "tôi", "mình", None])

    # ── Build query text ───────────────────────────────────────────────────
    prefixes = {
        "anh": ["Cho anh", "Anh", "A ơi cho anh", "Anh ơi"],
        "em": ["Cho em", "Em", "Em muốn", "Em ơi cho em"],
        "tôi": ["Cho tôi", "Tôi muốn", "Tôi cần"],
        "mình": ["Mình", "Bọn mình", "Mình muốn"],
        None: ["", "Làm ơn", "Xin"],
    }

    qty = random.choice(QUANTITIES)
    size = random.choice(SIZES)
    item = random.choice(ALL_ITEMS)
    time_val = random.choice(TIMES)
    date_val = random.choice(DATES)
    price = random.choice(PRICES)

    # Strip suffix for template lookup (đặt bàn_order vs đặt bàn_faq)
    action_key = action.replace("_order", "").replace("_faq", "")
    ctx_template = random.choice(CONTEXT_TEMPLATES.get(action_key, []) + [""])
    if ctx_template:
        ctx_filled = ctx_template.format(qty=qty, item=item, size=size, time=time_val, date=date_val, price=price)
    else:
        ctx_filled = ""

    # ── Build query text from action category ───────────────────────────────
    # Strip suffix for action matching
    action_key = action.replace("_order", "").replace("_faq", "")

    if action_key == "đặt món":
        templates = [
            f"{random.choice(prefixes[subject])} {qty} ly {item} size {size}",
            f"{random.choice(prefixes[subject])} {qty} {item} {size}",
            f"{random.choice(prefixes[subject])} order {qty} {item}",
            f"{random.choice(prefixes[subject])} lấy {qty} {item}",
            f"{random.choice(prefixes[subject])} gọi {qty} cốc {item} size {size}",
            f"{random.choice(prefixes[subject])} thêm {qty} {item} size {size}",
            f"ship {qty} {item} {size}",
            f"get me {qty} {item}",
            f"I'd like {qty} {item}",
            f"can I have {qty} {item} {size}",
        ]
        query = random.choice(templates)
        # Derive context from query
        context_parts = []
        qty_m = re.search(rf"({qty}|one|two|three|{qty})", query, re.I)
        if qty_m:
            context_parts.append(qty_m.group(0))
        if re.search(r"(size [SML])", query, re.I):
            sm = re.search(r"size ([SML])", query, re.I)
            context_parts.append(f"size: {sm.group(1).upper()}")
        if item in query:
            context_parts.append(item)
        # Extract time if present
        if re.search(r"\d+h|\d{1,2}:\d{2}", query):
            tm = re.search(r"(\d+h|\d{1,2}:\d{2})", query)
            context_parts.append(f"thời gian: {tm.group(0)}")

    elif action_key == "tính tiền":
        templates = [
            f"{random.choice(prefixes[subject])} tính tiền",
            f"{random.choice(prefixes[subject])} bao nhiêu tiền",
            f"Tổng cộng {random.choice(['bao nhiêu', 'hết bao nhiêu'])}",
            f"How much is it",
            f"What's the total",
        ]
        query = random.choice(templates)
        context_parts = []

    elif action_key == "đặt bàn":
        templates = [
            f"{random.choice(prefixes[subject])} đặt bàn {qty} người {date_val} lúc {time_val}",
            f"{random.choice(prefixes[subject])} book bàn {qty} người",
            f"{random.choice(prefixes[subject])} đặt chỗ {date_val}",
            f"Reserve a table for {qty}",
            f"I'd like to book a table for {qty} people",
        ]
        query = random.choice(templates)
        context_parts = []
        qty_m = re.search(r"(\d+|one|two|three|four)", query, re.I)
        if qty_m:
            context_parts.append(f"{qty_m.group(0)} người")
        if date_val in query:
            context_parts.append(f"ngày: {date_val}")
        if time_val in query:
            context_parts.append(f"giờ: {time_val}")

    elif action_key == "tư vấn theo khẩu vị":
        templates = [
            f"{random.choice(prefixes[subject])} có gì ngon ít ngọt",
            f"{random.choice(prefixes[subject])} gợi ý món không ngọt",
            f"recommend something not too sweet",
            f"what's good that's not sweet",
            f"{random.choice(prefixes[subject])} món nào ít ngọt",
            f"{random.choice(prefixes[subject])} thích vị nhẹ",
            f"suggest a light drink",
        ]
        query = random.choice(templates)
        context_parts = []
        if re.search(r"(ít ngọt|không ngọt|bớt ngọt)", query):
            context_parts.append("ít ngọt")
        if re.search(r"(nhẹ|vị nhẹ)", query):
            context_parts.append("vị nhẹ")

    elif action_key == "tư vấn không caffeine":
        templates = [
            f"{random.choice(prefixes[subject])} không uống caffeine",
            f"{random.choice(prefixes[subject])} có gì không cà phê",
            f"recommend something without caffeine",
            f"what drinks don't have coffee",
            f"{random.choice(prefixes[subject])} món nào không caffeine",
        ]
        query = random.choice(templates)
        context_parts = ["không caffeine"]

    elif action_key == "tư vấn theo thời tiết":
        templates = [
            f"{random.choice(prefixes[subject])} ngày nóng nên uống gì",
            f"{random.choice(prefixes[subject])} gợi ý món mát",
            f"it's hot today, what should I drink",
            f"{random.choice(prefixes[subject])} buổi sáng nên uống gì",
            f"{random.choice(prefixes[subject])} tư vấn đồ uống mùa hè",
        ]
        query = random.choice(templates)
        context_parts = []
        if re.search(r"(nóng|mát|mùa hè)", query):
            context_parts.append("thời tiết: nóng")
        if re.search(r"(sáng)", query):
            context_parts.append("buổi: sáng")

    elif action_key == "tư vấn theo giá":
        templates = [
            f"{random.choice(prefixes[subject])} gợi ý món dưới {price}k",
            f"{random.choice(prefixes[subject])} có món giá rẻ không",
            f"recommend something under {price}k",
            f"what's cheap here",
            f"{random.choice(prefixes[subject])} budget {price}k",
        ]
        query = random.choice(templates)
        context_parts = [f"giá: dưới {price}k"]

    elif action_key == "tư vấn theo sở thích":
        templates = [
            f"{random.choice(prefixes[subject])} gợi ý cho {random.choice(['anh', 'em', 'tôi'])}",
            f"{random.choice(prefixes[subject])} recommend something",
            f"what do you recommend",
            f"what's good here",
            f"{random.choice(prefixes[subject])} nên uống gì",
            f"can you suggest a drink",
        ]
        query = random.choice(templates)
        context_parts = []

    elif action_key == "hỏi món bán chạy":
        templates = [
            f"{random.choice(prefixes[subject])} món nào bán chạy",
            f"What's your best seller",
            f"Món phổ biến nhất là gì",
            f"what's most popular",
            f"{random.choice(prefixes[subject])} có gì ngon nhất",
        ]
        query = random.choice(templates)
        context_parts = []

    elif "hỏi giờ" in action:
        templates = [
            f"{random.choice(prefixes[subject])} mấy giờ mở cửa",
            f"{random.choice(prefixes[subject])} what time do you open",
            f"Quán mở cửa mấy giờ",
            f"When do you open",
            f"{random.choice(prefixes[subject])} mấy giờ đóng cửa",
            f"What time do you close",
        ]
        query = random.choice(templates)
        context_parts = []

    elif action_key == "hỏi wifi":
        templates = [
            f"{random.choice(prefixes[subject])} cho xin wifi",
            f"Wifi tên gì",
            f"What's the wifi name",
            f"{random.choice(prefixes[subject])} password wifi là gì",
            f"Wifi password",
        ]
        query = random.choice(templates)
        context_parts = ["wifi"]

    elif action_key == "hỏi thanh toán":
        templates = [
            f"{random.choice(prefixes[subject])} có thanh toán thẻ không",
            f"Can I pay by card",
            f"{random.choice(prefixes[subject])} chấp nhận thẻ gì",
            f"Do you accept credit cards",
            f"{random.choice(prefixes[subject])} thanh toán QR được không",
            f"Cash or card",
        ]
        query = random.choice(templates)
        context_parts = []
        if re.search(r"(thẻ|qr|card|visa)", query, re.I):
            context_parts.append("thanh toán: thẻ/QR")

    elif action_key == "hỏi giao hàng":
        templates = [
            f"{random.choice(prefixes[subject])} có giao hàng không",
            f"Do you deliver",
            f"{random.choice(prefixes[subject])} ship đơn được không",
            f"Can you deliver to my place",
            f"{random.choice(prefixes[subject])} giao tận nơi không",
        ]
        query = random.choice(templates)
        context_parts = ["giao hàng"]

    elif action_key == "hỏi chỗ ngồi":
        templates = [
            f"{random.choice(prefixes[subject])} có chỗ ngồi làm việc không",
            f"Is there seating available",
            f"{random.choice(prefixes[subject])} có ổ cắm điện không",
            f"Do you have parking",
            f"{random.choice(prefixes[subject])} chỗ đậu xe có không",
        ]
        query = random.choice(templates)
        context_parts = []
        if re.search(r"(ngồi|làm việc|seating)", query):
            context_parts.append("chỗ ngồi")
        if re.search(r"(đậu xe|parking)", query):
            context_parts.append("đậu xe")

    elif action_key == "hỏi hóa đơn":
        templates = [
            f"{random.choice(prefixes[subject])} có xuất hóa đơn không",
            f"Can I get a receipt",
            f"{random.choice(prefixes[subject])} xuất VAT được không",
            f"Do you issue invoices",
        ]
        query = random.choice(templates)
        context_parts = ["hóa đơn"]

    elif action_key == "hỏi sinh nhật":
        templates = [
            f"{random.choice(prefixes[subject])} có hỗ trợ sinh nhật không",
            f"Do you have birthday discounts",
            f"{random.choice(prefixes[subject])} ưu đãi sinh nhật có không",
            f"Birthday special",
        ]
        query = random.choice(templates)
        context_parts = ["sinh nhật"]

    elif action_key == "hỏi hoàn tiền":
        templates = [
            f"{random.choice(prefixes[subject])} có hoàn tiền không",
            f"Can I get a refund",
            f"{random.choice(prefixes[subject])} không hài lòng thì sao",
        ]
        query = random.choice(templates)
        context_parts = ["hoàn tiền"]

    elif action_key == "hỏi thành viên":
        templates = [
            f"{random.choice(prefixes[subject])} có thẻ thành viên không",
            f"Do you have a membership card",
            f"{random.choice(prefixes[subject])} tích điểm được không",
            f"Can I earn points",
        ]
        query = random.choice(templates)
        context_parts = ["thành viên"]

    elif action_key == "hỏi khuyến mãi":
        templates = [
            f"{random.choice(prefixes[subject])} có khuyến mãi gì hôm nay",
            f"Are there any promotions today",
            f"{random.choice(prefixes[subject])} có giảm giá không",
            f"any discount today",
        ]
        query = random.choice(templates)
        context_parts = ["khuyến mãi"]

    elif action_key == "hỏi đồ ăn":
        templates = [
            f"{random.choice(prefixes[subject])} có bánh gì ăn kèm",
            f"What snacks do you have",
            f"{random.choice(prefixes[subject])} có món ăn nhẹ không",
            f"Do you have any food",
        ]
        query = random.choice(templates)
        context_parts = ["đồ ăn nhẹ"]

    elif action_key == "hỏi món chay":
        templates = [
            f"{random.choice(prefixes[subject])} có món chay không",
            f"Do you have vegetarian options",
            f"{random.choice(prefixes[subject])} đồ chay có không",
            f"Vegan drinks available",
        ]
        query = random.choice(templates)
        context_parts = ["món chay/vegan"]

    elif action_key == "đặt bàn":
        templates = [
            f"{random.choice(prefixes[subject])} đặt bàn {qty} người",
            f"I'd like to reserve a table for {qty}",
            f"Book a table for {qty} people",
        ]
        query = random.choice(templates)
        context_parts = [f"{qty} người"]

    elif action_key == "hỏi đặt bàn":
        templates = [
            f"{random.choice(prefixes[subject])} có đặt bàn trước không",
            f"Can I make a reservation",
            f"{random.choice(prefixes[subject])} đặt bàn được không",
            f"Do you take reservations",
        ]
        query = random.choice(templates)
        context_parts = ["đặt bàn"]

    elif action_key == "chào hỏi":
        templates = [
            "hello", "hi", "chào", "hey", "alo", "haha", "ok", "okay",
            "good morning", "good afternoon", "yo", "what's up",
        ]
        query = random.choice(templates)
        context_parts = []

    elif action_key == "tạm biệt":
        templates = [
            "bye", "tạm biệt", "cảm ơn", "thanks", "thank you", "cảm ơn nhé",
            "tks", "cheers", "chào nhé",
        ]
        query = random.choice(templates)
        context_parts = []

    else:  # không rõ
        templates = [
            "ừm", "ờ", "hmm", "uhhh", "...", "abc", "xyz", "hjc",
            "hichic", "hjx", "huhu",
        ]
        query = random.choice(templates)
        context_parts = []

    # Clean up query
    query = re.sub(r"\s+", " ", query.strip())

    # ── Build label ─────────────────────────────────────────────────────────
    detected_subject = _detect_subject(query) or subject
    context_str = ", ".join(context_parts) if context_parts else None
    # Canonical action (strip _order, _faq suffix for cache key)
    canonical_action = action_key

    label = {
        "subject": detected_subject,
        "action": canonical_action,
        "context": context_str,
    }

    return {"query": query, "label": label, "intent": intent}


def _build_samples_for_action(intent: str, action: str, n: int) -> List[Dict]:
    """Generate n samples for a specific (intent, action) pair."""
    samples = []
    for _ in range(n):
        sample = _build_query_and_label(intent, action, action)
        samples.append(sample)
    return samples


def generate_intent_extraction_data(
    train_out: Path,
    test_out: Path,
    train_target: int = TRAIN_COUNT,
    test_target: int = TEST_COUNT,
    seed: int = SEED,
) -> None:
    """Generate train + test JSONL for C2.1 Intent Extraction."""
    random.seed(seed)

    # Build per-action sample counts (balanced-ish)
    all_actions: List[tuple] = []
    for intent, actions in ACTION_TAXONOMY.items():
        for action, _ in actions:
            all_actions.append((intent, action))

    # Distribute samples evenly
    samples_per_action = train_target // len(all_actions)
    remainder = train_target % len(all_actions)

    train_samples: List[Dict] = []
    for idx, (intent, action) in enumerate(all_actions):
        n = samples_per_action + (1 if idx < remainder else 0)
        train_samples.extend(_build_samples_for_action(intent, action, n))

    # Shuffle
    random.shuffle(train_samples)

    # Split: 80% train, 20% test from generated data
    test_samples = train_samples[-test_target:]
    train_final = train_samples[:-test_target]

    print(f"Train: {len(train_final)}, Test: {len(test_samples)}")
    print(f"Action distribution (train):")
    action_counts = {}
    for s in train_final:
        a = s["label"]["action"]
        action_counts[a] = action_counts.get(a, 0) + 1
    for a, c in sorted(action_counts.items()):
        print(f"  {a}: {c}")

    # ── Write JSONL with chat template ─────────────────────────────────────
    SYSTEM_PROMPT = (
        'Bạn là intent extractor cho ứng dụng F&B. '
        'Trả về JSON với 3 trường: "subject" (chủ ngữ), '
        '"action" (hành động chính, dùng làm cache key), '
        '"context" (thông tin bổ sung). '
        'Trả lời JSON only.'
    )

    def write_jsonl(path: Path, samples: List[Dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                query = sample["query"]
                label = sample["label"]
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
                ]
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    write_jsonl(train_out, train_final)
    write_jsonl(test_out, test_samples)
    print(f"\nSaved: {train_out} ({len(train_final)} train), {test_out} ({len(test_samples)} test)")

    # Summary stats
    print("\n=== Intent Extraction Data Summary ===")
    for intent in ACTION_TAXONOMY:
        count = sum(1 for s in train_final if s["intent"] == intent)
        print(f"  {intent}: {count} ({count/len(train_final)*100:.1f}%)")

    actions = set(s["label"]["action"] for s in train_final)
    print(f"  Total unique actions: {len(actions)}")


if __name__ == "__main__":
    generate_intent_extraction_data(TRAIN_OUT, TEST_OUT, TRAIN_COUNT, TEST_COUNT, SEED)
    print("\nDone! Data generated at:")
    print(f"  {TRAIN_OUT}")
    print(f"  {TEST_OUT}")
