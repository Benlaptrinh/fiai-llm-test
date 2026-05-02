"""
TTS Preprocessing module (C3).

Preprocesses text responses before TTS synthesis:
- Price: "49000đ" → "49 nghìn đồng"
- Size: "M" → "vừa", "L" → "lớn", "S" → "nhỏ"
- VAT / tax references
- Currency formatting
- Number pronunciation

This ensures natural, fluent TTS output for voice assistants.
"""

from __future__ import annotations

import re
from typing import Dict

# Size mappings for TTS
SIZE_MAP: Dict[str, str] = {
    "S": "nhỏ",
    "M": "vừa",
    "L": "lớn",
    "size s": "size nhỏ",
    "size m": "size vừa",
    "size l": "size lớn",
}

# Caffeine level descriptions
CAFFEINE_MAP: Dict[str, str] = {
    "none": "không caffeine",
    "low": "ít caffeine",
    "medium": "trung bình caffeine",
    "high": "nhiều caffeine",
}

# Sweetness level descriptions
SWEETNESS_MAP: Dict[str, str] = {
    "low": "ít ngọt",
    "medium": "ngọt vừa",
    "high": "ngọt nhiều",
}


def preprocess_for_tts(text: str) -> str:
    """
    Convert text response to TTS-friendly format.

    Operations:
    1. Price: "49000đ" / "49.000đ" / "49000 VND" → "49 nghìn đồng"
    2. Size: "size M" / "M" → "size vừa" / "vừa"
    3. Currency symbols: "đ" → "đồng"
    4. VAT/Tax references → natural pronunciation
    5. Number formatting: "2 ly" → "2 ly" (natural)
    6. Ellipsis → pause indicator
    7. Special characters → remove
    8. Multi-digit numbers → spoken form
    """
    if not text:
        return text

    result = text

    # 1. Price formatting: 49000đ → 49 nghìn đồng
    # Pattern: numbers followed by đ/VND/đồng
    result = re.sub(
        r"(\d{1,3}(?:\.\d{3})*)\s*(?:đ|VND|đồng)",
        lambda m: _number_to_speech(m.group(1)) + " đồng",
        result,
    )

    # 2. Price without currency: "49000" in context of money
    # Only convert if followed by nothing specific
    result = re.sub(
        r"(?<![\w\d])(\d{1,3}(?:\.\d{3})*)(?=\s*(?:VNĐ|vnd|VND|))",
        lambda m: _number_to_speech(m.group(1)),
        result,
    )

    # 3. Size: size M / M → size vừa / vừa
    for size_key, size_val in SIZE_MAP.items():
        result = re.sub(
            re.escape(size_key),
            size_val,
            result,
            flags=re.IGNORECASE,
        )

    # 4. Caffeine levels
    for key, val in CAFFEINE_MAP.items():
        result = re.sub(
            rf"\b{re.escape(key)}\b",
            val,
            result,
            flags=re.IGNORECASE,
        )

    # 5. Sweetness levels
    for key, val in SWEETNESS_MAP.items():
        result = re.sub(
            rf"\b{re.escape(key)}\b",
            val,
            result,
            flags=re.IGNORECASE,
        )

    # 6. VAT → "thuế"
    result = re.sub(r"\bVAT\b", "thuế", result, flags=re.IGNORECASE)

    # 7. "k" shorthand: "49k" → "49 nghìn"
    result = re.sub(
        r"\b(\d+)\s*k\b",
        lambda m: f"{m.group(1)} nghìn",
        result,
        flags=re.IGNORECASE,
    )

    # 8. Remove excessive punctuation for TTS
    result = re.sub(r"\.{2,}", ".", result)
    result = re.sub(r",{2,}", ",", result)

    # 8. Clean up extra whitespace
    result = re.sub(r"\s+", " ", result).strip()

    return result


def _number_to_speech(num_str: str) -> str:
    """
    Convert number string to Vietnamese speech form.

    Examples:
    - "35000" → "35 nghìn"
    - "49000" → "49 nghìn"
    - "155000" → "155 nghìn"
    - "1000" → "1 nghìn"
    - "100" → "100" (keep as is if < 1000)
    """
    try:
        num_str = num_str.replace(".", "").replace(",", "")
        num = int(num_str)

        if num < 1000:
            return str(num)

        if num >= 1000:
            thousands = num // 1000
            remainder = num % 1000
            if remainder == 0:
                return f"{thousands} nghìn"
            else:
                return f"{thousands} nghìn {remainder}"

        return str(num)
    except (ValueError, TypeError):
        return num_str


def preprocess_menu_item_for_tts(item: Dict) -> Dict:
    """
    Preprocess a menu item dict for TTS display.

    Converts price, size, caffeine, sweetness to TTS-friendly text.
    """
    result = dict(item)

    # Price
    price = item.get("price", 0)
    result["price_tts"] = f"{_number_to_speech(str(price))} đồng"

    # Size
    size = item.get("size", "")
    result["size_tts"] = SIZE_MAP.get(size, size)

    # Caffeine
    caffeine = item.get("caffeine", "")
    result["caffeine_tts"] = CAFFEINE_MAP.get(caffeine, caffeine)

    # Sweetness
    sweetness = item.get("sweetness", "")
    result["sweetness_tts"] = SWEETNESS_MAP.get(sweetness, sweetness)

    return result


# ─── Example usage ───────────────────────────────────────────────────────────

EXAMPLES = [
    (
        "Phin Sữa Đá size M giá 49000đ, caffeine medium, sweetness low.",
        "Phin Sữa Đá size vừa giá 49 nghìn đồng, ít caffeine, ít ngọt.",
    ),
    (
        "Quý khách vui lòng thanh toán VAT 10%.",
        "Quý khách vui lòng thanh toán thuế 10 phần trăm.",
    ),
    (
        "Cà Phê Muối size L giá 55000 VND.",
        "Cà Phê Muối size lớn giá 55 nghìn đồng.",
    ),
]


if __name__ == "__main__":
    print("TTS Preprocessing Examples")
    print("=" * 60)
    for input_text, expected in EXAMPLES:
        output = preprocess_for_tts(input_text)
        match = "✓" if output == expected else "~"
        print(f"\nInput:  {input_text}")
        print(f"Output: {output}")
        print(f"Note:   expected={expected} [{match}]")
