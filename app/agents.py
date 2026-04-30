"""
Multi-agent system:
- OrderAgent
- ConsultantAgent
- FAQAgent
- IgnoreAgent

Each agent handles specific domain logic.
"""


class OrderAgent:
    def answer(self, query):
        return "Đã nhận order"


class ConsultantAgent:
    def answer(self, query):
        return "Gợi ý món"


class FAQAgent:
    def answer(self, query):
        return "Thông tin quán"


class IgnoreAgent:
    def answer(self, query):
        return "Bạn cần gì ạ?"
