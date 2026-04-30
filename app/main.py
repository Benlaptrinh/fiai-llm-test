"""
Main FastAPI application.

Flow:
User -> Router -> Agent -> Response
"""

from fastapi import FastAPI

from app.agents import ConsultantAgent, FAQAgent, IgnoreAgent, OrderAgent
from app.router_agent import classify_intent

app = FastAPI()

order = OrderAgent()
consultant = ConsultantAgent()
faq = FAQAgent()
ignore = IgnoreAgent()


@app.post("/chat")
def chat(query: dict):
    text = query["query"]

    intent = classify_intent(text)["action"]

    if intent == "order":
        ans = order.answer(text)
    elif intent == "consultant":
        ans = consultant.answer(text)
    elif intent == "faq":
        ans = faq.answer(text)
    else:
        ans = ignore.answer(text)

    return {"intent": intent, "answer": ans}
