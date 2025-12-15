import requests
import os
from dotenv import load_dotenv
from langchain.agents import Tool

load_dotenv()
NTFY_URL = os.getenv("NTFY_URL")
PUSH_TITLE = "TWINMIND-AI ALERT"


def push(message: str):
    requests.post(
        NTFY_URL,
        data=message,
        headers={
            "Title": PUSH_TITLE,
            "Priority": "urgent"
        }
    )
    return {"status": "success", "sent_message": message}

def push_tool():
    push_tool = Tool(name="send_push_notification", func=push, description="Use this tool when you want to send a push notification")
    return push_tool

# push("Push notification service is active.")