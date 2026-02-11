import os
import smtplib
import json
from email.mime.text import MIMEText

import requests
import pandas as pd


def send_email_alert(signal, ticker):
    """
    Sends email using Gmail SMTP.
    Uses environment variables for credentials.
    Set GMAIL_USER, GMAIL_RECIPIENT, and GMAIL_APP_PASSWORD as env vars.
    Requires app password: https://myaccount.google.com/apppasswords
    """
    sender = os.environ.get("GMAIL_USER", "your.email@gmail.com")
    receiver = os.environ.get("GMAIL_RECIPIENT", "recipient@example.com")
    password = os.environ.get("GMAIL_APP_PASSWORD")

    if not password:
        print("GMAIL_APP_PASSWORD environment variable not set. Skipping email alert.")
        return

    msg = MIMEText(f"{ticker} alert: {signal}")
    msg["Subject"] = f"Stock Alert: {ticker}"
    msg["From"] = sender
    msg["To"] = receiver

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print(f"Alert email sent for {ticker}!")
    except Exception as e:
        print(f"Email failed: {e}")


def send_discord_alert(signal, ticker, webhook_url):
    """
    Sends alert to Discord channel via webhook.
    """
    if not webhook_url:
        print("No Discord webhook URL provided. Skipping Discord alert.")
        return

    payload = {
        "content": f"**{ticker} Alert**\nSignal: {signal}",
        "username": "Stock Bot"
    }

    try:
        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        print(f"Discord alert sent for {ticker}!")
    except requests.exceptions.RequestException as err:
        print(f"Discord alert failed: {err}")


def generate_alert(signal, ticker, historical_context=True, discord_webhook_url=None):
    """
    Generates and sends an alert with optional historical context.
    Example: "SCOM has risen 8% in the past month".
    """
    message = f"{ticker}: {signal}\n"

    if historical_context:
        try:
            data = pd.read_csv("data/processed/kenya_processed.csv")
            if len(data) >= 30:
                latest = data.iloc[-1]
                month_ago = data.iloc[-30]
                percent_change = ((latest["Price"] - month_ago["Price"]) / month_ago["Price"]) * 100
                message += f"Historical Context: {percent_change:.1f}% over past 30 days"
            else:
                message += "Historical Context: Not enough data (< 30 days)"
        except (FileNotFoundError, KeyError) as e:
            message += f"Historical Context: Unavailable ({e})"

    # Send via email
    send_email_alert(message, ticker)

    # Send via Discord if webhook URL is provided
    webhook_url = discord_webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
    if webhook_url:
        send_discord_alert(message, ticker, webhook_url)
