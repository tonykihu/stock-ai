# This script sends email alerts using Gmail's SMTP server.
import smtplib
from email.mime.text import MIMEText
import getpass  # For secure password input

def send_email_alert(signal, ticker):
    """
    Sends email using Gmail SMTP.
    Requires app password: https://myaccount.google.com/apppasswords
    """
    sender = "your.email@gmail.com"
    receiver = "recipient@example.com"
    
    # Secure password handling
    password = getpass.getpass("Enter Gmail app password: ")
    
    msg = MIMEText(f"{ticker} alert: {signal}")
    msg['Subject'] = f"Stock Alert: {ticker}"
    msg['From'] = sender
    msg['To'] = receiver

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print("Alert email sent!")
    except Exception as e:
        print(f"Email failed: {str(e)}")


# This script sends alerts to a Discord channel using a webhook.
import requests
import json

def send_discord_alert(signal, ticker, webhook_url):
    """
    Sends alert to Discord channel via webhook.
    """
    payload = {
        "content": f"**{ticker} Alert**\nSignal: {signal}",
        "username": "Stock Bot"
    }
    
    try:
        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(f"Discord alert failed: {err}")


# utils/alerts.py (updated)
def generate_alert(signal, ticker, historical_context=True):
    """
    Adds historical context to alerts (e.g., "SCOM has risen 8% in the past month").
    """
    data = pd.read_csv("data/processed/kenya_processed.csv")
    latest = data.iloc[-1]
    month_ago = data.iloc[-30]
    percent_change = ((latest["Price"] - month_ago["Price"]) / month_ago["Price"]) * 100
    
    message = f"{ticker}: {signal}\n"
    if historical_context:
        message += f"Historical Context: {percent_change:.1f}% over past 30 days"
    
    # Send via email/Discord (use functions from earlier)
    send_email_alert(message, ticker)
    # OR
    send_discord_alert(message, ticker, WEBHOOK_URL)