import os
import smtplib
from email.mime.text import MIMEText

def send_alert(signal, ticker):
    """
    Sends an email alert for a stock signal.
    Uses environment variables for credentials (never hardcode passwords).
    Set GMAIL_USER and GMAIL_APP_PASSWORD as environment variables.
    """
    sender = os.environ.get("GMAIL_USER", "your_email@gmail.com")
    receiver = os.environ.get("ALERT_EMAIL", "alerts@example.com")
    password = os.environ.get("GMAIL_APP_PASSWORD")

    if not password:
        print("GMAIL_APP_PASSWORD environment variable not set. Skipping email alert.")
        return

    msg = MIMEText(f"{ticker}: {signal}")
    msg["Subject"] = f"Stock Alert: {ticker}"
    msg["From"] = sender
    msg["To"] = receiver

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print(f"Alert email sent for {ticker}: {signal}")
    except Exception as e:
        print(f"Email alert failed: {e}")
