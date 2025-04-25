import smtplib
from email.mime.text import MIMEText

def send_alert(signal, ticker):
    sender = "your_email@gmail.com"
    receiver = "alerts@example.com"
    password = "your_app_password"  # Use Gmail App Password

    msg = MIMEText(f"{ticker}: {signal}")
    msg["Subject"] = "Stock Alert"
    msg["From"] = sender
    msg["To"] = receiver

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())

# Trigger in dashboard
if st.button("Send Test Alert"):
    send_alert("BUY", "AAPL")