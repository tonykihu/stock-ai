import gspread
gc = gspread.service_account("credentials.json")
print("Authentication successful!" if gc else "Failed")