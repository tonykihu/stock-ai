import requests
from bs4 import BeautifulSoup

url="https://live.mystocks.co.ke"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
print(soup.find_all("table")[0].prettify()[:500]) # Print first 500 chars of the table