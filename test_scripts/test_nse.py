from selenium import webdriver
driver = webdriver.Chrome()
driver.get("https://live.mystocks.co.ke")
print(driver.page_source[:500])  # Inspect dynamic content
