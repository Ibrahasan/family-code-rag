from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from pathlib import Path


class DynamicWebScraper:
    def __init__(self, url):
        self.url = url

    @property
    def url(self):
        return self._url

    @url.setter
    def url(self, value):
        if not isinstance(value, str) or not value.startswith("http"):
            raise ValueError("The URL must start with 'http' or 'https'.")
        self._url = value

    def extract_and_save(self, file_path="aile_mecellesi.txt"):
        edge_options = EdgeOptions()
        edge_options.add_argument("--headless=new")
        edge_options.add_argument("--disable-gpu")
        edge_options.add_argument("--no-sandbox")
        edge_options.add_argument("--disable-dev-shm-usage")
        edge_options.add_argument("--window-size=1920,1080")
        edge_options.add_experimental_option("excludeSwitches", ["enable-logging"])
        
        print("Edge WebDriver is being installed....")
        service = EdgeService(EdgeChromiumDriverManager().install())
        driver = webdriver.Edge(service=service, options=edge_options)
        
        try:
            print(f"[{self.url}] page is loading...")
            driver.get(self.url)
            
            print("Loading the main text (the law)...")
            
            wait = WebDriverWait(driver, 20)
            
            target_element = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[class*='ZoomDocument_ZoomDocumentText']"))
            )
            
            text_content = target_element.text
            
            if not text_content:
                print("Warning: Element found, but it is empty.")
                return

            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_content)
                
            print(f"The cleaned text was successfully saved to the file '{file_path}'.")
            
        except Exception as e:
            print(f"An error occurred. The element was not found or the site did not load: {e}")
            
        finally:
            driver.quit()

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    RAW_DATA_DIR = BASE_DIR / "data" / "raw"
    
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    OUTPUT_FILE = RAW_DATA_DIR / "aile_mecellesi.txt"
    
    target_url = "https://e-qanun.az/framework/46946"
    scraper = DynamicWebScraper(target_url)
    
    scraper.extract_and_save(str(OUTPUT_FILE))