import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

class NHSScraper:
    BASE_URL = "https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    def get_condition_links(self):
        try:
            response = requests.get(self.BASE_URL, headers=self.HEADERS, timeout=15)
            soup = BeautifulSoup(response.content, "html.parser")
            links = set()
            
            for a in soup.find_all("a", href=True):
                href = a["href"]
                # بنفلتر الروابط عشان ناخد الأمراض بس
                if "/illnesses-and-conditions/" in href:
                    parts = [p for p in href.split("/") if p]
                    # بنتأكد إن الرابط بيشاور على مرض حقيقي مش مجرد صفحة قسم أو حرف أبجدي
                    if len(parts) >= 2 and "a-to-z" not in parts[-1]:
                        if href.startswith("/"):
                            full_url = "https://www.nhsinform.scot" + href
                        else:
                            full_url = href
                        links.add(full_url)
                        
            return list(links)
        except Exception as e:
            print(f"Error fetching links: {e}")
            return []

    def scrape_condition(self, url):
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=15)
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Get condition name
            h1 = soup.find("h1")
            if not h1:
                return None
            condition_name = h1.get_text(strip=True)
            
            # Improved extraction of symptoms and info
            def get_section_text(keywords):
                header = soup.find(lambda tag: tag.name in ['h2', 'h3'] and any(k in tag.get_text().lower() for k in keywords))
                text = ""
                if header:
                    for sibling in header.find_next_siblings():
                        if sibling.name in ['h2', 'h3', 'header', 'footer']: break
                        text += sibling.get_text(" ", strip=True) + " "
                return text.strip()

            symptoms_text = get_section_text(['symptom', 'sign'])
            causes_text = get_section_text(['cause', 'why it happens'])
            rec_text = get_section_text(['treatment', 'how to treat', 'recommendation', 'what to do'])

            # Fallback if specific sections not found
            if len(symptoms_text) < 50:
                paragraphs = soup.select(".article-content p, .main-content p")
                symptoms_text = " ".join([p.get_text(strip=True) for p in paragraphs[:3]])

            data = {
                "condition": condition_name,
                "url": url,
                "symptoms": symptoms_text if symptoms_text else "Check NHS website for detailed symptoms.",
                "causes": causes_text if causes_text else "Causes vary by individual case.",
                "warnings": "Seek medical help if symptoms persist or worsen.",
                "recommendations": rec_text if rec_text else "Follow professional medical advice."
            }
            return data
        except Exception as e:
            # صغرنا رسالة الخطأ عشان متزحمش الشاشة
            pass 
            return None

    def run(self, limit=None):
        print("Fetching condition links from NHS website...")
        links = self.get_condition_links()
        
        if not links:
            print("Failed to fetch real links! Using dummy data...")
            return [
                {"condition": "Common Cold", "symptoms": "Runny nose, sneezing, sore throat", "causes": "Viral infection", "warnings": "Seek help if breathing is difficult", "recommendations": "Rest and fluids"},
                {"condition": "Flu", "symptoms": "High fever, chills, muscle aches", "causes": "Influenza virus", "warnings": "Emergency if chest pain occurs", "recommendations": "Antiviral drugs and rest"}
            ]

        print(f"Successfully found {len(links)} actual disease links!")

        if limit:
            links = links[:limit]
            
        results = []
        for link in tqdm(links, desc="Scraping Conditions"):
            data = self.scrape_condition(link)
            if data:
                results.append(data)
            time.sleep(0.2) # قللنا الوقت لـ 0.2 عشان يخلص الـ 200 حالة أسرع
            
        return results