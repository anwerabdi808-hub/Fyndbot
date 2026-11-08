"""
Fyndbot v2: hittar billiga begagnade annonser, jämför mot marknaden,
låter Claude bedöma skick utifrån bilden och skickar notis till Telegram.

Installera:  pip install requests beautifulsoup4 zeep anthropic
Kör:         python fyndbot.py
"""
import base64
import json
import os
import re
import sqlite3
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ----------------------------- INSTÄLLNINGAR -----------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TRADERA_APP_ID = os.getenv("TRADERA_APP_ID", "")
TRADERA_APP_KEY = os.getenv("TRADERA_APP_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")  # för bildanalys

SEARCHES = [
    {"q": "xbox one", "max_price": 600, "avoid": ["defekt", "trasig", "reservdelar", "fungerar ej"]},
    {"q": "xbox series s", "max_price": 1800, "avoid": ["defekt", "trasig", "reservdelar"]},
    {"q": "ps5 kontroll", "max_price": 300, "avoid": ["defekt", "trasig", "drift"]},
]

MIN_PROFIT = 0
SHIPPING_COST = 70
RESALE_FACTOR = 0.85
MIN_COMPARABLES = 3
INTERVAL_SECONDS = 600

IMAGE_MODEL = "claude-haiku-5-5"  # billig modell räcker för bildkoll
MIN_CONDITION = 3                 # 1-5, annonser under detta sorteras bort
# -------------------------------------------------------------------------

HEADERS = {"User-Agent": "Mozilla/5.0 (privat fyndbot)"}


@dataclass
class Listing:
    source: str
    title: str
    price: float
    url: str
    image_url: Optional[str] = None


# ------------------------------- KÄLLOR ----------------------------------
def fetch_tradera(query):
    """Traderas officiella API. Kräver gratis app-nyckel."""
    if not TRADERA_APP_ID:
        return []
    from zeep import Client

    client = Client("https://api.tradera.com/v3/searchservice.asmx?WSDL")
    headers = {
        "AuthenticationHeader": {"AppId": TRADERA_APP_ID, "AppKey": TRADERA_APP_KEY},
        "ConfigurationHeader": {"Sandbox": 0, "MaxResultAge": 60},
    }
    res = client.service.Search(
        query=query, categoryId=0, pageNumber=1,
        orderBy="Relevance", _soapheaders=headers,
    )
    out = []
    for it in (res.Items or []):
        price = getattr(it, "BuyItNowPrice", None)
        if price:
            out.append(Listing("Tradera", it.ShortDescription, float(price),
                               it.ItemUrl, getattr(it, "ThumbnailLink", None)))
    return out


def fetch_blocket(query):
    """HTML-scraping. Justera om Blocket ändrar sidan."""
    r = requests.get("https://www.blocket.se/annonser/hela_sverige",
                     params={"q": query, "sort": "rel"}, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        print(f"Blocket svarade {r.status_code}")
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    for a in soup.select("a[href*='/annons/']"):
        text = a.get_text(" ", strip=True)
        m = re.search(r"(\d[\d\s]*)\s*kr", text)
        if not m:
            continue
        url = a["href"]
        if url.startswith("/"):
            url = "https://www.blocket.se" + url
        img = a.select_one("img")
        out.append(Listing("Blocket", text[:80], float(re.sub(r"\s", "", m.group(1))),
                           url, img.get("src") if img else None))
    return out


def fetch_vinted(query):
    """Vinteds inofficiella API. Bryter troligen mot villkoren och kan blockeras."""
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get("https://www.vinted.se", timeout=20)  # hämtar sessionskakor
    r = s.get("https://www.vinted.se/api/v2/catalog/items",
              params={"search_text": query, "order": "newest_first", "per_page": 30},
              timeout=20)
    if r.status_code != 200:
        print(f"Vinted svarade {r.status_code} (blockerad?)")
        return []
    out = []
    for it in r.json().get("items", []):
        p = it.get("price")
        amount = p.get("amount") if isinstance(p, dict) else p
        if amount is None:
            continue
        out.append(Listing("Vinted", it.get("title", ""), float(amount),
                           it.get("url", ""), (it.get("photo") or {}).get("url")))
    return out


# Facebook Marketplace saknas med flit: kräver inloggning, har inget öppet API
# och scraping ger risk för avstängt konto. Använd Facebooks egna sparade
# sökningar med notiser för den.
# Fler sajter: kopiera fetch_blocket och anpassa, returnera en lista Listing.
SOURCES = [fetch_tradera, fetch_blocket, fetch_vinted]


# --------------------------- BILDANALYS (AI) -----------------------------
IMAGE_PROMPT = """Du bedömer en begagnad vara utifrån en annonsbild.
Annonsens titel: "{title}"
Svara ENDAST med JSON, utan annan text:
{{"skick": <1-5, 5 = som ny>,
 "visar_produkten": <true/false, visar bilden verkligen varan?>,
 "synliga_skador": [<kort lista, tom om inga>],
 "misstankt_replika_eller_fel_produkt": <true/false>,
 "kommentar": "<en mening på svenska>"}}
Var försiktig: en bild kan inte bevisa att något fungerar, bara visa tecken
på slitage, skador, saknade delar eller dålig kvalitet."""


def check_condition(listing):
    """Returnerar dict med bedömning, eller None om bildkoll inte är möjlig."""
    if not (ANTHROPIC_API_KEY and listing.image_url):
        return None
    try:
        import anthropic

        img = requests.get(listing.image_url, headers=HEADERS, timeout=20)
        media_type = img.headers.get("content-type", "image/jpeg").split(";")[0]
        data = base64.standard_b64encode(img.content).decode()
        msg = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY).messages.create(
            model=IMAGE_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": IMAGE_PROMPT.format(title=listing.title)},
            ]}],
        )
        text = msg.content[0].text
        return json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception as e:
        print(f"Bildkoll misslyckades: {e}")
        return None


# ------------------------------ LOGIK ------------------------------------
def find_deals(search):
    listings = []
    for fetch in SOURCES:
        try:
            listings += fetch(search["q"])
        except Exception as e:
            print(f"{fetch.__name__} misslyckades: {e}")
        time.sleep(2)

    clean = [l for l in listings if l.price >= 50
             and not any(w in l.title.lower() for w in search["avoid"])]
    if len(clean) < MIN_COMPARABLES:
        print(f"'{search['q']}': för få annonser ({len(clean)}) för jämförelse")
        return []
    market = statistics.median(l.price for l in clean)
    resale = market * RESALE_FACTOR

    deals = []
    for l in clean:
        if l.price > search["max_price"]:
            continue
        profit = resale - l.price - SHIPPING_COST
        if profit >= MIN_PROFIT:
            deals.append((l, market, profit))
    return sorted(deals, key=lambda d: -d[2])


def already_seen(db, url):
    if db.execute("SELECT 1 FROM seen WHERE url=?", (url,)).fetchone():
        return True
    db.execute("INSERT INTO seen VALUES (?)", (url,))
    db.commit()
    return False


def notify(text):
    print(text)
    if TELEGRAM_TOKEN:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=20)


def main():
    db = sqlite3.connect("seen.db")
    db.execute("CREATE TABLE IF NOT EXISTS seen (url TEXT PRIMARY KEY)")
    while True:
        for s in SEARCHES:
            for l, market, profit in find_deals(s):
                if already_seen(db, l.url):
                    continue
                # Bildkollen körs bara på annonser som redan klarat pris och vinst
                check = check_condition(l)
                if check:
                    if (not check.get("visar_produkten", True)
                            or check.get("misstankt_replika_eller_fel_produkt")
                            or check.get("skick", 0) < MIN_CONDITION):
                        print(f"Sorterade bort: {l.title} ({check.get('kommentar')})")
                        continue
                    skick = (f"Skick (AI, från bild): {check['skick']}/5. "
                             f"{check.get('kommentar', '')}")
                else:
                    skick = "Skick: ej bildkollat"
                notify(
                    f"FYND ({l.source}): {l.title}\n"
                    f"Pris: {l.price:.0f} kr | Marknad: {market:.0f} kr\n"
                    f"Beräknad vinst: ca {profit:.0f} kr\n{skick}\n{l.url}"
                )
        if os.getenv("RUN_ONCE"):  # GitHub Actions kör skriptet en gång per schema
            break
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
