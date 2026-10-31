import requests
import csv
import json
from bs4 import BeautifulSoup


def scrape_hockey_teams(page=1):
    base_url = "https://www.scrapethissite.com/pages/forms/?page_num={}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    hockey_teams = []

    for page_num in range(1, page + 1):
        url = base_url.format(page_num)

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to scrape page {page_num}. Error code: {response.status_code}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        rows = soup.find_all("tr", class_="team")

        for row in rows:
            team_name = row.find("td", class_="name").text.strip()
            year = row.find("td", class_="year").text.strip()
            wins = row.find("td", class_="wins").text.strip()
            losses = row.find("td", class_="losses").text.strip()
            win_percent = row.find("td", class_="pct").text.strip()
            goals_for = row.find("td", class_="gf").text.strip()
            goals_against = row.find("td", class_="ga").text.strip()
            diff = row.find("td", class_="diff").text.strip()


            hockey_teams.append({
                "team_name": team_name,
                "year": year,
                "wins": wins,
                "losses": losses,
                "win_percent": win_percent,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "diff": diff
            })
    return hockey_teams

def save_to_csv(scraped_data, filename='hockey_teams.csv'):
    if not scraped_data:
        return

    with open(filename, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=scraped_data[0].keys())
        writer.writeheader()
        writer.writerows(scraped_data)
    print(f"Save to csv: {filename}")

def save_to_tsv(scraped_data, filename="hockey_teams.tsv"):
    if not data:
        return
    with open(filename, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=scraped_data[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(scraped_data)
    print(f"Save to tsv: {filename}")


def save_to_json(scraped_data, filename="hockey_teams.json"):
    with open(filename, mode="w", encoding="utf-8") as file:
        json.dump(scraped_data, file, ensure_ascii=False, indent=4)
    print(f"Save to json: {filename}")

data = scrape_hockey_teams(24)
save_to_csv(data)
save_to_tsv(data)
save_to_json(data)