"""
A script that downloads and parses Salmon Scotland mortality reports.
"""

from bs4 import BeautifulSoup
from requests import get
import os
from pathlib import Path
import tabula

WEBSITE = "https://www.salmonscotland.co.uk"
REPORT_URL = f"{WEBSITE}/reports/monthly-mortality-rate-%s-%d"


def download(month: str, year: int):
    url = REPORT_URL % (month, year)
    parse_page = get(url).content
    parser = BeautifulSoup(parse_page, "html.parser")
    div = parser.find("div", class_="download-link")
    a = div.find("a")
    download_link = WEBSITE + a["href"]

    report_out_folder = Path("output/reports/")
    filename = report_out_folder / f"SS-{month}-{year}.pdf"
    os.makedirs(str(report_out_folder), exist_ok=True)
    downloaded_pdf = get(download_link)

    with filename.open("wb") as f:
        f.write(downloaded_pdf.content)

    return filename


# download("November", 2021)

# TODO: add tabula execution and parsing...
