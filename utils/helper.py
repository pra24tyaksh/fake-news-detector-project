from __future__ import annotations

import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:  # pragma: no cover - covered by dependency installation
    trafilatura = None


MIN_TEXT_CHARS = 120
REQUEST_TIMEOUT_SECONDS = 10


class PredictionError(ValueError):
    """Raised for user-facing prediction validation errors."""


def is_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def clean_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_article_text(url: str) -> str:
    if not is_url(url):
        raise PredictionError("Please enter a valid http or https article link.")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise PredictionError(
            "Could not fetch that link. Try another article link or paste the article text manually."
        ) from exc

    html = response.text
    extracted = ""

    if trafilatura is not None:
        extracted = trafilatura.extract(html, url=url, include_comments=False, include_tables=False) or ""

    if not extracted:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        extracted = " ".join(paragraphs)

    extracted = clean_whitespace(extracted)
    if len(extracted) < MIN_TEXT_CHARS:
        raise PredictionError(
            "The article text could not be extracted clearly. Please paste the article text manually."
        )

    return extracted
