import os, re, sys, warnings
from typing import Any
import pandas as pd
from bs4 import BeautifulSoup, Comment
from urllib.parse import urljoin

warnings.filterwarnings("ignore")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from config.settings import DATE_START, DATE_END, STOCKS, MC_SLUG_MAP

START_DATE = DATE_START
END_DATE   = DATE_END
STOCK_COUNT = len(STOCKS)
MAX_ARTICLES_PER_STOCK = 1000
MIN_ARTICLES_THRESHOLD = 2
FETCH_DELAY_SECONDS = 0

FY_START = pd.to_datetime(START_DATE)
FY_END   = pd.to_datetime(END_DATE)

_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(_BASE_DIR, "data", "news")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "news.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://www.moneycontrol.com/",
}

_DATE_PATTERN = re.compile(r"<span[^>]*>(.*?)</span>")
_IST_PATTERN  = re.compile(r"IST.*")
_ARTICLE_CARD_SELECTOR = 'li.clearfix[id^="newslist-"]'
_PAGINATION_NEXT_SELECTOR = 'nav.pagenation[aria-label="Pagination"] a[aria-label="Go to Next page"]'
_SPINNER_FRAMES = ("|", "/", "-", "\\")


def _ensure_playwright() -> bool:
    """Defer the optional Playwright import so the module stays importable."""
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except ImportError:
        print("[ERROR] `playwright` package not installed.")
        print("        Install it with: pip install -r requirements.txt")
        print("        Then install Chromium with: python -m playwright install chromium")
        return False


def _extract_article_date(card):
    """Moneycontrol renders the publication timestamp inside an HTML comment."""
    for comment in card.find_all(string=lambda value: isinstance(value, Comment)):
        match = re.search(r"<span[^>]*>(.*?)</span>", str(comment), flags=re.S)
        if match:
            raw_date = _IST_PATTERN.sub("", match.group(1)).strip()
            parsed_date = pd.to_datetime(raw_date, errors="coerce")
            if not pd.isna(parsed_date):
                return parsed_date

    for span in card.find_all("span"):
        raw_date = _IST_PATTERN.sub("", span.get_text(" ", strip=True)).strip()
        parsed_date = pd.to_datetime(raw_date, errors="coerce")
        if not pd.isna(parsed_date):
            return parsed_date

    return None


def _parse_news_page(page_content, stock, page_url):
    soup = BeautifulSoup(page_content, "html.parser")
    cards = soup.select(_ARTICLE_CARD_SELECTOR)
    if not cards:
        cards = [
            li for li in soup.select("main li.clearfix")
            if li.find("h2") and li.find("a", href=True)
        ]
    if not cards:
        return [], False

    articles = []
    should_stop = False
    for card in cards:
        h2 = card.find("h2")
        link = card.find("a", href=True)
        if not h2 or not link:
            continue

        parsed_date = _extract_article_date(card)
        if parsed_date is None:
            continue

        if parsed_date < FY_START:
            should_stop = True
            break
        if parsed_date > FY_END:
            continue

        title = h2.get_text(" ", strip=True)
        article_href = link.get("href")
        article_url = urljoin(page_url, str(article_href).strip()) if article_href else page_url
        articles.append({
            "Date": parsed_date.strftime("%Y-%m-%d"),
            "Stock": stock,
            "News_Text": title,
            "Source": "Moneycontrol",
            "_article_url": article_url,
        })

    return articles, should_stop


def _load_tag_page(page, url, attempt=1, max_attempts=3):
    try:
        response = page.goto(url, wait_until="domcontentloaded", timeout=30000)
        if response is not None and response.status in (403, 429):
            raise RuntimeError(f"HTTP {response.status} for {url}")
        try:
            page.wait_for_selector("main li.clearfix h2", timeout=15000)
        except Exception:
            page.wait_for_selector("nav.pagenation[aria-label='Pagination']", timeout=15000)
        page.wait_for_timeout(500)
        return page.content()
    except Exception as exc:
        if attempt >= max_attempts:
            raise RuntimeError(f"Failed to load {url}: {type(exc).__name__}: {exc}") from exc
        page.wait_for_timeout(1000 * attempt)
        return _load_tag_page(page, url, attempt=attempt + 1, max_attempts=max_attempts)


def _launch_moneycontrol_browser(playwright, *, headless, channel=None):
    launch_kwargs = {
        "headless": headless,
        "args": [
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
        ],
    }
    if channel is not None:
        launch_kwargs["channel"] = channel
    return playwright.chromium.launch(**launch_kwargs)


def _create_moneycontrol_context(browser):
    context = browser.new_context(
        user_agent=HEADERS["User-Agent"],
        locale="en-US",
        viewport={"width": 1440, "height": 900},
        ignore_https_errors=True,
        extra_http_headers={
            "Accept-Language": HEADERS["Accept-Language"],
            "Referer": HEADERS["Referer"],
        },
    )
    context.set_default_timeout(15000)
    return context


def _collect_moneycontrol_page_articles(page_html, stock, page_url, seen_article_urls):
    page_articles, should_stop = _parse_news_page(page_html, stock, page_url)
    unique_articles = []
    for article in page_articles:
        article_url = article.pop("_article_url", None)
        article_key = article_url or (article["Date"], article["Stock"], article["News_Text"])
        if article_key in seen_article_urls:
            continue
        seen_article_urls.add(article_key)
        unique_articles.append(article)

    soup = BeautifulSoup(page_html, "html.parser")
    next_link = soup.select_one(_PAGINATION_NEXT_SELECTOR)
    next_url = next_link.get("href") if next_link else None
    if next_url:
        next_url = urljoin(page_url, str(next_url).strip())

    return unique_articles, should_stop, next_url


def _print_fetch_progress(stock, spinner_index, page_count, article_count):
    frame = _SPINNER_FRAMES[spinner_index % len(_SPINNER_FRAMES)]
    print(
        f"\r  {stock:<14} fetching {frame} pages={page_count:<3} articles={article_count:<4}",
        end="",
        flush=True,
    )


def _clear_fetch_progress():
    print("\r" + " " * 72 + "\r", end="", flush=True)


def _scrape_moneycontrol_tag_pages(stock, start_url, *, headless, channel=None):
    articles = []
    seen_article_urls = set()
    current_url = start_url
    visited_urls = set()
    page_count = 0
    spinner_index = 0

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = _launch_moneycontrol_browser(playwright, headless=headless, channel=channel)
        try:
            context = _create_moneycontrol_context(browser)
            page = context.new_page()

            consecutive_errors = 0
            for _ in range(100):
                if current_url in visited_urls:
                    break
                visited_urls.add(current_url)
                page_count += 1
                _print_fetch_progress(stock, spinner_index, page_count, len(articles))
                spinner_index += 1

                try:
                    page_html = _load_tag_page(page, current_url)
                except Exception as exc:
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        raise RuntimeError(f"[GIVE UP after 3 errors: {type(exc).__name__}]") from exc
                    continue

                page_articles, should_stop, next_url = _collect_moneycontrol_page_articles(
                    page_html,
                    stock,
                    current_url,
                    seen_article_urls,
                )
                articles.extend(page_articles)
                _print_fetch_progress(stock, spinner_index, page_count, len(articles))
                spinner_index += 1

                if should_stop:
                    break
                if not next_url:
                    break
                if next_url == current_url:
                    break
                current_url = next_url
                consecutive_errors = 0
        finally:
            browser.close()

    _clear_fetch_progress()
    return articles


def _collect_moneycontrol_articles(stock, start_url):
    launch_variants = [
        {"headless": True, "channel": "chrome"},
        {"headless": True, "channel": "msedge"},
        {"headless": True, "channel": None},
        {"headless": False, "channel": None},
    ]
    for variant in launch_variants:
        try:
            articles = _scrape_moneycontrol_tag_pages(stock, start_url, **variant)
        except Exception:
            if variant["headless"]:
                continue
            raise
        if articles or not variant["headless"]:
            return articles
    return []


def _ensure_transformers() -> bool:
    """B-NEWS: do NOT run pip install at runtime."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        print("[ERROR] `transformers` package not installed.")
        print("        Install it with: pip install -r requirements.txt")
        return False


def _iter_finbert_result_groups(results: Any) -> list[list[dict[str, Any]]]:
    if results is None:
        return []
    if isinstance(results, dict):
        return [[results]]
    if not isinstance(results, list) or not results:
        return []
    first_item = results[0]
    if isinstance(first_item, dict):
        return [results]
    return [group for group in results if isinstance(group, list)]


def add_finbert_sentiment(df, *, strict: bool = False):
    """B-NEWS: optional strict mode raises instead of silent placeholders."""
    finbert_ok = _ensure_transformers()
    sentiment_is_real = False

    if finbert_ok:
        try:
            from transformers import pipeline
            print("\n[INFO] Running FinBERT (ProsusAI/finbert) on headlines ...")
            pipe = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                top_k=None,
                device=-1,
            )
            texts = df["News_Text"].fillna("").tolist()
            results = pipe(texts, batch_size=32, truncation=True, max_length=128)
            pos, neg, neu = [], [], []
            for res_list in _iter_finbert_result_groups(results):
                if not res_list:
                    continue
                s = {
                    str(r.get("label", "")): float(r.get("score", 0.0))
                    for r in res_list
                    if isinstance(r, dict)
                }
                pos.append(s.get("positive", 0.333))
                neg.append(s.get("negative", 0.333))
                neu.append(s.get("neutral",  0.334))
            df = df.copy()
            df["news_positive"] = pos
            df["news_negative"] = neg
            df["news_neutral"]  = neu
            sentiment_is_real = True
            print("[INFO] FinBERT scoring complete.")
        except Exception as e:
            print(f"\n[ERROR] FinBERT failed: {type(e).__name__}: {e}")
            if strict:
                raise
            print("[WARNING] Falling back to neutral placeholders (strict=False).")

    if not sentiment_is_real:
        if strict:
            raise RuntimeError(
                "FinBERT unavailable and strict=True. Install transformers."
            )
        print("\n[WARNING] *** SENTIMENT IS NEUTRAL PLACEHOLDER ***")
        df = df.copy()
        df["news_positive"] = 0.333
        df["news_negative"] = 0.333
        df["news_neutral"]  = 0.334

    df["news_score"] = df["news_positive"] - df["news_negative"]
    return df


def _validate_sentiment(df):
    print("\n" + "=" * 55)
    print("  SENTIMENT VALIDATION SUMMARY")
    print("=" * 55)
    sc = df["news_score"]
    print(f"  Total articles  : {len(df)}")
    print(f"  news_score min  : {sc.min():.4f}")
    print(f"  news_score max  : {sc.max():.4f}")
    print(f"  news_score mean : {sc.mean():.4f}")
    print(f"  news_score std  : {sc.std():.4f}")
    pct_zeros = (sc == 0).mean() * 100
    print(f"  % zeros         : {pct_zeros:.1f}%")
    print("\n  Distribution buckets:")
    print(f"    Positive (>0.1)      : {(sc > 0.1).sum()} ({(sc > 0.1).mean():.1%})")
    print(f"    Neutral (-0.1 to 0.1): {((sc >= -0.1) & (sc <= 0.1)).sum()}")
    print(f"    Negative (<-0.1)     : {(sc < -0.1).sum()} ({(sc < -0.1).mean():.1%})")
    print("=" * 55)


def fetch_news(stock):
    slug = MC_SLUG_MAP.get(stock, stock.lower())
    start_url = f"https://www.moneycontrol.com/news/tags/{slug}.html"
    if not _ensure_playwright():
        print(" [SKIP] Playwright unavailable")
        return []

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(" [SKIP] Playwright import failed")
        return []

    try:
        articles = _collect_moneycontrol_articles(stock, start_url)
    except Exception as exc:
        _clear_fetch_progress()
        print(f"  {stock:<14}[ERROR] {type(exc).__name__}: {exc}")
        articles = []

    count = len(articles)
    if count < MIN_ARTICLES_THRESHOLD:
        print(f"  {stock:<14}[SKIP] Only {count} articles (min {MIN_ARTICLES_THRESHOLD})")
        return []
    print(f"  {stock:<14}[OK] {count} articles")
    return articles[:MAX_ARTICLES_PER_STOCK]


def clean_data(df):
    df = df.dropna(subset=["News_Text", "Date"])
    df["News_Text"] = (
        df["News_Text"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    )
    df = df[df["News_Text"].str.len() >= 10]
    df = df.drop_duplicates(subset=["News_Text"], keep="first")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[(df["Date"] >= FY_START) & (df["Date"] <= FY_END)]
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df.sort_values(by=["Date", "Stock", "News_Text"]).reset_index(drop=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  BUILD NEWS DATASET")
    print(f"  Date range : {START_DATE} -> {END_DATE}")
    print("=" * 60)

    all_articles = []
    successful = []
    try:
        for stock in STOCKS:
            news = fetch_news(stock)
            if news:
                all_articles.extend(news)
                successful.append(stock)
            else:
                print(f"  {stock:<14}[NO NEWS] news_score=0 in merged")
            if FETCH_DELAY_SECONDS > 0:
                from time import sleep
                sleep(FETCH_DELAY_SECONDS)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    if not all_articles:
        print("\n[ERROR] No articles collected.")
        return

    df_clean = clean_data(pd.DataFrame(all_articles))
    if df_clean.empty:
        print("\n[ERROR] No articles survived cleaning.")
        return

    # Fail loudly by default — silent neutral placeholders would corrupt
    # every downstream sentiment feature with no visible error.
    # Set EXPLAINABLE_MULTIMODAL_FUSION_FOR_INDIAN_STOCK_MARKET_ALLOW_NEUTRAL_NEWS=1
    # to keep graceful degradation.
    _allow_neutral = os.environ.get(
        "EXPLAINABLE_MULTIMODAL_FUSION_FOR_INDIAN_STOCK_MARKET_ALLOW_NEUTRAL_NEWS"
    ) == "1"
    df_clean = add_finbert_sentiment(df_clean, strict=not _allow_neutral)
    cols_to_save = [
        "Date", "Stock", "News_Text", "Source",
        "news_positive", "news_negative", "news_neutral", "news_score",
    ]
    df_clean[cols_to_save].to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Saved -> {OUTPUT_FILE}  ({len(df_clean)} rows)")
    _validate_sentiment(df_clean)


if __name__ == "__main__":
    main()
