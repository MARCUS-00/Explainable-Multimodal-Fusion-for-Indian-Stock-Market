import os, time
import pandas as pd
from collections import defaultdict

from config.settings import DATE_START, DATE_END, STOCKS

try:
    from nsepython import nsefetch
except ImportError:
    _NSEPYTHON_OK = False
    nsefetch = None
else:
    _NSEPYTHON_OK = True

START_DATE  = DATE_START
END_DATE    = DATE_END
STOCK_COUNT = len(STOCKS)

_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(_BASE_DIR, "data", "events")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "events.csv")

EVENT_SCORES = {
    "EARNINGS":      1.0,
    "UNION_BUDGET":  1.0,
    "RBI_POLICY":    1.0,
    "REPO_RATE":     1.0,
    "GDP":           0.9,
    "STOCK_SPLIT":   0.9,
    "BONUS":         0.9,
    "MERGER":        0.9,
    "CPI":           0.8,
    "PMI":           0.8,
    "DIVIDEND":      0.7,
    "BOARD_MEETING": 0.7,
    "NONE":          0.0,
}

_ALL_RBI_POLICY_DATES = {
    "2020-02-06", "2020-03-27", "2020-05-22", "2020-08-06", "2020-10-09", "2020-12-04",
    "2021-02-05", "2021-04-07", "2021-06-04", "2021-08-06", "2021-10-08", "2021-12-08",
    "2022-02-10", "2022-04-08", "2022-05-04", "2022-06-08", "2022-08-05", "2022-09-30", "2022-12-07",
    "2023-02-08", "2023-04-06", "2023-06-08", "2023-08-10", "2023-10-06", "2023-12-08",
    "2024-02-08", "2024-04-05", "2024-06-07", "2024-08-08", "2024-10-09", "2024-12-06",
    "2025-02-07", "2025-04-09", "2025-06-06", "2025-08-06", "2025-10-08", "2025-12-05",
    "2026-02-06", "2026-04-09", "2026-06-05", "2026-08-06", "2026-10-08", "2026-12-04",
}

_ALL_REPO_RATE_CHANGE_DATES = {
    "2020-03-27", "2020-05-22",
    "2022-05-04", "2022-06-08", "2022-08-05", "2022-09-30", "2022-12-07",
    "2023-02-08",
    "2025-02-07", "2025-04-09",
    "2026-02-06",
}

# B-EVENTS: real published release dates (replace synthetic first-of-month rules)
_ALL_UNION_BUDGET_DATES = {
    "2020-02-01", "2021-02-01", "2022-02-01", "2023-02-01",
    "2024-02-01", "2024-07-23", "2025-02-01", "2026-02-02",
}

_ALL_CPI_DATES = {
    "2020-01-13","2020-02-12","2020-03-12","2020-04-13","2020-05-12",
    "2020-06-12","2020-07-13","2020-08-12","2020-09-14","2020-10-12",
    "2020-11-12","2020-12-14","2021-01-12","2021-02-12","2021-03-12",
    "2021-04-12","2021-05-12","2021-06-14","2021-07-12","2021-08-12",
    "2021-09-13","2021-10-12","2021-11-12","2021-12-13","2022-01-12",
    "2022-02-14","2022-03-14","2022-04-12","2022-05-12","2022-06-13",
    "2022-07-12","2022-08-12","2022-09-12","2022-10-12","2022-11-14",
    "2022-12-12","2023-01-12","2023-02-13","2023-03-13","2023-04-12",
    "2023-05-12","2023-06-12","2023-07-12","2023-08-14","2023-09-12",
    "2023-10-12","2023-11-13","2023-12-12","2024-01-12","2024-02-12",
    "2024-03-12","2024-04-12","2024-05-13","2024-06-12","2024-07-12",
    "2024-08-12","2024-09-12","2024-10-14","2024-11-12","2024-12-12",
    "2025-01-13","2025-02-12","2025-03-12","2025-04-15","2025-05-13",
    "2025-06-12","2025-07-14","2025-08-12","2025-09-12","2025-10-13",
    "2025-11-12","2025-12-12","2026-01-12","2026-02-12","2026-03-12",
    "2026-04-13",
}

_ALL_GDP_DATES = {
    "2020-02-28","2020-05-29","2020-08-31","2020-11-27","2021-02-26",
    "2021-05-31","2021-08-31","2021-11-30","2022-02-28","2022-05-31",
    "2022-08-31","2022-11-30","2023-02-28","2023-05-31","2023-08-31",
    "2023-11-30","2024-02-29","2024-05-31","2024-08-30","2024-11-29",
    "2025-02-28","2025-05-30","2025-08-29","2025-11-28","2026-02-27",
}

_ALL_PMI_DATES = {
    "2020-01-02","2020-02-03","2020-03-02","2020-04-01","2020-05-04",
    "2020-06-01","2020-07-01","2020-08-03","2020-09-01","2020-10-01",
    "2020-11-02","2020-12-01","2021-01-04","2021-02-01","2021-03-01",
    "2021-04-01","2021-05-03","2021-06-01","2021-07-01","2021-08-02",
    "2021-09-01","2021-10-01","2021-11-01","2021-12-01","2022-01-03",
    "2022-02-01","2022-03-01","2022-04-01","2022-05-02","2022-06-01",
    "2022-07-01","2022-08-01","2022-09-01","2022-10-03","2022-11-01",
    "2022-12-01","2023-01-02","2023-02-01","2023-03-01","2023-04-03",
    "2023-05-02","2023-06-01","2023-07-03","2023-08-01","2023-09-01",
    "2023-10-03","2023-11-01","2023-12-01","2024-01-02","2024-02-01",
    "2024-03-01","2024-04-01","2024-05-02","2024-06-03","2024-07-01",
    "2024-08-01","2024-09-02","2024-10-01","2024-11-01","2024-12-02",
    "2025-01-02","2025-02-03","2025-03-03","2025-04-01","2025-05-02",
    "2025-06-02","2025-07-01","2025-08-01","2025-09-01","2025-10-01",
    "2025-11-03","2025-12-01","2026-01-02","2026-02-02","2026-03-02",
    "2026-04-01",
}


def _fallback_to_calendar_rule(date_set, start, end, rule_fn):
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    if not date_set:
        published_max = s - pd.Timedelta(days=1)
    else:
        published_max = max(pd.to_datetime(d) for d in date_set)
    # FIX B-EVENTS: returns (full_set, estimated_set). Real published dates
    # are kept; the rule only generates synthetic dates past the latest
    # published date. Estimated dates are tracked separately so event rows
    # can be tagged with is_estimated=1.
    result = set(date_set)
    estimated = set()
    cur = max(published_max + pd.Timedelta(days=1), s)
    while cur <= e:
        if rule_fn(cur):
            ds = cur.strftime("%Y-%m-%d")
            result.add(ds)
            estimated.add(ds)
        cur = cur + pd.Timedelta(days=1)
    if estimated:
        import warnings
        warnings.warn(
            f"B-EVENTS: {len(estimated)} synthetic dates emitted past "
            f"{published_max.date()} (calendar-rule fallback).",
            stacklevel=2,
        )
    return result, estimated


def _filter_dates_to_range(date_set, start, end):
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    result = set()
    for d in date_set:
        try:
            if s <= pd.to_datetime(d) <= e:
                result.add(d)
        except Exception:
            pass
    return result


def fetch_corporate_events(symbol, start_date, end_date):
    events = []
    current = pd.to_datetime(start_date)
    final_end = pd.to_datetime(end_date)
    print(f"  {symbol:<14}", end="", flush=True)

    if not _NSEPYTHON_OK or nsefetch is None:
        print(" [SKIP] nsepython not available")
        return []

    while current <= final_end:
        chunk_end = min(current + pd.DateOffset(months=3), final_end)
        s_str = current.strftime("%d-%m-%Y")
        e_str = chunk_end.strftime("%d-%m-%Y")
        from urllib.parse import quote as _quote
        _sym = _quote(symbol, safe="")
        corp_url = (f"https://www.nseindia.com/api/corporate-announcements"
                    f"?index=equities&symbol={_sym}&from_date={s_str}&to_date={e_str}")
        board_url = (f"https://www.nseindia.com/api/event-calendar"
                     f"?index=equities&symbol={_sym}&from_date={s_str}&to_date={e_str}")
        for url in [corp_url, board_url]:
            try:
                data = nsefetch(url)
                if isinstance(data, list) and len(data) > 0:
                    events.extend(data)
                    print(".", end="", flush=True)
                elif isinstance(data, dict) and "data" in data and len(data["data"]) > 0:
                    events.extend(data["data"])
                    print(".", end="", flush=True)
                else:
                    print("x", end="", flush=True)
            except Exception:
                print("x", end="", flush=True)
        current = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.5)

    print(f" [DONE: {len(events)} events]")
    return events


def _classify_event(raw_purpose):
    p = str(raw_purpose).upper()
    if any(k in p for k in ["FINANCIAL RESULTS", "EARNINGS"]):
        return "EARNINGS"
    if "DIVIDEND"      in p: return "DIVIDEND"
    if "BOARD MEETING" in p: return "BOARD_MEETING"
    if "SPLIT"         in p: return "STOCK_SPLIT"
    if "BONUS"         in p: return "BONUS"
    if "MERGER"        in p: return "MERGER"
    return "NONE"


## removed orphan main() (was an empty placeholder)


def _build_macro_event_dates(start_date, end_date):
    RBI_POLICY_DATES = _filter_dates_to_range(_ALL_RBI_POLICY_DATES, start_date, end_date)
    REPO_RATE_DATES = _filter_dates_to_range(_ALL_REPO_RATE_CHANGE_DATES, start_date, end_date)
    union_budget_dates = _filter_dates_to_range(_ALL_UNION_BUDGET_DATES, start_date, end_date)
    cpi_dates = _filter_dates_to_range(_ALL_CPI_DATES, start_date, end_date)
    gdp_dates = _filter_dates_to_range(_ALL_GDP_DATES, start_date, end_date)
    pmi_dates = _filter_dates_to_range(_ALL_PMI_DATES, start_date, end_date)

    trading_days = pd.date_range(start=start_date, end=end_date, freq="B")
    trading_set = set(trading_days)
    first_wd_of_month = set(
        trading_days.to_series().groupby([trading_days.year, trading_days.month]).first()
    )
    cpi_dates, cpi_est = _fallback_to_calendar_rule(
        cpi_dates, start_date, end_date,
        rule_fn=lambda d: (d in trading_set) and (d.day == 12))
    pmi_dates, pmi_est = _fallback_to_calendar_rule(
        pmi_dates, start_date, end_date,
        rule_fn=lambda d: d in first_wd_of_month)
    gdp_dates, gdp_est = _fallback_to_calendar_rule(
        gdp_dates, start_date, end_date,
        rule_fn=lambda d: (d in trading_set) and (d.month in (2, 5, 8, 11)) and (d.day >= 26))
    estimated_dates = cpi_est | pmi_est | gdp_est
    return (trading_days, RBI_POLICY_DATES, REPO_RATE_DATES,
            union_budget_dates, cpi_dates, gdp_dates, pmi_dates, estimated_dates)


def _collect_corporate_event_map(start_date, end_date):
    corporate_event_map = defaultdict(list)
    for symbol in STOCKS:
        raw = fetch_corporate_events(symbol, start_date, end_date)
        for item in raw:
            date_val = item.get("date") or item.get("bm_date") or item.get("exDate")
            if not date_val:
                continue
            try:
                parsed = pd.to_datetime(date_val).strftime("%Y-%m-%d")
            except Exception:
                continue
            event_type = _classify_event(item.get("purpose", item.get("subject", "")))
            if event_type != "NONE":
                key = (parsed, symbol)
                if event_type not in corporate_event_map[key]:
                    corporate_event_map[key].append(event_type)
    return corporate_event_map


def _build_event_rows(trading_days, corporate_event_map, union_budget_dates,
                      RBI_POLICY_DATES, REPO_RATE_DATES, gdp_dates, cpi_dates, pmi_dates,
                      estimated_dates=None):
    rows = []
    for date in trading_days:
        date_str = date.strftime("%Y-%m-%d")
        for symbol in STOCKS:
            day_events = []
            categories = set()
            if date_str in union_budget_dates: day_events.append("UNION_BUDGET"); categories.add("GOVT")
            if date_str in RBI_POLICY_DATES:   day_events.append("RBI_POLICY");   categories.add("GOVT")
            if date_str in REPO_RATE_DATES:    day_events.append("REPO_RATE");    categories.add("GOVT")
            if date_str in gdp_dates:          day_events.append("GDP");          categories.add("MACRO")
            if date_str in cpi_dates:          day_events.append("CPI");          categories.add("MACRO")
            if date_str in pmi_dates:          day_events.append("PMI");          categories.add("MACRO")
            for ev in corporate_event_map.get((date_str, symbol), []):
                if ev not in day_events:
                    day_events.append(ev)
                    categories.add("STOCK")
            if day_events:
                day_events.sort()
                event_name     = "|".join(day_events)
                event_category = "|".join(sorted(categories))
                event_score    = max(EVENT_SCORES.get(e, 0.0) for e in day_events)
                is_event       = 1
            else:
                event_name = event_category = "NONE"
                event_score = 0.0
                is_event    = 0
            rows.append({
                "date":            date_str,
                "symbol":          symbol,
                "event_category":  event_category,
                "event_name":      event_name,
                "event_score_max": event_score,
                "event_count":     len(day_events),
                "is_event":        is_event,
                # FIX B-EVENTS: tag estimated (calendar-rule fallback) macro dates
                "is_estimated":    1 if (estimated_dates and (date_str in estimated_dates) and is_event) else 0,
            })
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trading_days, RBI_POLICY_DATES, REPO_RATE_DATES, union_budget_dates, cpi_dates, gdp_dates, pmi_dates, estimated_dates = _build_macro_event_dates(START_DATE, END_DATE)

    print("=" * 60)
    print("  BUILD EVENTS DATASET")
    print(f"  RBI policy dates : {len(RBI_POLICY_DATES)} (real)")
    print(f"  Repo rate dates  : {len(REPO_RATE_DATES)} (real)")
    print(f"  Union budget     : {len(union_budget_dates)} (real)")
    print(f"  CPI release      : {len(cpi_dates)} (real)")
    print(f"  GDP release      : {len(gdp_dates)} (real)")
    print(f"  PMI release      : {len(pmi_dates)} (real)")
    print("=" * 60)

    print("\n  Fetching NSE corporate events:")
    print("  " + "-" * 50)
    corporate_event_map = _collect_corporate_event_map(START_DATE, END_DATE)

    print("\n  Constructing final dataset (macro + corporate) ...")
    rows = _build_event_rows(trading_days, corporate_event_map, union_budget_dates,
                             RBI_POLICY_DATES, REPO_RATE_DATES, gdp_dates, cpi_dates, pmi_dates,
                             estimated_dates=estimated_dates)

    df = pd.DataFrame(rows, columns=[
        "date", "symbol", "event_category", "event_name",
        "event_score_max", "event_count", "is_event", "is_estimated",
    ])

    # Drop NONE rows — dates with no event for a stock carry zero information.
    # build_features.py uses a left-merge, so missing rows naturally become 0
    # for all event features.  Keeping NONE rows triples the file size with
    # pure noise and slows down every merge.
    total_rows   = len(df)
    df = df[df["is_event"] == 1].reset_index(drop=True)
    dropped      = total_rows - len(df)
    print(f"  [INFO] Dropped {dropped:,} NONE rows ({dropped/total_rows*100:.1f}% of total)")
    print(f"  [INFO] Kept {len(df):,} real event rows only")

    df = df.rename(columns={"date": "Date", "symbol": "Stock"})
    df.to_csv(OUTPUT_FILE, index=False)

    # Warn on any estimated (synthetic) macro dates so downstream code can
    # account for them if needed.
    n_est = int(df["is_estimated"].sum()) if "is_estimated" in df.columns else 0
    if n_est > 0:
        print(f"  [WARN] {n_est} event rows are marked as estimated (calendar-rule fallback).")
    print(f"\n[OK] Saved -> {OUTPUT_FILE}  rows={len(df)}  event_days={int(df['is_event'].sum())}")


if __name__ == "__main__":
    main()
