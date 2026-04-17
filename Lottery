
import io
import re
import time
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="ロト・ナンバーズAI予想 完全版 Pro v2", layout="wide")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

BASE = "https://takarakuji.rakuten.co.jp"

PRODUCTS = {
    "NUMBERS3": {
        "past_url": f"{BASE}/backnumber/numbers3_past/",
        "recent_prefix": "/backnumber/numbers3/",
        "detail_prefix": "/backnumber/numbers3_detail/",
        "kind": "numbers",
        "digits": 3,
        "price": 200,
    },
    "NUMBERS4": {
        "past_url": f"{BASE}/backnumber/numbers4_past/",
        "recent_prefix": "/backnumber/numbers4/",
        "detail_prefix": "/backnumber/numbers4_detail/",
        "kind": "numbers",
        "digits": 4,
        "price": 200,
    },
    "MINI LOTO": {
        "past_url": f"{BASE}/backnumber/mini_past/",
        "recent_prefix": "/backnumber/mini/",
        "detail_prefix": "/backnumber/miniloto_detail/",
        "kind": "loto",
        "count": 5,
        "max_num": 31,
        "price": 200,
    },
    "LOTO6": {
        "past_url": f"{BASE}/backnumber/loto6_past/",
        "recent_prefix": "/backnumber/loto6/",
        "detail_prefix": "/backnumber/loto6_detail/",
        "kind": "loto",
        "count": 6,
        "max_num": 43,
        "price": 200,
    },
}

# -------------------------
# utility
# -------------------------
def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()

def rank_to_score(s: pd.Series, ascending: bool = False) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return x.rank(pct=True, ascending=ascending).fillna(0.0)

def count_consecutive(nums: List[int]) -> int:
    nums = sorted(nums)
    return sum(nums[i + 1] - nums[i] == 1 for i in range(len(nums) - 1))

def to_excel_bytes(df_data: pd.DataFrame, df_pred: pd.DataFrame, extra: Dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df_data.to_excel(writer, sheet_name="過去データ", index=False)
        df_pred.to_excel(writer, sheet_name="予想", index=False)
        for name, df in extra.items():
            if df is not None and len(df) > 0:
                df.to_excel(writer, sheet_name=name[:31], index=False)
    return bio.getvalue()

# -------------------------
# fetch
# -------------------------
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_html(url: str) -> str:
    sess = requests.Session()
    sess.headers.update(HEADERS)
    resp = sess.get(url, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"
    return resp.text

def soup_from_url(url: str) -> BeautifulSoup:
    return BeautifulSoup(fetch_html(url), "html.parser")

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def collect_detail_urls(product_name: str, recent_only: bool = True) -> List[str]:
    meta = PRODUCTS[product_name]
    soup = soup_from_url(meta["past_url"])
    urls: List[str] = []

    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        if href.startswith(meta["recent_prefix"]):
            # mini/lastresults は重複しやすいので除外
            if "lastresults" in href:
                continue
            urls.append(BASE + href)
        elif not recent_only and href.startswith(meta["detail_prefix"]):
            urls.append(BASE + href)

    # MINI LOTO は past ページ上のリンク構造が変わることがあるので、
    # recent_only 時は当月〜過去12か月の月別URLも直接補完する
    if product_name == "MINI LOTO" and recent_only:
        now = pd.Timestamp.today().normalize().replace(day=1)
        for i in range(0, 13):
            ym = (now - pd.DateOffset(months=i)).strftime("%Y%m")
            urls.append(f"{BASE}/backnumber/mini/{ym}/")

    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

# -------------------------
# parser
# -------------------------
def parse_numbers_monthly_page(url: str, digits: int) -> List[Dict[str, Any]]:
    text = normalize_spaces(soup_from_url(url).get_text(" ", strip=True))
    pattern = re.compile(
        rf"回号\s*第(\d+)回\s*抽せん日\s*(\d{{4}}/\d{{2}}/\d{{2}})\s*当せん番号\s*(\d{{{digits}}})"
    )
    rows = []
    for draw_no, draw_date, num in pattern.findall(text):
        row = {"回号": int(draw_no), "抽選日": draw_date, "番号": num.zfill(digits)}
        for i, ch in enumerate(row["番号"], start=1):
            row[f"d{i}"] = int(ch)
        rows.append(row)
    return rows

def parse_numbers_detail_page(url: str, digits: int) -> List[Dict[str, Any]]:
    text = normalize_spaces(soup_from_url(url).get_text(" ", strip=True))
    pattern = re.compile(rf"第(\d+)回\s+(\d{{4}}/\d{{2}}/\d{{2}})\s+(\d{{{digits}}})")
    rows = []
    for draw_no, draw_date, num in pattern.findall(text):
        row = {"回号": int(draw_no), "抽選日": draw_date, "番号": num.zfill(digits)}
        for i, ch in enumerate(row["番号"], start=1):
            row[f"d{i}"] = int(ch)
        rows.append(row)
    return rows

def _extract_loto_rows_from_text(text: str, count: int) -> List[Dict[str, Any]]:
    text = normalize_spaces(text)
    rows: List[Dict[str, Any]] = []

    # パターン1: 「本数字 2 3 4 20 28 ボーナス数字 (12)」
    pattern1 = re.compile(
        rf"回号\s*第(\d+)回\s*抽せん日\s*(\d{{4}}/\d{{2}}/\d{{2}})\s*本数字\s*((?:\d+\s+){{{count-1}}}\d+)\s*ボーナス数字\s*\(?(\d+)\)?"
    )
    for draw_no, draw_date, nums_text, bonus in pattern1.findall(text):
        nums = [int(x) for x in nums_text.split()]
        if len(nums) == count:
            row = {
                "回号": int(draw_no),
                "抽選日": draw_date,
                "本数字": " ".join(map(str, nums)),
                "ボーナス": int(bonus),
            }
            for i, n in enumerate(nums, start=1):
                row[f"n{i}"] = n
            rows.append(row)

    if rows:
        return rows

    # パターン2: 「第1381回 2026/04/07 2 3 4 20 28 12」
    pattern2 = re.compile(
        rf"第(\d+)回\s+(\d{{4}}/\d{{2}}/\d{{2}})\s+((?:\d+\s+){{{count}}}\d+)"
    )
    for draw_no, draw_date, nums_text in pattern2.findall(text):
        nums = [int(x) for x in nums_text.split()]
        if len(nums) != count + 1:
            continue
        main_nums = nums[:count]
        bonus = nums[count]
        row = {
            "回号": int(draw_no),
            "抽選日": draw_date,
            "本数字": " ".join(map(str, main_nums)),
            "ボーナス": int(bonus),
        }
        for i, n in enumerate(main_nums, start=1):
            row[f"n{i}"] = n
        rows.append(row)

    return rows

def parse_loto_monthly_page(url: str, count: int) -> List[Dict[str, Any]]:
    return _extract_loto_rows_from_text(soup_from_url(url).get_text(" ", strip=True), count)

def parse_loto_detail_page(url: str, count: int) -> List[Dict[str, Any]]:
    return _extract_loto_rows_from_text(soup_from_url(url).get_text(" ", strip=True), count)

# -------------------------
# scraping
# -------------------------
def scrape_numbers(product_name: str, digits: int, recent_only: bool = True) -> pd.DataFrame:
    urls = collect_detail_urls(product_name, recent_only=recent_only)
    rows: List[Dict[str, Any]] = []
    prog = st.progress(0, text="データ取得中...")
    for i, url in enumerate(urls):
        try:
            if "_detail/" in url:
                rows.extend(parse_numbers_detail_page(url, digits))
            else:
                rows.extend(parse_numbers_monthly_page(url, digits))
        except Exception:
            pass
        prog.progress((i + 1) / max(1, len(urls)), text=f"データ取得中... {i+1}/{len(urls)}")
        time.sleep(0.03)
    prog.empty()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["回号"]).sort_values("回号", ascending=False).reset_index(drop=True)

def scrape_loto(product_name: str, count: int, recent_only: bool = True) -> pd.DataFrame:
    urls = collect_detail_urls(product_name, recent_only=recent_only)
    rows: List[Dict[str, Any]] = []
    prog = st.progress(0, text="データ取得中...")
    for i, url in enumerate(urls):
        try:
            if "_detail/" in url:
                rows.extend(parse_loto_detail_page(url, count))
            else:
                rows.extend(parse_loto_monthly_page(url, count))
        except Exception:
            pass
        prog.progress((i + 1) / max(1, len(urls)), text=f"データ取得中... {i+1}/{len(urls)}")
        time.sleep(0.03)
    prog.empty()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["回号"]).sort_values("回号", ascending=False).reset_index(drop=True)

# -------------------------
# scoring
# -------------------------
def numbers_position_stats(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    frames = []
    for pos in range(1, digits + 1):
        col = f"d{pos}"
        freq = df[col].value_counts().reindex(range(10), fill_value=0)
        recent_freq = []
        overdue = []
        streak = []
        for num in range(10):
            ser = (df[col] == num).astype(int).reset_index(drop=True)
            idxs = np.where(ser.values == 1)[0]
            overdue.append(int(idxs[0]) if len(idxs) else len(df))
            recent_freq.append(int(ser.head(min(30, len(df))).sum()))
            s = 0
            for v in ser.values:
                if v == 1:
                    s += 1
                else:
                    break
            streak.append(s)
        frames.append(pd.DataFrame({
            "position": pos,
            "digit": range(10),
            "freq_all": freq.values,
            "freq_recent30": recent_freq,
            "overdue": overdue,
            "hit_streak": streak,
        }))
    return pd.concat(frames, ignore_index=True)

def score_numbers_candidates(df: pd.DataFrame, digits: int, config: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pos_stats = numbers_position_stats(df, digits)
    recent_numbers = df["番号"].astype(str).tolist()
    top_digits_by_pos = {}

    for pos in range(1, digits + 1):
        tmp = pos_stats[pos_stats["position"] == pos].copy()
        tmp["base"] = (
            rank_to_score(tmp["freq_all"], ascending=False) * config["w_freq"]
            + rank_to_score(tmp["freq_recent30"], ascending=False) * config["w_recent"]
            + rank_to_score(tmp["overdue"], ascending=False) * config["w_overdue"]
            + rank_to_score(tmp["hit_streak"], ascending=True) * config["w_streak_avoid"]
        )
        top_digits_by_pos[pos] = tmp.sort_values("base", ascending=False)["digit"].tolist()[: config["top_k_digits"]]

    rng = np.random.default_rng(config["seed"])
    candidates = []
    tries = 0
    while len(candidates) < config["n_candidates"] and tries < config["n_candidates"] * 25:
        tries += 1
        pick = "".join(str(rng.choice(top_digits_by_pos[pos])) for pos in range(1, digits + 1))
        if pick not in candidates:
            candidates.append(pick)

    rows = []
    sum_center = df[[f"d{i}" for i in range(1, digits + 1)]].sum(axis=1).median()

    for c in candidates:
        digits_list = [int(x) for x in c]
        chosen_rows = []
        for pos, d in enumerate(digits_list, start=1):
            chosen_rows.append(pos_stats[(pos_stats["position"] == pos) & (pos_stats["digit"] == d)].iloc[0])

        freq_all_raw = np.mean([r["freq_all"] for r in chosen_rows])
        freq_recent_raw = np.mean([r["freq_recent30"] for r in chosen_rows])
        overdue_raw = np.mean([r["overdue"] for r in chosen_rows])
        streak_raw = np.mean([r["hit_streak"] for r in chosen_rows])
        digit_sum = sum(digits_list)
        odd_cnt = sum(x % 2 == 1 for x in digits_list)
        high_cnt = sum(x >= 5 for x in digits_list)
        unique_cnt = len(set(digits_list))
        same_recent = 1 if c in set(recent_numbers[:30]) else 0
        sum_distance = abs(digit_sum - sum_center)
        odd_balance_distance = abs(odd_cnt - digits / 2)
        high_balance_distance = abs(high_cnt - digits / 2)
        pattern_bonus = {1: 0.1, 2: 0.5, 3: 0.85, 4: 1.0}.get(unique_cnt, 1.0) if digits == 4 else {1: 0.2, 2: 0.8, 3: 1.0}.get(unique_cnt, 1.0)

        rows.append({
            "candidate": c,
            "freq_all_raw": freq_all_raw,
            "freq_recent_raw": freq_recent_raw,
            "overdue_raw": overdue_raw,
            "streak_penalty_raw": streak_raw,
            "digit_sum": digit_sum,
            "odd_cnt": odd_cnt,
            "high_cnt": high_cnt,
            "unique_cnt": unique_cnt,
            "same_recent": same_recent,
            "sum_distance": sum_distance,
            "odd_balance_distance": odd_balance_distance,
            "high_balance_distance": high_balance_distance,
            "pattern_bonus_raw": pattern_bonus,
        })

    scored = pd.DataFrame(rows)
    scored["s_freq_all"] = rank_to_score(scored["freq_all_raw"], ascending=False)
    scored["s_freq_recent"] = rank_to_score(scored["freq_recent_raw"], ascending=False)
    scored["s_overdue"] = rank_to_score(scored["overdue_raw"], ascending=False)
    scored["s_streak_avoid"] = rank_to_score(scored["streak_penalty_raw"], ascending=True)
    scored["s_sum"] = rank_to_score(scored["sum_distance"], ascending=True)
    scored["s_odd_even"] = rank_to_score(scored["odd_balance_distance"], ascending=True)
    scored["s_high_low"] = rank_to_score(scored["high_balance_distance"], ascending=True)
    scored["s_pattern"] = rank_to_score(scored["pattern_bonus_raw"], ascending=False)
    scored["s_recent_avoid"] = scored["same_recent"].map({0: 1.0, 1: 0.0})

    scored["頻度寄与"] = scored["s_freq_all"] * config["w_freq"]
    scored["直近頻度寄与"] = scored["s_freq_recent"] * config["w_recent"]
    scored["未出現寄与"] = scored["s_overdue"] * config["w_overdue"]
    scored["連続回避寄与"] = scored["s_streak_avoid"] * config["w_streak_avoid"]
    scored["合計寄与"] = scored["s_sum"] * config["w_sum"]
    scored["奇数偶数寄与"] = scored["s_odd_even"] * config["w_odd_even"]
    scored["高低寄与"] = scored["s_high_low"] * config["w_high_low"]
    scored["重複パターン寄与"] = scored["s_pattern"] * config["w_pattern"]
    scored["直近重複回避寄与"] = scored["s_recent_avoid"] * config["w_recent_avoid"]

    score_cols = ["頻度寄与","直近頻度寄与","未出現寄与","連続回避寄与","合計寄与","奇数偶数寄与","高低寄与","重複パターン寄与","直近重複回避寄与"]
    scored["AIスコア"] = scored[score_cols].sum(axis=1)
    scored = scored.sort_values(["AIスコア", "overdue_raw"], ascending=False).reset_index(drop=True)
    scored["印"] = ""
    for i, m in enumerate(["◎", "○", "▲", "△", "☆"]):
        if i < len(scored):
            scored.loc[i, "印"] = m
    return scored, pos_stats

def loto_number_stats(df: pd.DataFrame, count: int, max_num: int) -> pd.DataFrame:
    cols = [f"n{i}" for i in range(1, count + 1)]
    rows = []
    for n in range(1, max_num + 1):
        hits = []
        for _, row in df[cols].iterrows():
            hits.append(1 if n in set(row.values.tolist()) else 0)
        idxs = [i for i, v in enumerate(hits) if v == 1]
        streak = 0
        for v in hits:
            if v == 1:
                streak += 1
            else:
                break
        rows.append({
            "num": n,
            "freq_all": int(sum(hits)),
            "freq_recent30": int(sum(hits[:min(30, len(hits))])),
            "overdue": int(idxs[0]) if len(idxs) else len(df),
            "hit_streak": streak,
        })
    return pd.DataFrame(rows)

def loto_pair_stats(df: pd.DataFrame, count: int) -> pd.DataFrame:
    cols = [f"n{i}" for i in range(1, count + 1)]
    ctr = Counter()
    for _, row in df[cols].iterrows():
        nums = sorted(row.values.tolist())
        for a, b in combinations(nums, 2):
            ctr[(a, b)] += 1
    rows = [{"a": k[0], "b": k[1], "pair_freq": v} for k, v in ctr.items()]
    if not rows:
        return pd.DataFrame(columns=["a","b","pair_freq"])
    return pd.DataFrame(rows).sort_values("pair_freq", ascending=False).reset_index(drop=True)

def score_loto_candidates(df: pd.DataFrame, count: int, max_num: int, config: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [f"n{i}" for i in range(1, count + 1)]
    stats = loto_number_stats(df, count, max_num)
    pair_df = loto_pair_stats(df, count)
    recent_sets = [tuple(sorted(x)) for x in df[cols].values.tolist()]

    stats["base"] = (
        rank_to_score(stats["freq_all"], ascending=False) * config["w_freq"]
        + rank_to_score(stats["freq_recent30"], ascending=False) * config["w_recent"]
        + rank_to_score(stats["overdue"], ascending=False) * config["w_overdue"]
        + rank_to_score(stats["hit_streak"], ascending=True) * config["w_streak_avoid"]
    )
    candidate_pool = stats.sort_values("base", ascending=False)["num"].tolist()[: config["pool_size"]]

    sum_center = float(df[cols].sum(axis=1).median())
    odd_center = float(df[cols].apply(lambda r: sum(v % 2 == 1 for v in r), axis=1).median())
    consec_center = float(df[cols].apply(lambda r: count_consecutive(r.tolist()), axis=1).median())
    spread_center = float(df[cols].apply(lambda r: max(r.tolist()) - min(r.tolist()), axis=1).median())

    rng = np.random.default_rng(config["seed"])
    candidates = []
    tries = 0
    while len(candidates) < config["n_candidates"] and tries < config["n_candidates"] * 80:
        tries += 1
        pick = sorted(rng.choice(candidate_pool, count, replace=False).tolist())
        if pick not in candidates:
            candidates.append(pick)

    pair_map = {}
    if not pair_df.empty:
        for r in pair_df.itertuples(index=False):
            pair_map[(int(r.a), int(r.b))] = float(r.pair_freq)

    rows = []
    for candidate in candidates:
        srows = stats.set_index("num").loc[candidate]
        freq_all_raw = srows["freq_all"].mean()
        freq_recent_raw = srows["freq_recent30"].mean()
        overdue_raw = srows["overdue"].mean()
        streak_raw = srows["hit_streak"].mean()

        pair_vals = [pair_map.get((a, b), 0.0) for a, b in combinations(candidate, 2)]
        pair_avg_raw = float(np.mean(pair_vals)) if pair_vals else 0.0

        sum_value = sum(candidate)
        odd_cnt = sum(x % 2 == 1 for x in candidate)
        consecutive_cnt = count_consecutive(candidate)
        spread = max(candidate) - min(candidate)
        overlap_recent = max([len(set(candidate) & set(rs)) for rs in recent_sets[:20]] + [0])
        same_recent = 1 if tuple(candidate) in set(recent_sets[:30]) else 0

        rows.append({
            "candidate": " ".join(map(str, candidate)),
            "freq_all_raw": freq_all_raw,
            "freq_recent_raw": freq_recent_raw,
            "overdue_raw": overdue_raw,
            "streak_penalty_raw": streak_raw,
            "pair_avg_raw": pair_avg_raw,
            "sum_value": sum_value,
            "odd_cnt": odd_cnt,
            "consecutive_cnt": consecutive_cnt,
            "spread": spread,
            "overlap_recent": overlap_recent,
            "same_recent": same_recent,
            "sum_distance": abs(sum_value - sum_center),
            "odd_balance_distance": abs(odd_cnt - odd_center),
            "consec_distance": abs(consecutive_cnt - consec_center),
            "spread_distance": abs(spread - spread_center),
        })

    scored = pd.DataFrame(rows)
    scored["s_freq_all"] = rank_to_score(scored["freq_all_raw"], ascending=False)
    scored["s_freq_recent"] = rank_to_score(scored["freq_recent_raw"], ascending=False)
    scored["s_overdue"] = rank_to_score(scored["overdue_raw"], ascending=False)
    scored["s_streak_avoid"] = rank_to_score(scored["streak_penalty_raw"], ascending=True)
    scored["s_pair"] = rank_to_score(scored["pair_avg_raw"], ascending=False)
    scored["s_sum"] = rank_to_score(scored["sum_distance"], ascending=True)
    scored["s_odd_even"] = rank_to_score(scored["odd_balance_distance"], ascending=True)
    scored["s_consecutive"] = rank_to_score(scored["consec_distance"], ascending=True)
    scored["s_spread"] = rank_to_score(scored["spread_distance"], ascending=True)
    scored["s_recent_overlap_avoid"] = rank_to_score(scored["overlap_recent"], ascending=True)
    scored["s_recent_avoid"] = scored["same_recent"].map({0: 1.0, 1: 0.0})

    scored["頻度寄与"] = scored["s_freq_all"] * config["w_freq"]
    scored["直近頻度寄与"] = scored["s_freq_recent"] * config["w_recent"]
    scored["未出現寄与"] = scored["s_overdue"] * config["w_overdue"]
    scored["連続回避寄与"] = scored["s_streak_avoid"] * config["w_streak_avoid"]
    scored["ペア寄与"] = scored["s_pair"] * config["w_pair"]
    scored["合計寄与"] = scored["s_sum"] * config["w_sum"]
    scored["奇数偶数寄与"] = scored["s_odd_even"] * config["w_odd_even"]
    scored["連番寄与"] = scored["s_consecutive"] * config["w_consecutive"]
    scored["広がり寄与"] = scored["s_spread"] * config["w_spread"]
    scored["直近重複抑制寄与"] = scored["s_recent_overlap_avoid"] * config["w_recent_overlap_avoid"]
    scored["直近完全重複回避寄与"] = scored["s_recent_avoid"] * config["w_recent_avoid"]

    score_cols = ["頻度寄与","直近頻度寄与","未出現寄与","連続回避寄与","ペア寄与","合計寄与","奇数偶数寄与","連番寄与","広がり寄与","直近重複抑制寄与","直近完全重複回避寄与"]
    scored["AIスコア"] = scored[score_cols].sum(axis=1)
    scored = scored.sort_values(["AIスコア", "overdue_raw"], ascending=False).reset_index(drop=True)
    scored["印"] = ""
    for i, m in enumerate(["◎", "○", "▲", "△", "☆"]):
        if i < len(scored):
            scored.loc[i, "印"] = m

    work = df[cols].copy()
    work["連番数"] = work.apply(lambda r: count_consecutive(r.tolist()), axis=1)
    dist = work["連番数"].value_counts().sort_index().rename_axis("連番数").reset_index(name="出現回数")
    dist["出現率"] = (dist["出現回数"] / max(1, len(work)) * 100).round(2)

    return scored, stats.sort_values("base", ascending=False).reset_index(drop=True), pair_df, dist

# -------------------------
# buy plan / sim
# -------------------------
def build_buy_plan(pred: pd.DataFrame, product_name: str, point_limit: int, strategy: str) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    unit_price = PRODUCTS[product_name].get("price", 200)
    plan = pred.head(max(1, point_limit)).copy()

    if strategy == "本命寄り":
        base = list(range(len(plan), 0, -1))
        plan["購入口数"] = base
    elif strategy == "均等":
        plan["購入口数"] = 1
    else:
        plan["購入口数"] = np.where(plan.index < min(3, len(plan)), 2, 1)

    plan["1口金額"] = unit_price
    plan["購入金額"] = plan["購入口数"] * unit_price
    return plan

def numbers_match_count(pred_num: str, actual_num: str) -> int:
    pred_num = str(pred_num).zfill(len(str(actual_num)))
    actual_num = str(actual_num).zfill(len(pred_num))
    return sum(a == b for a, b in zip(pred_num, actual_num))

def loto_match_count(pred_set: List[int], actual_set: List[int]) -> int:
    return len(set(pred_set) & set(actual_set))

def approximate_numbers_payout(product_name: str, match_cnt: int) -> int:
    if product_name == "NUMBERS3":
        return {3: 9000, 2: 1000}.get(match_cnt, 0)
    if product_name == "NUMBERS4":
        return {4: 75000, 3: 5000, 2: 500}.get(match_cnt, 0)
    return 0

def approximate_loto_payout(product_name: str, match_cnt: int) -> int:
    if product_name == "MINI LOTO":
        return {5: 10000000, 4: 10000, 3: 1000}.get(match_cnt, 0)
    if product_name == "LOTO6":
        return {6: 100000000, 5: 300000, 4: 6800, 3: 1000}.get(match_cnt, 0)
    return 0

def simulate_roi_numbers(df: pd.DataFrame, pred: pd.DataFrame, product_name: str, test_rounds: int, point_limit: int) -> pd.DataFrame:
    work = df.sort_values("回号", ascending=False).head(test_rounds).copy()
    picks = pred.head(point_limit)["candidate"].astype(str).tolist()
    unit_price = PRODUCTS[product_name]["price"]
    rows = []
    for _, r in work.iterrows():
        actual = str(r["番号"]).zfill(len(picks[0]))
        total_cost = len(picks) * unit_price
        total_return = 0
        best_match = 0
        for p in picks:
            m = numbers_match_count(p, actual)
            best_match = max(best_match, m)
            total_return += approximate_numbers_payout(product_name, m)
        rows.append({
            "回号": r["回号"],
            "実績": actual,
            "買い目数": len(picks),
            "最高一致数": best_match,
            "購入額": total_cost,
            "払戻想定": total_return,
            "収支": total_return - total_cost,
        })
    return pd.DataFrame(rows)

def simulate_roi_loto(df: pd.DataFrame, pred: pd.DataFrame, product_name: str, test_rounds: int, point_limit: int) -> pd.DataFrame:
    count = PRODUCTS[product_name]["count"]
    cols = [f"n{i}" for i in range(1, count + 1)]
    work = df.sort_values("回号", ascending=False).head(test_rounds).copy()
    picks = [list(map(int, x.split())) for x in pred.head(point_limit)["candidate"].astype(str).tolist()]
    unit_price = PRODUCTS[product_name]["price"]
    rows = []
    for _, r in work.iterrows():
        actual = [int(r[c]) for c in cols]
        total_cost = len(picks) * unit_price
        total_return = 0
        best_match = 0
        for p in picks:
            m = loto_match_count(p, actual)
            best_match = max(best_match, m)
            total_return += approximate_loto_payout(product_name, m)
        rows.append({
            "回号": r["回号"],
            "実績": " ".join(map(str, actual)),
            "買い目数": len(picks),
            "最高一致数": best_match,
            "購入額": total_cost,
            "払戻想定": total_return,
            "収支": total_return - total_cost,
        })
    return pd.DataFrame(rows)

# -------------------------
# UI
# -------------------------
st.title("ロト・ナンバーズAI予想 完全版 Pro v2")
st.caption("MINI LOTO の取得を強化した修正版。")

with st.expander("📊 AI予想の見方", expanded=False):
    st.markdown("""
### AIスコア
- 数値が高いほど有力候補です。
- 当選確率そのものではなく、**過去傾向にどれだけ形が近いか**の総合点です。

### 印
- ◎ 本命
- ○ 安定型
- ▲ 攻め寄り
- △ 中穴
- ☆ 穴狙い

### LOTO系の見方
- **合計**: 数字の合計。極端すぎない候補が上位
- **奇数数**: 奇数の数。偏りすぎない候補が上位
- **連番数**: 連続数字の組数。0〜1程度に寄りやすい
- **ペア平均**: 一緒に出やすい数字同士か
- **平均未出現**: しばらく出ていない数字をどれくらい含むか

### 寄与列
- 頻度寄与 / 直近頻度寄与 / 未出現寄与 / 連続回避寄与 / ペア寄与 / 合計寄与 / 奇数偶数寄与 / 連番寄与 / 広がり寄与
- 値が大きいほど、その要素がその予想を押し上げています。
""")

with st.sidebar:
    product_name = st.selectbox("くじ選択", list(PRODUCTS.keys()))
    recent_only = st.checkbox("直近1年のみ取得", value=True)
    seed = st.number_input("乱数シード", min_value=1, max_value=999999, value=1234, step=1)
    top_n = st.slider("表示する予想件数", 5, 30, 10)

    st.markdown("### 自動買い目")
    point_limit = st.slider("買い目点数", 1, 20, 5)
    buy_strategy = st.selectbox("買い方", ["均等", "本命寄り", "バランス"])

    st.markdown("### 回収率シミュレーター")
    sim_rounds = st.slider("直近何回で試すか", 10, 100, 30, 5)

    st.markdown("### 重み")
    w_freq = st.slider("頻度", 0.0, 2.0, 0.8, 0.1)
    w_recent = st.slider("直近頻度", 0.0, 2.0, 0.8, 0.1)
    w_overdue = st.slider("未出現", 0.0, 2.0, 1.0, 0.1)
    w_streak_avoid = st.slider("連続出現回避", 0.0, 2.0, 0.7, 0.1)
    w_sum = st.slider("合計値", 0.0, 2.0, 0.8, 0.1)
    w_odd_even = st.slider("奇数偶数", 0.0, 2.0, 0.7, 0.1)
    w_recent_avoid = st.slider("直近完全重複回避", 0.0, 2.0, 1.0, 0.1)

    if PRODUCTS[product_name]["kind"] == "numbers":
        w_high_low = st.slider("高低バランス", 0.0, 2.0, 0.6, 0.1)
        w_pattern = st.slider("重複パターン", 0.0, 2.0, 0.5, 0.1)
        top_k_digits = st.slider("各桁の候補数", 2, 8, 4)
        n_candidates = st.slider("内部候補数", 50, 2000, 300, 50)
    else:
        w_pair = st.slider("ペア相性", 0.0, 2.0, 1.0, 0.1)
        w_consecutive = st.slider("連番傾向", 0.0, 2.0, 0.5, 0.1)
        w_spread = st.slider("広がり", 0.0, 2.0, 0.4, 0.1)
        w_recent_overlap_avoid = st.slider("直近重複抑制", 0.0, 2.0, 0.7, 0.1)
        pool_size = st.slider("候補数字プール", 8, 25, 16)
        n_candidates = st.slider("内部候補数", 50, 5000, 500, 50)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["予想", "自動買い目", "回収率シミュレーター", "分析", "過去データ"])

if st.button("予想開始", type="primary"):
    meta = PRODUCTS[product_name]
    try:
        with st.spinner("データ取得中..."):
            if meta["kind"] == "numbers":
                df = scrape_numbers(product_name, digits=meta["digits"], recent_only=recent_only)
            else:
                df = scrape_loto(product_name, count=meta["count"], recent_only=recent_only)

        if df.empty:
            st.error(f"{product_name} のデータ取得に失敗しました。")
            st.stop()

        if meta["kind"] == "numbers":
            cfg = {
                "w_freq": w_freq, "w_recent": w_recent, "w_overdue": w_overdue,
                "w_streak_avoid": w_streak_avoid, "w_sum": w_sum, "w_odd_even": w_odd_even,
                "w_high_low": w_high_low, "w_pattern": w_pattern, "w_recent_avoid": w_recent_avoid,
                "top_k_digits": int(top_k_digits), "n_candidates": int(n_candidates), "seed": int(seed),
            }
            scored, pos_summary = score_numbers_candidates(df, digits=meta["digits"], config=cfg)
            pred = scored.head(top_n).copy()

            with tab1:
                st.subheader("予想結果")
                st.dataframe(pred[["印","candidate","AIスコア","digit_sum","odd_cnt","high_cnt","unique_cnt","overdue_raw",
                                   "頻度寄与","直近頻度寄与","未出現寄与","連続回避寄与","合計寄与","奇数偶数寄与","高低寄与","重複パターン寄与","直近重複回避寄与"]]
                             .rename(columns={"candidate":"予想","digit_sum":"合計","odd_cnt":"奇数数","high_cnt":"高数字数","unique_cnt":"ユニーク数","overdue_raw":"平均未出現"}),
                             use_container_width=True)

            with tab2:
                plan = build_buy_plan(pred, product_name, point_limit, buy_strategy)
                st.dataframe(plan[["印","candidate","AIスコア","購入口数","1口金額","購入金額"]].rename(columns={"candidate":"予想"}), use_container_width=True)
                st.metric("合計購入金額", int(plan["購入金額"].sum()) if not plan.empty else 0)

            with tab3:
                sim = simulate_roi_numbers(df, pred, product_name, sim_rounds, point_limit)
                if not sim.empty:
                    total_cost = int(sim["購入額"].sum())
                    total_return = int(sim["払戻想定"].sum())
                    roi = (total_return / total_cost * 100) if total_cost else 0
                    c1, c2, c3 = st.columns(3)
                    c1.metric("総購入額", total_cost)
                    c2.metric("総払戻想定", total_return)
                    c3.metric("回収率", f"{roi:.2f}%")
                    st.dataframe(sim, use_container_width=True)

            with tab4:
                st.dataframe(pos_summary.sort_values(["position","freq_all"], ascending=[True,False]), use_container_width=True)

            with tab5:
                st.dataframe(df.head(100), use_container_width=True)

            excel_bytes = to_excel_bytes(df, pred.rename(columns={"candidate":"予想"}), {"桁別統計": pos_summary})

        else:
            cfg = {
                "w_freq": w_freq, "w_recent": w_recent, "w_overdue": w_overdue,
                "w_streak_avoid": w_streak_avoid, "w_pair": w_pair, "w_sum": w_sum,
                "w_odd_even": w_odd_even, "w_consecutive": w_consecutive, "w_spread": w_spread,
                "w_recent_overlap_avoid": w_recent_overlap_avoid, "w_recent_avoid": w_recent_avoid,
                "pool_size": int(pool_size), "n_candidates": int(n_candidates), "seed": int(seed),
            }
            scored, number_stats, pair_df, consecutive_dist = score_loto_candidates(df, meta["count"], meta["max_num"], cfg)
            pred = scored.head(top_n).copy()

            with tab1:
                st.subheader("予想結果")
                st.dataframe(pred[["印","candidate","AIスコア","sum_value","odd_cnt","consecutive_cnt","pair_avg_raw","overdue_raw",
                                   "頻度寄与","直近頻度寄与","未出現寄与","連続回避寄与","ペア寄与","合計寄与","奇数偶数寄与","連番寄与","広がり寄与","直近重複抑制寄与","直近完全重複回避寄与"]]
                             .rename(columns={"candidate":"予想","sum_value":"合計","odd_cnt":"奇数数","consecutive_cnt":"連番数","pair_avg_raw":"ペア平均","overdue_raw":"平均未出現"}),
                             use_container_width=True)

            with tab2:
                plan = build_buy_plan(pred, product_name, point_limit, buy_strategy)
                st.dataframe(plan[["印","candidate","AIスコア","購入口数","1口金額","購入金額"]].rename(columns={"candidate":"予想"}), use_container_width=True)
                st.metric("合計購入金額", int(plan["購入金額"].sum()) if not plan.empty else 0)

            with tab3:
                sim = simulate_roi_loto(df, pred, product_name, sim_rounds, point_limit)
                if not sim.empty:
                    total_cost = int(sim["購入額"].sum())
                    total_return = int(sim["払戻想定"].sum())
                    roi = (total_return / total_cost * 100) if total_cost else 0
                    c1, c2, c3 = st.columns(3)
                    c1.metric("総購入額", total_cost)
                    c2.metric("総払戻想定", total_return)
                    c3.metric("回収率", f"{roi:.2f}%")
                    st.dataframe(sim, use_container_width=True)

            with tab4:
                st.subheader("連番出現分布")
                st.dataframe(consecutive_dist, use_container_width=True)
                if not consecutive_dist.empty:
                    st.bar_chart(consecutive_dist.set_index("連番数")[["出現回数"]])
                st.subheader("数字別統計")
                st.dataframe(number_stats, use_container_width=True)
                st.subheader("強いペア")
                st.dataframe(pair_df.head(100), use_container_width=True)

            with tab5:
                st.dataframe(df.head(100), use_container_width=True)

            excel_bytes = to_excel_bytes(df, pred.rename(columns={"candidate":"予想"}), {"数字別統計": number_stats, "ペア統計": pair_df.head(300), "連番分布": consecutive_dist})

        st.download_button(
            "Excelをダウンロード",
            data=excel_bytes,
            file_name=f"{product_name.lower().replace(' ', '_')}_ai_prediction_pro_v2.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"処理に失敗しました: {e}")

st.markdown("---")
st.caption("MINI LOTO は /backnumber/mini/ と /backnumber/miniloto_detail/ の両方に対応するよう調整済みです。")
