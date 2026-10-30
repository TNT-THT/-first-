#!/usr/bin/env python3
"""Build the GitHub Pages Chinese research index from explicitly configured RSS/Atom feeds.

This is intentionally a GitHub Actions batch job, not a runtime proxy or scraper:
- the browser only reads web/public/data/china-sources.json from the same Pages site;
- no user query is sent to the original feed sites;
- only feed summaries and links are retained, with bounded lengths;
- a failed feed keeps its last successful entries rather than replacing the index with emptiness.

Usage:
  python3 scripts/update_china_sources.py \
    --config data/china-feeds.json \
    --output web/public/data/china-sources.json
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024
MAX_TITLE_CHARS = 180
MAX_SNIPPET_CHARS = 800
MAX_URL_CHARS = 2048
USER_AGENT = "OJ-AI-Assistant-GitHub-Index/1.0 (+https://github.com/TNT-THT/-first-)"


class _TextExtractor(HTMLParser):
    """Small, dependency-free HTML-to-text helper for RSS descriptions."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def clean_text(value: Any, limit: int) -> str:
    raw = html.unescape(str(value or ""))
    parser = _TextExtractor()
    try:
        parser.feed(raw)
        parser.close()
        raw = " ".join(parser.parts)
    except Exception:
        # A malformed description is not worth failing the whole source.
        raw = re.sub(r"<[^>]*>", " ", raw)
    raw = re.sub(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    # HTML block boundaries may add a space before punctuation; keep CJK and ordinary prose readable.
    raw = re.sub(r"\s+([,.;:!?，。；：！？、】【）])", r"\1", raw)
    raw = re.sub(r"([（【])\s+", r"\1", raw)
    if len(raw) <= limit:
        return raw
    return f"{raw[: max(0, limit - 1)]}…"


def valid_https_url(value: Any) -> str:
    url = str(value or "").strip()
    if not url or len(url) > MAX_URL_CHARS:
        return ""
    try:
        parsed = urlparse(url)
    except Exception:
        return ""
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        return ""
    return url


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def child_text(element: ET.Element, names: set[str]) -> str:
    for child in list(element):
        if local_name(child.tag) in names:
            value = "".join(child.itertext()).strip()
            if value:
                return value
    return ""


def entry_link(element: ET.Element) -> str:
    fallback = ""
    for child in list(element):
        if local_name(child.tag) != "link":
            continue
        link = child.attrib.get("href", "").strip() or "".join(child.itertext()).strip()
        if not link:
            continue
        rel = child.attrib.get("rel", "alternate").lower()
        if rel in {"", "alternate"}:
            return link
        if not fallback:
            fallback = link
    return fallback


def normalise_date(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    try:
        date = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        try:
            date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return ""
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    date = date.astimezone(timezone.utc)
    if not 1990 <= date.year <= 2200:
        return ""
    return date.isoformat().replace("+00:00", "Z")


def parse_feed(xml_text: str, source: dict[str, Any], max_items: int) -> list[dict[str, str]]:
    # 不接受 DTD / entity，避免第三方 XML 利用实体展开消耗 GitHub Actions 资源。
    if "<!DOCTYPE" in xml_text.upper() or "<!ENTITY" in xml_text.upper():
        raise ValueError("Feed 含不允许的 DTD/entity")
    root = ET.fromstring(xml_text)
    entries = [element for element in root.iter() if local_name(element.tag) in {"item", "entry"}]
    out: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for entry in entries:
        if len(out) >= max_items:
            break
        title = clean_text(child_text(entry, {"title"}), MAX_TITLE_CHARS)
        url = valid_https_url(entry_link(entry))
        if not title or not url or url in seen_urls:
            continue
        seen_urls.add(url)
        summary = child_text(entry, {"description", "summary", "content", "encoded", "subtitle"})
        published = child_text(entry, {"pubdate", "published", "updated", "date", "issued"})
        published_at = normalise_date(published)
        stable_id = hashlib.sha256(f"{source['id']}\0{url}".encode("utf-8")).hexdigest()[:24]
        out.append(
            {
                "id": stable_id,
                "source": source["id"],
                "sourceName": source["label"],
                "category": source.get("category", ""),
                "title": title,
                "url": url,
                "snippet": clean_text(summary, MAX_SNIPPET_CHARS),
                **({"publishedAt": published_at} if published_at else {}),
            }
        )
    return out


class _NoRedirect(HTTPRedirectHandler):
    """A feed may not bounce the Actions runner to an arbitrary network location."""

    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def fetch_feed(url: str, timeout: int) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/atom+xml, application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.1",
        },
    )
    # Redirects are intentionally disabled. The manifest must name the canonical HTTPS feed,
    # so a compromised upstream cannot turn this scheduled job into a request relay.
    opener = build_opener(_NoRedirect())
    with opener.open(request, timeout=timeout) as response:  # nosec B310: URLs are repository-reviewed HTTPS feed URLs
        final_url = valid_https_url(response.geturl())
        if final_url != url:
            raise ValueError("Feed 地址发生未允许的跳转")
        content = response.read(MAX_DOWNLOAD_BYTES + 1)
        if len(content) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Feed 超过 2 MB 上限")
        charset = response.headers.get_content_charset() or "utf-8"
        return content.decode(charset, errors="replace")


def load_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return fallback


def source_metadata(source: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "id": source["id"],
        "label": source["label"],
        "category": source.get("category", ""),
        "homepage": source.get("homepage", ""),
        "feed": source["feed"],
        **extra,
    }


def sort_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item.get("publishedAt", "")), str(item.get("title", "")))


def build_index(config: dict[str, Any], previous: dict[str, Any], timeout: int) -> tuple[dict[str, Any], list[str]]:
    if config.get("schemaVersion") != 1:
        raise ValueError("不支持的 feed manifest schemaVersion")
    max_per_source = int(config.get("maxItemsPerSource", 30))
    max_per_source = min(60, max(1, max_per_source))
    configured = config.get("sources")
    if not isinstance(configured, list):
        raise ValueError("sources 必须是数组")

    previous_items = previous.get("items") if isinstance(previous.get("items"), list) else []
    previous_by_source: dict[str, list[dict[str, Any]]] = {}
    for item in previous_items:
        if isinstance(item, dict) and isinstance(item.get("source"), str):
            previous_by_source.setdefault(item["source"], []).append(item)
    previous_sources = previous.get("sources") if isinstance(previous.get("sources"), list) else []
    previous_source_by_id = {
        row.get("id"): row for row in previous_sources if isinstance(row, dict) and isinstance(row.get("id"), str)
    }

    source_rows: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    notices: list[str] = []
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    ids: set[str] = set()
    configured_ids: set[str] = set()

    for raw_source in configured:
        if not isinstance(raw_source, dict) or raw_source.get("enabled", True) is False:
            continue
        source_id = str(raw_source.get("id", "")).strip()
        label = clean_text(raw_source.get("label", ""), 60)
        feed = valid_https_url(raw_source.get("feed"))
        homepage = valid_https_url(raw_source.get("homepage"))
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", source_id) or not label or not feed:
            raise ValueError("每个启用来源都需要安全的 id、label 和 HTTPS feed")
        if source_id in configured_ids:
            raise ValueError(f"来源 id 重复：{source_id}")
        configured_ids.add(source_id)
        source = {
            "id": source_id,
            "label": label,
            "category": clean_text(raw_source.get("category", ""), 40),
            "homepage": homepage,
            "feed": feed,
        }
        try:
            parsed = parse_feed(fetch_feed(feed, timeout), source, max_per_source)
            if not parsed:
                raise ValueError("Feed 没有可用条目")
            source_rows.append(source_metadata(source, status="ok", fetchedAt=now, itemCount=len(parsed)))
            items.extend(parsed)
            notices.append(f"{source_id}: {len(parsed)} 条")
        except Exception as exc:  # feed failures should not erase the last useful snapshot
            cached = [item for item in previous_by_source.get(source_id, []) if isinstance(item, dict)]
            cached = cached[:max_per_source]
            previous_row = previous_source_by_id.get(source_id, {})
            last_success = previous_row.get("fetchedAt") if isinstance(previous_row, dict) else None
            source_rows.append(
                source_metadata(
                    source,
                    status="stale" if cached else "error",
                    fetchedAt=last_success if isinstance(last_success, str) else None,
                    itemCount=len(cached),
                    error=clean_text(str(exc), 160),
                )
            )
            items.extend(cached)
            notices.append(f"{source_id}: 失败，保留 {len(cached)} 条缓存")

    deduped: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for item in sorted(items, key=sort_key, reverse=True):
        url = valid_https_url(item.get("url"))
        item_id = str(item.get("id", ""))
        if not url or not item_id or url in seen_urls or item_id in ids:
            continue
        seen_urls.add(url)
        ids.add(item_id)
        deduped.append(item)

    return (
        {
            "schemaVersion": 1,
            "generatedAt": now,
            "generator": "GitHub Actions RSS snapshot; this file contains no user query data.",
            "sources": source_rows,
            "items": deduped,
        },
        notices,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="data/china-feeds.json", type=Path)
    parser.add_argument("--output", default="web/public/data/china-sources.json", type=Path)
    parser.add_argument("--timeout", default=15, type=int)
    args = parser.parse_args()

    config = load_json(args.config, None)
    if not isinstance(config, dict):
        print(f"配置无效：{args.config}", file=sys.stderr)
        return 2
    previous = load_json(args.output, {})
    if not isinstance(previous, dict):
        previous = {}
    try:
        index, notices = build_index(config, previous, max(3, min(60, args.timeout)))
    except (ValueError, TypeError) as exc:
        print(f"无法生成中国资料索引：{exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("中国资料索引已更新：" + "；".join(notices))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
