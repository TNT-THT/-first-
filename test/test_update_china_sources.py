#!/usr/bin/env python3
"""Regression tests for the dependency-free GitHub Actions RSS/Atom indexer."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "update_china_sources.py"
spec = importlib.util.spec_from_file_location("china_indexer", SCRIPT)
assert spec and spec.loader
indexer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indexer)


class ChinaIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {"id": "demo", "label": "示例发布方", "category": "技术社区"}

    def test_checked_in_public_schema_matches_enabled_manifest(self) -> None:
        manifest = json.loads((ROOT / "data" / "china-feeds.json").read_text(encoding="utf-8"))
        published = json.loads((ROOT / "web" / "public" / "data" / "china-sources.json").read_text(encoding="utf-8"))
        expected = [
            (source["id"], source["label"], source.get("category", ""), source["feed"])
            for source in manifest["sources"]
            if source.get("enabled", True)
        ]
        actual = [
            (source["id"], source["label"], source.get("category", ""), source["feed"])
            for source in published["sources"]
        ]
        self.assertEqual(manifest["schemaVersion"], 1)
        self.assertEqual(published["schemaVersion"], 1)
        self.assertEqual(actual, expected)

    def test_parse_atom_sanitises_summary_and_rejects_non_https_link(self) -> None:
        xml = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <title>二分查找 &amp; 复杂度</title>
            <link href="https://example.com/valid" />
            <updated>2026-10-30T00:00:00Z</updated>
            <summary><![CDATA[<p>时间复杂度为 <b>O(log n)</b>。</p>]]></summary>
          </entry>
          <entry><title>不安全</title><link href="javascript:alert(1)" /></entry>
        </feed>"""
        items = indexer.parse_feed(xml, self.source, 10)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "二分查找 & 复杂度")
        self.assertEqual(items[0]["snippet"], "时间复杂度为 O(log n)。")
        self.assertEqual(items[0]["publishedAt"], "2026-10-30T00:00:00Z")

    def test_rejects_dtd_before_parsing(self) -> None:
        with self.assertRaises(ValueError):
            indexer.parse_feed(
                "<!DOCTYPE rss [<!ENTITY x 'expanded'>]><rss><channel><item><title>&x;</title></item></channel></rss>",
                self.source,
                10,
            )

    def test_failed_refresh_keeps_previous_source_items_as_stale(self) -> None:
        previous = {
            "sources": [{"id": "demo", "status": "ok", "fetchedAt": "2026-10-29T00:00:00Z"}],
            "items": [
                {
                    "id": "old-entry",
                    "source": "demo",
                    "sourceName": "示例发布方",
                    "category": "技术社区",
                    "title": "上一条资料",
                    "url": "https://example.com/old",
                    "snippet": "旧快照",
                }
            ]
        }
        config = {
            "schemaVersion": 1,
            "maxItemsPerSource": 5,
            "sources": [
                {
                    **self.source,
                    "homepage": "https://example.com/",
                    "feed": "https://example.com/feed.xml",
                    "enabled": True,
                }
            ],
        }
        original_fetch = indexer.fetch_feed
        try:
            indexer.fetch_feed = lambda *_args: (_ for _ in ()).throw(ValueError("offline"))
            generated, _notices = indexer.build_index(config, previous, timeout=5)
        finally:
            indexer.fetch_feed = original_fetch
        self.assertEqual(generated["sources"][0]["status"], "stale")
        self.assertEqual(generated["sources"][0]["fetchedAt"], "2026-10-29T00:00:00Z")
        self.assertEqual(generated["sources"][0]["itemCount"], 1)
        self.assertEqual(generated["items"][0]["url"], "https://example.com/old")


if __name__ == "__main__":
    unittest.main()
