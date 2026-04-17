"""
MoSPI website PDF scraper.

Enumerates publications/reports/public-documents from the undocumented
`https://www.mospi.gov.in/api/` (powering the React SPA) and downloads
every linked PDF to data/raw/. Resume-safe: skips files already present
in the manifest.

Usage:
    python -m src.scrape_mospi enumerate           # phase 1: write manifest
    python -m src.scrape_mospi download            # phase 2: fetch PDFs
    python -m src.scrape_mospi all                 # both phases
    python -m src.scrape_mospi enumerate --source web --max-pages 5   # smoke test

The two phases are split so you can inspect the manifest (and decide what
to actually download) before pulling several GB.
"""

from __future__ import annotations

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional
from urllib.parse import urlparse

import requests
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

API_BASE = "https://www.mospi.gov.in/api/"
SITE_BASE = "https://www.mospi.gov.in/"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.mospi.gov.in",
    "Referer": "https://www.mospi.gov.in/publications-reports",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
}

DEFAULT_PAGE_SIZE = 50
DEFAULT_RATE_LIMIT_S = 0.4
DEFAULT_DOWNLOAD_WORKERS = 4
DEFAULT_TIMEOUT = 60

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()
log = logging.getLogger("mospi_scrape")


@dataclass
class FileRef:
    source: str          # "web" | "archival" | "public-doc"
    parent_id: str       # publication_id or doc id
    parent_title: str
    chapter_id: str | None
    chapter_title: str | None
    file_slot: str       # "file_one" | "file_two" | "file_three"
    filename: str
    filemime: str
    filesize: int | None
    url: str             # absolute, ready to download
    published_date: str | None


@dataclass
class Manifest:
    files: list[FileRef] = field(default_factory=list)

    def to_jsonl(self, path: Path) -> None:
        with path.open("w") as f:
            for ref in self.files:
                f.write(json.dumps(asdict(ref)) + "\n")

    @classmethod
    def from_jsonl(cls, path: Path) -> "Manifest":
        m = cls()
        if not path.exists():
            return m
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    m.files.append(FileRef(**json.loads(line)))
        return m


def _resolve_url(path: str) -> str:
    if path.startswith(("http://", "https://")):
        return path
    return SITE_BASE + path.lstrip("/")


class NotFound(Exception):
    pass


def _api_post(
    session: requests.Session,
    endpoint: str,
    body: dict,
    rate_limit: float,
    max_retries: int = 4,
) -> dict:
    url = API_BASE + endpoint
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = session.post(url, json=body, timeout=DEFAULT_TIMEOUT)
            if r.status_code == 200:
                time.sleep(rate_limit + random.uniform(0, 0.2))
                return r.json()
            if r.status_code == 404:
                raise NotFound(endpoint)
            if r.status_code in (429, 502, 503, 504):
                log.warning("api %s -> %s, retrying in %.1fs", endpoint, r.status_code, delay)
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
        except NotFound:
            raise
        except requests.RequestException as e:
            last_err = e
            log.warning("api %s attempt %d failed: %s", endpoint, attempt + 1, e)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"api {endpoint} failed after {max_retries} attempts: {last_err}")


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    return s


def _files_from_record(
    rec: dict,
    source: str,
    parent_id: str,
    parent_title: str,
    chapter_id: str | None,
    chapter_title: str | None,
    published_date: str | None,
) -> Iterator[FileRef]:
    for slot in ("file_one", "file_two", "file_three"):
        f = rec.get(slot)
        if not f or not isinstance(f, dict):
            continue
        path = f.get("path")
        if not path:
            continue
        yield FileRef(
            source=source,
            parent_id=parent_id,
            parent_title=parent_title,
            chapter_id=chapter_id,
            chapter_title=chapter_title,
            file_slot=slot,
            filename=f.get("filename") or Path(urlparse(path).path).name,
            filemime=f.get("filemime") or "",
            filesize=f.get("filesize"),
            url=_resolve_url(path),
            published_date=published_date,
        )


def _enumerate_publications(
    session: requests.Session,
    data_source: str,
    page_size: int,
    rate_limit: float,
    max_pages: int | None,
) -> Iterator[FileRef]:
    """
    For data_source in {'web', 'archival'}: paginate the parent-publication
    list, then fan out per publication_id to chapter-data, yielding every PDF.
    """
    page = 1
    while True:
        body = {
            "lang": "en",
            "page_no": page,
            "page_size": page_size,
            "sort_by": "published_date",
            "sort_order": "DESC",
            "data_source": data_source,
        }
        resp = _api_post(
            session,
            "publications-reports/get-web-publications-report-list",
            body,
            rate_limit,
        )
        records = resp.get("data") or []
        if not records:
            break

        for rec in records:
            pub_id = str(rec.get("id"))
            pub_title = (rec.get("title") or "").strip()
            published = rec.get("published_year") or rec.get("published_date")

            # Some parent records embed a file directly (no chapters).
            yield from _files_from_record(
                rec, data_source, pub_id, pub_title,
                chapter_id=None, chapter_title=None, published_date=published,
            )

            yield from _enumerate_chapters(
                session, pub_id, pub_title, published, data_source, page_size, rate_limit
            )

        pagination = resp.get("pagination") or {}
        total_pages = pagination.get("totalPages") or 1
        if page >= total_pages:
            break
        if max_pages and page >= max_pages:
            break
        page += 1


def _enumerate_chapters(
    session: requests.Session,
    publication_id: str,
    parent_title: str,
    published_date: str | None,
    source: str,
    page_size: int,
    rate_limit: float,
) -> Iterator[FileRef]:
    page = 1
    while True:
        body = {
            "publication_id": publication_id,
            "lang": "en",
            "page_no": page,
            "page_size": page_size,
        }
        try:
            resp = _api_post(
                session,
                "publications-reports/get-web-chapter-data",
                body,
                rate_limit,
            )
        except NotFound:
            return  # most parents have no chapters; expected
        except RuntimeError as e:
            log.warning("chapter fetch failed for pub %s: %s", publication_id, e)
            return

        chapters = resp.get("data") or []
        if not chapters:
            break

        for ch in chapters:
            ch_id = str(ch.get("chapter_id") or "")
            ch_title = (ch.get("chapter_title") or "").strip()
            ch_pub = ch.get("published_date") or published_date
            yield from _files_from_record(
                ch, source, publication_id, parent_title,
                chapter_id=ch_id, chapter_title=ch_title, published_date=ch_pub,
            )
            for sub in ch.get("sub_chapters") or []:
                sub_id = str(sub.get("chapter_id") or "")
                sub_title = (sub.get("chapter_title") or "").strip()
                yield from _files_from_record(
                    sub, source, publication_id, parent_title,
                    chapter_id=f"{ch_id}.{sub_id}",
                    chapter_title=f"{ch_title} / {sub_title}",
                    published_date=ch_pub,
                )

        pagination = resp.get("pagination") or {}
        total_pages = pagination.get("totalPages") or 1
        if page >= total_pages:
            break
        page += 1


def _enumerate_flat_endpoint(
    session: requests.Session,
    endpoint: str,
    source_label: str,
    page_size: int,
    rate_limit: float,
    max_pages: int | None,
    extra_body: dict | None = None,
) -> Iterator[FileRef]:
    """Generic paginator for the endpoints that return a flat list of records
    with file_one/two/three (no chapter-fanout): public-doc, acts-and-policies,
    latest-release."""
    page = 1
    while True:
        body = {
            "lang": "en",
            "page_no": page,
            "page_size": page_size,
            "sort_by": "published_date",
            "sort_order": "DESC",
            **(extra_body or {}),
        }
        resp = _api_post(session, endpoint, body, rate_limit)
        records = resp.get("data") or []
        if not records:
            break
        for rec in records:
            doc_id = str(rec.get("id"))
            title = (rec.get("title_en") or rec.get("title") or "").strip()
            published = rec.get("published_year") or rec.get("published_date")
            yield from _files_from_record(
                rec, source_label, doc_id, title,
                chapter_id=None, chapter_title=None, published_date=published,
            )
        pagination = resp.get("pagination") or {}
        total_pages = pagination.get("totalPages") or 1
        if page >= total_pages or (max_pages and page >= max_pages):
            break
        page += 1


def _enumerate_public_docs(session, page_size, rate_limit, max_pages):
    return _enumerate_flat_endpoint(
        session, "public-doc/get-web-pub-doc-list", "public-doc",
        page_size, rate_limit, max_pages, extra_body={"data_source": "web"},
    )


def _enumerate_acts_policies(session, page_size, rate_limit, max_pages):
    return _enumerate_flat_endpoint(
        session, "acts-and-policies/get-web-acts-policies-list", "acts-policies",
        page_size, rate_limit, max_pages,
    )


def _enumerate_press_release(session, page_size, rate_limit, max_pages):
    return _enumerate_flat_endpoint(
        session, "latest-release/get-web-latest-release-list", "press-release",
        page_size, rate_limit, max_pages,
    )


def _safe_local_name(ref: FileRef) -> Path:
    """Make a stable, collision-resistant local path under data/raw/<source>/."""
    sub = ref.source
    parent = ref.parent_id or "unknown"
    chapter = ref.chapter_id or "0"
    base = Path(ref.filename).name.replace("/", "_")
    return Path(sub) / parent / f"{chapter}__{base}"


def _download_one(
    session: requests.Session,
    ref: FileRef,
    out_root: Path,
    rate_limit: float,
) -> tuple[FileRef, bool, str]:
    dst = out_root / _safe_local_name(ref)
    if dst.exists() and dst.stat().st_size > 0:
        return ref, True, "skip-exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with session.get(ref.url, stream=True, timeout=DEFAULT_TIMEOUT) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
        tmp.rename(dst)
        time.sleep(rate_limit + random.uniform(0, 0.2))
        return ref, True, "ok"
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return ref, False, f"error: {e}"


ALL_SOURCES = ("web", "archival", "public-doc", "acts-policies", "press-release")


@app.command()
def enumerate(
    out: Path = typer.Option(Path("data/raw/manifest.jsonl"), help="Manifest output path."),
    source: str = typer.Option(
        "all",
        help="Comma-separated subset, or 'all'. Sources: web | archival | public-doc | acts-policies | press-release",
    ),
    page_size: int = typer.Option(DEFAULT_PAGE_SIZE),
    rate_limit_s: float = typer.Option(DEFAULT_RATE_LIMIT_S),
    max_pages: Optional[int] = typer.Option(None, help="Stop after this many parent-list pages (smoke test)."),
    per_source_records: Optional[int] = typer.Option(None, help="Stop each source after this many file refs (smoke test)."),
) -> None:
    """Phase 1: enumerate every PDF reachable via the API and write the manifest.

    Sources map to the four Documents-menu categories on mospi.gov.in:
      - web + archival : Publications/Reports (live + archived)
      - public-doc     : Other Documents
      - acts-policies  : Acts & Policies (note: most entries are external links, few host PDFs)
      - press-release  : Press Release
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out.parent.mkdir(parents=True, exist_ok=True)

    session = _make_session()
    sources = set(ALL_SOURCES) if source == "all" else {s.strip() for s in source.split(",")}
    bad = sources - set(ALL_SOURCES)
    if bad:
        raise typer.BadParameter(f"unknown source(s): {sorted(bad)}; valid: {ALL_SOURCES}")

    manifest = Manifest()
    named_streams: list[tuple[str, Iterable[FileRef]]] = []
    if "web" in sources:
        named_streams.append(("web", _enumerate_publications(session, "web", page_size, rate_limit_s, max_pages)))
    if "archival" in sources:
        named_streams.append(("archival", _enumerate_publications(session, "archival", page_size, rate_limit_s, max_pages)))
    if "public-doc" in sources:
        named_streams.append(("public-doc", _enumerate_public_docs(session, page_size, rate_limit_s, max_pages)))
    if "acts-policies" in sources:
        named_streams.append(("acts-policies", _enumerate_acts_policies(session, page_size, rate_limit_s, max_pages)))
    if "press-release" in sources:
        named_streams.append(("press-release", _enumerate_press_release(session, page_size, rate_limit_s, max_pages)))

    seen_urls: set[str] = set()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.completed} files"),
        console=console,
    ) as progress:
        task = progress.add_task("Enumerating", total=None)
        for name, stream in named_streams:
            count_this_source = 0
            for ref in stream:
                if ref.url in seen_urls:
                    continue
                seen_urls.add(ref.url)
                manifest.files.append(ref)
                count_this_source += 1
                progress.update(task, advance=1)
                if per_source_records and count_this_source >= per_source_records:
                    log.info("hit per-source-records=%d for %s, moving on", per_source_records, name)
                    break

    manifest.to_jsonl(out)
    pdf_count = sum(1 for r in manifest.files if r.filemime == "application/pdf")
    console.print(
        f"[green]Wrote {len(manifest.files)} files ({pdf_count} PDFs) to {out}[/green]"
    )


@app.command()
def download(
    manifest: Path = typer.Option(Path("data/raw/manifest.jsonl")),
    out_dir: Path = typer.Option(Path("data/raw")),
    workers: int = typer.Option(DEFAULT_DOWNLOAD_WORKERS),
    rate_limit_s: float = typer.Option(DEFAULT_RATE_LIMIT_S),
    only_pdf: bool = typer.Option(True, help="Skip non-PDF files (xlsx, etc)."),
    limit: Optional[int] = typer.Option(None, help="Cap total downloads (smoke test)."),
    per_source_limit: Optional[int] = typer.Option(
        None, help="Cap downloads per source (web/archival/public-doc). Useful for balanced smoke tests."
    ),
) -> None:
    """Phase 2: download files from the manifest. Resume-safe."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    m = Manifest.from_jsonl(manifest)
    if not m.files:
        raise typer.Exit(f"manifest {manifest} is empty; run `enumerate` first")

    todo = [r for r in m.files if (not only_pdf or r.filemime == "application/pdf")]
    if per_source_limit:
        per: dict[str, int] = {}
        capped: list[FileRef] = []
        for r in todo:
            n = per.get(r.source, 0)
            if n < per_source_limit:
                capped.append(r)
                per[r.source] = n + 1
        todo = capped
        log.info("per-source caps applied: %s", per)
    if limit:
        todo = todo[:limit]

    session = _make_session()
    ok = 0
    fail = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading", total=len(todo))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_download_one, session, r, out_dir, rate_limit_s) for r in todo]
            for fut in as_completed(futures):
                ref, success, status = fut.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                    log.warning("FAIL %s -> %s", ref.url, status)
                progress.update(task, advance=1)

    console.print(f"[green]done: {ok} ok, {fail} failed[/green]")


@app.command(name="all")
def all_cmd(
    out_dir: Path = typer.Option(Path("data/raw")),
    page_size: int = typer.Option(DEFAULT_PAGE_SIZE),
    workers: int = typer.Option(DEFAULT_DOWNLOAD_WORKERS),
    rate_limit_s: float = typer.Option(DEFAULT_RATE_LIMIT_S),
) -> None:
    """Run enumerate + download back to back."""
    manifest_path = out_dir / "manifest.jsonl"
    enumerate(
        out=manifest_path,
        source="all",
        page_size=page_size,
        rate_limit_s=rate_limit_s,
        max_pages=None,
    )
    download(
        manifest=manifest_path,
        out_dir=out_dir,
        workers=workers,
        rate_limit_s=rate_limit_s,
        only_pdf=True,
        limit=None,
    )


if __name__ == "__main__":
    app()
