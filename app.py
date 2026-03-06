# /// script
# dependencies = ["fastapi", "uvicorn", "python-multipart","google-genai","pydantic", "requests", "Pillow"]
# ///

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sympy import im
import uvicorn
import pathlib
import os
import io
import time
import random
import json
import requests
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from plan_Creation import *
from plan_execution import execute_plan_v1
import pandas as pd
import pdfplumber
import re
import time

from bs4 import BeautifulSoup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# FRONT END API CHECKS

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request 
templates = Jinja2Templates(directory="templates")

@app.get("/ui", response_class=HTMLResponse)
async def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
from dotenv import load_dotenv

# Load environment variables from .env file (override system variables)
load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# GEMINI_API_KEY = "AIzaSyDh7TfjKwBEI2eoE4xObDfyBbRh25YGe8k"
client = genai.Client(api_key=GEMINI_API_KEY)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _is_image_filename(name: str) -> bool:
    ext = pathlib.Path(name).suffix.lower()
    return ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}

def _is_image_content_type(content_type: str) -> bool:
    return (content_type or "").startswith("image/")
def _is_csv(content_type: str, filename: str) -> bool:
    ext = pathlib.Path(filename).suffix.lower()
    return ext == ".csv" or (content_type or "") in {"text/csv", "application/vnd.ms-excel"}

def _is_json(content_type: str, filename: str) -> bool:
    ext = pathlib.Path(filename).suffix.lower()
    return ext == ".json" or (content_type or "").lower() in {"application/json", "text/json"}

def _is_excel(content_type: str, filename: str) -> bool:
    ext = pathlib.Path(filename).suffix.lower()
    ct = (content_type or "").lower()
    return ext in {".xls", ".xlsx"} or ct in {
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }

def _is_pdf(content_type: str, filename: str) -> bool:
    ext = pathlib.Path(filename).suffix.lower()
    ct = (content_type or "").lower()
    return ext == ".pdf" or ct in {"application/pdf"}


def get_image_description(image_path_or_url: str, max_retries: int = 5) -> str:
    """Returns a concise description for a local image path or URL using Gemini."""
    _client = genai.Client(api_key=GEMINI_API_KEY)

    try:
        if image_path_or_url.startswith(("http://", "https://")):
            resp = requests.get(image_path_or_url, timeout=30)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content))
        else:
            image = Image.open(image_path_or_url)
    except Exception as e:
        return f"Could not open image: {e}"

    for attempt in range(max_retries):
        try:
            response = _client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    "Generate a detailed description of this image. ",
                    "If any numerical data mention that as well",
                    "Give a detailed description and understanding from the image",
                    image
                ]
            )
            text = getattr(response, "text", "").strip()
            return text or "No description returned."
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 1))
            else:
                return f"Description unavailable due to error: {e}"

def get_csv_metadata(file_path: str, sample_rows: int = 1) -> dict:
    """
    Efficiently read minimal data to report CSV schema and a sample row.
    - Tries to infer dtypes from a small chunk.
    - Returns columns, dtypes (as strings), and one sample row (if present).
    """
    try:
        # Read a small chunk to infer schema
        chunk_iter = pd.read_csv(
            file_path,
            nrows=None,            # allow chunksize to control rows
            chunksize=2048,        # small chunk for inference
            low_memory=False,      # better type inference
            dtype_backend="numpy_nullable",  # stable dtypes as strings
            encoding="utf-8",
            on_bad_lines="skip",   # be forgiving
        )
        first_chunk = next(chunk_iter, None)
        if first_chunk is None or first_chunk.empty:
            return {
                "columns": [],
                "dtypes": {},
                "sample_row": {}
            }

        # Normalize column names to str
        first_chunk.columns = [str(c) for c in first_chunk.columns]

        # Build dtype mapping as strings
        dtypes = {col: str(first_chunk.dtypes[col]) for col in first_chunk.columns}

        # Get a small sample (default 1 row)
        sample_df = first_chunk.head(sample_rows)
        sample_row = sample_df.iloc[0].to_dict() if not sample_df.empty else {}

        return {
            "columns": list(first_chunk.columns),
            "dtypes": dtypes,
            "sample_row": sample_row
        }

    except StopIteration:
        return {"columns": [], "dtypes": {}, "sample_row": {}}
    except UnicodeDecodeError:
        # Retry with latin-1 fallback for weird encodings
        try:
            df = pd.read_csv(
                file_path,
                nrows=2048,
                low_memory=False,
                dtype_backend="numpy_nullable",
                encoding="latin-1",
                on_bad_lines="skip",
            )
            df.columns = [str(c) for c in df.columns]
            dtypes = {col: str(df.dtypes[col]) for col in df.columns}
            sample_row = df.head(1).iloc[0].to_dict() if not df.empty else {}
            return {
                "columns": list(df.columns),
                "dtypes": dtypes,
                "sample_row": sample_row
            }
        except Exception:
            return {"columns": [], "dtypes": {}, "sample_row": {}}
    except Exception:
        # Keep it resilient; upstream can still process the file even if metadata fails
        return {"columns": [], "dtypes": {}, "sample_row": {}}

def get_json_metadata(file_path: str, max_preview_bytes: int = 131072) -> dict:
    """
    Read a small portion of a JSON file and summarize structure.
    Handles top-level object or array of objects.
    Returns keys, types, and a sample item.
    """
    try:
        # Fast path: if small enough, load fully; else stream then parse
        if os.path.getsize(file_path) <= max_preview_bytes:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # Stream first N bytes but ensure valid JSON by falling back to full load on failure
            with open(file_path, "r", encoding="utf-8") as f:
                chunk = f.read(max_preview_bytes)
            try:
                data = json.loads(chunk)
            except Exception:
                # As a safe fallback (still bounded by OS limits), try full parse
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

        # Normalize to an object sample
        if isinstance(data, list):
            sample = next((x for x in data if isinstance(x, dict)), {})
        elif isinstance(data, dict):
            sample = data
        else:
            return {"top_level_type": type(data).__name__, "keys": [], "sample_object": {}}

        keys = list(sample.keys())
        dtypes = {k: type(sample.get(k)).__name__ for k in keys}

        return {
            "top_level_type": "array" if isinstance(data, list) else "object",
            "keys": keys,
            "dtypes": dtypes,
            "sample_object": sample,
        }
    except Exception:
        return {"top_level_type": "unknown", "keys": [], "dtypes": {}, "sample_object": {}}

def get_excel_metadata(file_path: str, sample_rows: int = 3) -> dict:
    """
    Probe up to a few sheets to summarize structure:
    - sheet names
    - columns
    - dtypes (as strings)
    - one sample row per sheet (if available)
    """
    try:
        import pandas as pd
        xls = pd.ExcelFile(file_path)
        sheets_meta = []
        for sheet_name in xls.sheet_names[:5]:  # cap for speed
            try:
                df = xls.parse(sheet_name, nrows=sample_rows, dtype_backend="numpy_nullable")
                df.columns = [str(c) for c in df.columns]
                dtypes = {c: str(df.dtypes[c]) for c in df.columns}
                sample = df.head(1).iloc[0].to_dict() if not df.empty else {}
                sheets_meta.append({
                    "name": sheet_name,
                    "columns": list(df.columns),
                    "dtypes": dtypes,
                    "sample_row": sample
                })
            except Exception:
                sheets_meta.append({
                    "name": sheet_name,
                    "columns": [],
                    "dtypes": {},
                    "sample_row": {}
                })
        return {"sheets": sheets_meta}
    except Exception:
        return {"sheets": []}


def get_pdf_metadata(file_path: str, max_pages: int = 5, max_text_chars: int = 4000) -> dict:
    """
    Extract a quick summary from a PDF:
    - page count
    - text preview (first N chars aggregated from first `max_pages`)
    - table summaries: page, heuristic header detection, columns (if header), sample rows

    Table extraction uses pdfplumber's table parser heuristics (no external deps).
    Header detection:
      - If first row on a page looks string-like and repeats across pages, treat as header.
      - Otherwise, no header: we still return the first row as sample to help the planner infer headers.
    """
    try:
        def looks_like_header(row):
            # Heuristic: mostly non-empty strings, not numbers; short-ish cells.
            if not row or not isinstance(row, list):
                return False
            str_like = sum(1 for c in row if isinstance(c, str) and bool(re.search(r"[A-Za-z]", c or "")))
            num_like = sum(1 for c in row if isinstance(c, str) and re.fullmatch(r"[-+]?[\d,.]+", (c or "").strip()))
            avg_len = sum(len((c or "")) for c in row) / max(len(row), 1)
            return (str_like >= max(1, len(row)//2)) and (num_like <= len(row)//3) and (avg_len <= 40)

        text_buf = []
        tables_meta = []
        header_candidates = []

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages[:max_pages]):
                # text preview
                t = (page.extract_text() or "").strip()
                if t:
                    text_buf.append(t)

                # basic table extraction (pdfplumber)
                try:
                    raw_tables = page.extract_tables(
                        table_settings={
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                            "intersection_tolerance": 5,
                        }
                    )
                except Exception:
                    raw_tables = []

                for tbl in raw_tables:
                    # tbl is a list of rows (lists of cells)
                    header_row = tbl[0] if tbl else None
                    has_header = looks_like_header(header_row) if header_row else False
                    if has_header and header_row:
                        header_candidates.append(tuple((c or "").strip() for c in header_row))
                        columns = [str((c or "")).strip() for c in header_row]
                        data_rows = tbl[1:3]  # sample 2 rows
                    else:
                        columns = []
                        data_rows = tbl[:2]

                    # normalize sample rows to dicts when header exists
                    sample_rows = []
                    if columns and data_rows:
                        for r in data_rows:
                            sample_rows.append({columns[j]: (r[j] if j < len(r) else None) for j in range(len(columns))})

                    tables_meta.append({
                        "page_index": i,                    # 0-based
                        "has_header": has_header,
                        "columns": columns,
                        "sample_rows": sample_rows if columns else (data_rows or []),
                        "row_count_estimate": len(tbl) if isinstance(tbl, list) else None,
                    })

        # check for repeated headers across pages
        header_repeat = False
        if header_candidates:
            from collections import Counter
            c = Counter(header_candidates)
            most_common, freq = c.most_common(1)[0]
            header_repeat = freq >= 2  # seen on 2+ tables/pages

        text_preview = "\n".join(text_buf)
        if len(text_preview) > max_text_chars:
            text_preview = text_preview[:max_text_chars] + "…"

        return {
            "page_count": page_count,
            "text_preview": text_preview,
            "tables": tables_meta,
            "headers_repeat_across_pages": header_repeat,
        }
    except Exception:
        return {
            "page_count": None,
            "text_preview": "",
            "tables": [],
            "headers_repeat_across_pages": False,
        }


import re, tempfile
import pandas as pd
import requests
import pdfplumber
from urllib.parse import urlparse

_URL_RE = re.compile(r'https?://[^\s)>\]"\']+', re.I)

def _extract_urls(text: str):
    return list(dict.fromkeys(_URL_RE.findall(text or "")))  # unique, preserve order

def _extract_urls_comprehensive(text: str) -> List[str]:
    """
    Extract all URLs from text using multiple patterns to catch various URL formats.
    Handles http/https, ftp, file, mailto, and other schemes, with or without protocols.
    """
    print("In extract urls")
    if not text:
        return []
    
    urls = []
    
    # Pattern 1: Standard URLs with protocol (http/https/ftp/file/etc.)
    protocol_pattern = re.compile(
        r'\b(?:(?:https?|ftp|ftps|sftp|file|mailto|tel|sms|whatsapp)://[^\s<>"\'`\[\]{}|\\^]+)',
        re.IGNORECASE
    )
    urls.extend(protocol_pattern.findall(text))
    
    # Pattern 2: URLs without protocol (www. domains)
    www_pattern = re.compile(
        r'\b(?:www\.)[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*(?:\.[a-zA-Z]{2,})[^\s<>"\'`\[\]{}|\\^]*',
        re.IGNORECASE
    )
    www_urls = www_pattern.findall(text)
    urls.extend([f"https://{url}" for url in www_urls])
    
    # Pattern 3: Naked domains (domain.com without www)
    naked_domain_pattern = re.compile(
        r'\b[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}(?:/[^\s<>"\'`\[\]{}|\\^]*)?',
        re.IGNORECASE
    )
    potential_domains = naked_domain_pattern.findall(text)
    
    # Filter naked domains to avoid false positives (common file extensions, etc.)
    valid_tlds = {
        'com', 'org', 'net', 'edu', 'gov', 'mil', 'int', 'co', 'io', 'ai', 'app',
        'uk', 'ca', 'au', 'de', 'fr', 'jp', 'cn', 'in', 'br', 'ru', 'it', 'es',
        'nl', 'se', 'no', 'dk', 'fi', 'pl', 'be', 'ch', 'at', 'ie', 'nz', 'za',
        'mx', 'ar', 'cl', 'co', 'pe', 'kr', 'sg', 'my', 'th', 'vn', 'ph', 'id',
        'tw', 'hk', 'ae', 'sa', 'eg', 'tr', 'gr', 'cz', 'hu', 'ro', 'bg', 'hr',
        'lt', 'lv', 'ee', 'si', 'sk', 'is', 'mt', 'cy', 'lu', 'li', 'mc', 'sm',
        'tv', 'me', 'cc', 'ly', 'to', 'ws', 'nu', 'tk', 'ml', 'ga', 'cf'
    }
    
    for domain in potential_domains:
        # Skip if already captured by other patterns
        if any(domain in existing_url for existing_url in urls):
            continue
            
        # Extract TLD and validate
        parts = domain.split('.')
        if len(parts) >= 2:
            tld = parts[-1].lower()
            if tld in valid_tlds:
                # Additional validation: avoid common false positives
                if not any(domain.lower().endswith(f'.{ext}') for ext in [
                    'txt', 'doc', 'docx', 'pdf', 'jpg', 'jpeg', 'png', 'gif', 
                    'mp3', 'mp4', 'zip', 'rar', 'exe', 'dmg'
                ]):
                    urls.append(f"https://{domain}")
    
    # Pattern 4: File URLs and special protocols
    file_pattern = re.compile(
        r'\b(?:file://[^\s<>"\'`\[\]{}|\\^]+)',
        re.IGNORECASE
    )
    urls.extend(file_pattern.findall(text))
    
    # Pattern 5: IP addresses with ports
    ip_pattern = re.compile(
        r'\b(?:https?://)?(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?(?:/[^\s<>"\'`\[\]{}|\\^]*)?',
        re.IGNORECASE
    )
    ip_urls = ip_pattern.findall(text)
    for ip_url in ip_urls:
        if not ip_url.startswith(('http://', 'https://')):
            urls.append(f"http://{ip_url}")
        else:
            urls.append(ip_url)
    
    # Pattern 6: localhost and internal domains
    localhost_pattern = re.compile(
        r'\b(?:https?://)?(?:localhost|127\.0\.0\.1|0\.0\.0\.0|::1)(?::\d{1,5})?(?:/[^\s<>"\'`\[\]{}|\\^]*)?',
        re.IGNORECASE
    )
    localhost_urls = localhost_pattern.findall(text)
    for localhost_url in localhost_urls:
        if not localhost_url.startswith(('http://', 'https://')):
            urls.append(f"http://{localhost_url}")
        else:
            urls.append(localhost_url)
    
    # Clean up URLs and remove duplicates while preserving order
    cleaned_urls = []
    seen = set()
    
    for url in urls:
        # Remove trailing punctuation that might be captured
        url = re.sub(r'[.,;:!?)\]}]+$', '', url)
        
        # Normalize URL
        url = url.strip()
        
        # Skip empty or very short URLs
        if len(url) < 4:
            continue
            
        # Skip duplicates (case-insensitive)
        url_lower = url.lower()
        if url_lower not in seen:
            seen.add(url_lower)
            cleaned_urls.append(url)
    
    return cleaned_urls

def _detect_source_type_from_ct(ct: str, url: str):
    ct = (ct or "").lower()
    path = urlparse(url).path.lower()
    if "json" in ct or path.endswith(".json"): return "json"
    if "csv" in ct or path.endswith(".csv"): return "csv"
    if "pdf" in ct or path.endswith(".pdf"): return "pdf"
    # heuristic: html or unknown
    if "html" in ct or not ct: return "html"
    return "unknown"

from bs4 import BeautifulSoup
import re

def detect_noisy_values(table_html, headers):
    noisy_values = {}
    try:
        soup = BeautifulSoup(table_html, "lxml")
        rows = soup.find_all("tr")
        data_rows = []
        for row in rows[1:]:  # skip header row
            cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
            if len(cells) == len(headers):  # only keep matching length
                data_rows.append(cells)

        if not data_rows:
            return noisy_values  # no clean rows to check

        df = pd.DataFrame(data_rows, columns=headers)

        for col in df.columns:
            col_vals = df[col].dropna().astype(str)
            # Check if majority are numeric
            num_like_ratio = col_vals.str.match(r"^\d+(\.\d+)?$").sum() / len(col_vals)
            if num_like_ratio > 0.5:
                # Find values with extra non-numeric characters (noisy)
                noise = col_vals[col_vals.str.contains(r"[^\d.,-]", regex=True)].unique().tolist()
                if noise:
                    noisy_values[col] = noise

    except Exception as e:
        noisy_values["_error"] = str(e)

    return noisy_values



def _probe_url(url: str, timeout=15):
    info = {"filename": url, "url": url, "is_url": True}
    try:
        # HEAD first; some sites block HEAD → fall back to small GET
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        ct = r.headers.get("Content-Type","").split(";")[0].strip()
        if not ct or r.status_code >= 400:
            r = requests.get(url, stream=True, timeout=timeout)
            ct = r.headers.get("Content-Type","").split(";")[0].strip()
        info["type"] = ct
        stype = _detect_source_type_from_ct(ct, url)
        info["extension"] = {"csv":".csv","json":".json","pdf":".pdf"}.get(stype,"")
        info["saved_path"] = url  # for URLs we keep the url here
        info["source_type"] = stype

        if stype == "csv":
            # read tiny sample for schema
            sample = pd.read_csv(url, nrows=64)
            info["csv_metadata"] = {
                "columns": list(map(str, sample.columns)),
                "dtypes": {c: str(sample.dtypes[c]) for c in sample.columns},
                "sample_row": sample.head(1).to_dict(orient="records")[0] if not sample.empty else {}
            }
        elif stype == "json":
            js = requests.get(url, timeout=timeout).json()
            if isinstance(js, list):
                obj = next((x for x in js if isinstance(x, dict)), {})
            elif isinstance(js, dict):
                obj = js
            else:
                obj = {}
            info["json_metadata"] = {
                "top_level_type": "array" if isinstance(js, list) else type(js).__name__,
                "keys": list(obj.keys()),
                "dtypes": {k: type(obj.get(k)).__name__ for k in obj.keys()},
                "sample_object": obj
            }
        elif stype == "pdf":
            # download small temp file (pdfplumber needs a file)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                binr = requests.get(url, timeout=timeout).content
                tmp.write(binr)
                tmp.flush()
                tmp_path = tmp.name
            try:
                meta = {"page_count": None, "text_preview": "", "tables": []}
                with pdfplumber.open(tmp_path) as pdf:
                    meta["page_count"] = len(pdf.pages)
                    if pdf.pages:
                        t = (pdf.pages[0].extract_text() or "")[:100]
                        meta["text_preview"] = t
                info["pdf_metadata"] = meta
            finally:
                os.remove(tmp_path)
        elif stype == "html":
            from bs4 import BeautifulSoup
            try:
                html_resp = requests.get(url, timeout=timeout)
                html_resp.raise_for_status()
                soup = BeautifulSoup(html_resp.text, "html.parser")
            except Exception as e:
                info["html_metadata"] = {
                    "error": f"Failed to fetch or parse HTML: {type(e).__name__}: {e}"
                }
                return info

            all_tables = soup.find_all("table")
            tables_total = len(all_tables)
            tables_info = []

            for idx, tbl in enumerate(all_tables[:2]):
                # Extract opening tag with attributes
                opening_tag = str(tbl).split("</table>")[0].split(">")[0] + ">"
                caption_tag = tbl.find("caption")
                caption = caption_tag.get_text(strip=True) if caption_tag else ""

                # Extract headers + first 2 data rows
                rows = tbl.find_all("tr")
                headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])] if rows else []
                data_rows = []
                for r in rows[1:3]:
                    data_rows.append([c.get_text(strip=True) for c in r.find_all(["td", "th"])])

                # Detect noisy values
                noisy_vals = detect_noisy_values(str(tbl), headers)
                tables_info.append({
                    "table_index": idx,
                    "caption": caption,
                    "opening_tag": str(tbl).split(">")[0] + ">",
                    "headers": headers,
                    "noisy_values": noisy_vals
                })

            title = soup.title.string.strip() if soup.title else ""
            headings = {f"h{i}": [h.get_text(strip=True) for h in soup.find_all(f"h{i}")] for i in range(1, 7)}
            body_text_preview = soup.get_text(separator=" ", strip=True)[:50]

            info["html_metadata"] = {
                "title": title,
                "tables_total": tables_total,
                "tables_info": tables_info
            }

        return info
    except Exception as e:
        info["probe_error"] = f"{type(e).__name__}: {e}"
        info["source_type"] = "unknown"
        return info

import unicodedata

def _sanitize_text(s: str, mode: str = "replace") -> str:
    """
    Return a UTF‑8 safe string.
    - mode="replace": unknowns become �
    - mode="ignore": drop unencodable bytes
    - mode="ascii": strip to ASCII (best-effort)
    """
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFC", s)  # canonical normalize
    if mode == "ascii":
        return s.encode("ascii", "ignore").decode("ascii")
    # UTF‑8 can encode all Unicode; this ensures later .encode(...) calls won't explode
    return s.encode("utf-8", errors=mode).decode("utf-8", errors=mode)

def _to_safe(obj, mode: str = "replace"):
    """
    Recursively sanitize any strings inside dict/list/tuple/str so they
    are safe to print/write. Non-strings are returned as-is.
    """
    if isinstance(obj, dict):
        return {(_sanitize_text(k, mode) if isinstance(k, str) else k): _to_safe(v, mode) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_safe(x, mode) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_safe(list(obj), mode))
    if isinstance(obj, str):
        return _sanitize_text(obj, mode)
    return obj

def _safe_debug(obj, prefix=""):
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    print(prefix + s)


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
# ---------- route ----------

@app.post("/api")
async def upload_files(request: Request):
    try:
        start = time.time()
        print(start)
        upload_dir = pathlib.Path("uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        form_data = await request.form()

        data_files = []
        questions = None

        for param_name, param_value in form_data.items():
            # Check if the form parameter is a file
            if hasattr(param_value, "filename") and param_value.filename:
                filename = param_value.filename
                content_type = getattr(param_value, "content_type", "") or ""
                file_ext = pathlib.Path(filename).suffix.lower()

                # Critical change: Check the parameter name, not the filename
                if param_name == "questions.txt":
                    content = await param_value.read()
                    questions = content.decode("utf-8")
                    continue

                # For all other uploaded files, save them and process as data files
                file_path = upload_dir / filename
                content = await param_value.read()
                with open(file_path, "wb") as f:
                    f.write(content)

                file_info = {
                    "filename": filename,
                    "type": content_type,
                    "extension": file_ext,
                    "saved_path": str(file_path)
                }

                # If it's an image, generate and store description
                if _is_image_content_type(content_type) or _is_image_filename(filename):
                    file_info["image_description"] = get_image_description(str(file_path))
                    print("Image description generated", file_info["image_description"])

                # If it's a CSV, attach lightweight schema + sample row
                if _is_csv(content_type, filename):
                    file_info["csv_metadata"] = get_csv_metadata(str(file_path), sample_rows=1)
                # If it's a JSON file, attach structure summary
                if _is_json(content_type, filename):
                    file_info["json_metadata"] = get_json_metadata(str(file_path))
                # If it's an Excel file, attach sheet-wise structure summary
                if _is_excel(content_type, filename):
                    file_info["excel_metadata"] = get_excel_metadata(str(file_path), sample_rows=3)
                if _is_pdf(content_type, filename):
                    file_info["pdf_metadata"] = get_pdf_metadata(str(file_path), max_pages=3, max_text_chars=1000)
                           
                data_files.append(file_info)
        print("[APP] 1: Received All Files Info")
        # 1) Extract & probe URLs in questions
        urls = _extract_urls_comprehensive(questions)
        for u in urls:
            url_info = _probe_url(u)
            # normalize to planner-friendly shape
            data_files.append(url_info)
        url_present = False
        if len(urls)>0:
            url_present = True
        print("[APP] 2: Received All URLS info")
        # make the values themselves safe for later usage
        data_files = _to_safe(data_files, mode="replace")
        questions  = _to_safe(questions,  mode="replace")

        # optional: debug print
        _safe_debug(data_files, "data_files: ")
        _safe_debug(questions,  "Questions: ")
        if not questions:
            raise HTTPException(status_code=400, detail="questions.txt form field is required and must contain a file.")
        
        print("[APP] 3 Calling the planner Agent")
        # Build plan from questions + uploaded artifacts
        # plan = run_planner_agent_files(questions, data_files)
        plan = run_planner_agent_json_with_feedback_looping(questions,data_files)
        print("[APP] 4 GOT THE PLAN")
        
        if isinstance(plan, (dict, list)):
            # plan.json
            with open("plan.json", "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2, ensure_ascii=False)

        else:
            with open("plan.txt", "w", encoding="utf-8", errors="replace") as f:
                f.write(str(plan))
        # return JSONResponse({"Questions":questions,"data files":data_files,"plan":plan})

        print("[APP] 5 CALLING EXECUTE PLAN ")
        result = execute_plan_v1(plan, questions , data_files)
        print(result)
        print("[APP] 6 PLAN EXECUTED SUCCESSFULLY WITH THE RESULT")
        end = time.time()
        print("Starting",start)
        print("Ending",end)
        print(end-start)
        try:
            parsed = json.loads(result)
            with open("results.json","w") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            # Fallback: always return JSON, even if executor output is broken
            parsed = {
                "error": "Executor did not emit valid JSON.",
                "raw_output": result
            }
        return parsed

        if not result.get("ok", False):
            return JSONResponse(status_code=500, content=result)

        return JSONResponse(content=result["result"])

    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    # uvicorn.run("app:app", host="127.0.0.1", port=7680, reload=True)
    uvicorn.run("app:app", host="127.1.1.1", port=8000, reload=True)
