"""
MicroDrugNet — Automated Dataset Downloader
Downloads all required public datasets for training.
"""
import argparse
from pathlib import Path

import requests
from requests.exceptions import RequestException

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "raw"

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
)

DATASETS = {
    "pubchem_sample": {
        "url": "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/Compound_000000001_000500000.sdf.gz",
        "desc": "PubChem compounds (first 500k) — SMILES source",
        "out": RAW_DIR / "pubchem_sample.sdf.gz",
    },
    "hmdb_metabolites": {
        "url": "https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip",
        "desc": "Human Metabolome Database — metabolite fingerprints",
        "out": RAW_DIR / "hmdb_metabolites.zip",
        "manual_note": (
            "HMDB now commonly blocks direct scripted downloads with HTTP 403. "
            "Open https://www.hmdb.ca/downloads, submit the short download form, "
            "then place the metabolites archive in data/raw/ as hmdb_metabolites.zip."
        ),
    },
}

MANUAL_DATASETS = {
    "HMDB": {
        "url": "https://www.hmdb.ca/downloads",
        "file": "hmdb_metabolites.zip",
        "note": (
            "HMDB download links are currently gated behind a short form on the "
            "downloads page; save the metabolites archive into data/raw/"
        ),
    },
    "DrugBank": {
        "url": "https://go.drugbank.com/releases/latest#open-data",
        "file": "drugbank_all_full_database.xml.zip",
        "note": "Requires FREE academic registration at drugbank.com",
    },
    "HMP (Human Microbiome Project)": {
        "url": "https://hmpdacc.org/hmp/HMQCP/",
        "note": "Download 16S OTU tables — no login required",
    },
    "GMrepo": {
        "url": "https://gmrepo.humangut.info/Downloads",
        "note": "Download run_taxon_relative_abundance tables",
    },
    "miMDB": {
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8728195/",
        "note": "Supplementary Table S1 — microbiome-drug interaction pairs",
    },
}

def download(name, info):
    out = Path(info["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"  [SKIP] {name} already downloaded")
        return True
    print(f"  [GET]  {name}: {info['desc']}")
    try:
        with SESSION.get(info["url"], stream=True, timeout=120, allow_redirects=True) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
    except RequestException as exc:
        if out.exists():
            out.unlink()
        print(f"  [WARN] {name} download failed: {exc}")
        if info.get("manual_note"):
            print(f"         {info['manual_note']}")
        return False

    print(f"  [OK]   Saved to {out}")
    return True

def print_manual():
    print("\n" + "="*60)
    print("MANUAL DOWNLOADS REQUIRED (free, but need registration):")
    print("="*60)
    for name, info in MANUAL_DATASETS.items():
        print(f"\n{name}")
        print(f"  URL : {info['url']}")
        print(f"  Note: {info['note']}")
    print("\nPlace all files in:  data/raw/\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-only", action="store_true")
    args = parser.parse_args()

    print("MicroDrugNet — Dataset Downloader")
    failed = []
    for name, info in DATASETS.items():
        if not download(name, info):
            failed.append(name)

    if not args.auto_only:
        print_manual()

    if failed:
        print("\nAutomatic downloads that still need attention:")
        for name in failed:
            print(f"  - {name}")
