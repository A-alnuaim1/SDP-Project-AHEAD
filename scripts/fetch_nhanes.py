"""
Download the NHANES public-use files AHEAD is built from, straight from CDC.

    python scripts/fetch_nhanes.py            # into data/nhanes/raw/ (ignored by git)
    python scripts/fetch_nhanes.py <folder>

121 SAS transport (.xpt) files: 11 modules for each survey cycle from 1999-2000 to 2021-2023, about
283 MB in total. Files already in the folder are not downloaded again. Every request goes to
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/... with certificate checks on, and the result is written to
download_log.csv in the same folder. Then run scripts/rebuild_nhanes.py.

The cycles and file names follow the NHANES data-building notebooks a teammate's friend wrote for AHEAD
(research/AHEAD_NHANES_1999_2023_Data_Audit.ipynb and the age-18 correction notebook); this script does the
same downloads without Colab.
"""
from __future__ import annotations

import csv
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "nhanes" / "raw"

# (cycle label, CDC folder year, file suffix). 2017-March 2020 ('P_') replaces 2017-2018 ('_J'); never use both.
CYCLES = [("1999-2000", 1999, ""), ("2001-2002", 2001, "B"), ("2003-2004", 2003, "C"), ("2005-2006", 2005, "D"),
          ("2007-2008", 2007, "E"), ("2009-2010", 2009, "F"), ("2011-2012", 2011, "G"), ("2013-2014", 2013, "H"),
          ("2015-2016", 2015, "I"), ("2017-March2020", 2017, "P"), ("2021-2023", 2021, "L")]
MODULES = ["DEMO", "BMX", "SMQ", "PAQ", "BPQ", "DIQ", "MCQ", "BIOPRO", "GHB", "TCHOL", "ALB_CR"]
# The first three cycles name their laboratory files differently.
LEGACY = {1999: {"BIOPRO": "LAB18", "GHB": "LAB10", "TCHOL": "LAB13", "ALB_CR": "LAB16"},
          2001: {"BIOPRO": "L40_B", "GHB": "L10_B", "TCHOL": "L13_B", "ALB_CR": "L16_B"},
          2003: {"BIOPRO": "L40_C", "GHB": "L10_C", "TCHOL": "L13_C", "ALB_CR": "L16_C"}}


def file_stem(module: str, year: int, suffix: str) -> str:
    if module in LEGACY.get(year, {}):
        return LEGACY[year][module]
    if suffix == "P":
        return "P_" + module
    return f"{module}_{suffix}" if suffix else module


def local_name(year: int, stem: str) -> str:
    return f"{year}_{stem}.xpt"


def main(raw: Path) -> int:
    raw.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    log, total = [], 0
    for cycle, year, suffix in CYCLES:
        for module in MODULES:
            stem = file_stem(module, year, suffix)
            url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{stem}.xpt"
            target = raw / local_name(year, stem)
            status = "already downloaded"
            if not target.exists():
                status = "failed"
                for attempt in range(3):
                    try:
                        request = urllib.request.Request(url, headers={"User-Agent": "AHEAD student research (NHANES public-use)"})
                        with urllib.request.urlopen(request, timeout=120, context=context) as response:
                            body = response.read()
                        if not body.startswith(b"HEADER RECORD"):
                            raise ValueError("the response is not a SAS transport file")
                        target.write_bytes(body)
                        status = "downloaded"
                        break
                    except Exception as error:  # noqa: BLE001 - every failure is logged and retried
                        status = f"failed: {str(error)[:120]}"
                        time.sleep(2 * (attempt + 1))
                time.sleep(0.5)   # be gentle with the CDC server
            size = target.stat().st_size if target.exists() else 0
            total += size
            log.append({"cycle": cycle, "module": module, "file": stem, "url": url, "status": status, "bytes": size})
            print(f"{cycle:15} {module:7} {stem:12} {status}", flush=True)
    with open(raw / "download_log.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log[0]))
        writer.writeheader()
        writer.writerows(log)
    failed = [row for row in log if row["status"].startswith("failed")]
    print(f"\n{len(log) - len(failed)} of {len(log)} files present, {total / 1e6:.1f} MB; {len(failed)} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RAW))
