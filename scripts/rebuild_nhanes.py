"""
Build AHEAD's NHANES research file from the raw CDC downloads.

    python scripts/fetch_nhanes.py                      # 1. download the CDC files
    python scripts/rebuild_nhanes.py                    # 2. this script
    python scripts/build_nhanes_training.py             # 3. the two training files
    python train_model.py                               # 4. the models

Output: data/nhanes/nhanes_1999_2023_research.csv.gz, one row per NHANES participant aged 18 or over
(71,194 people), with the participant number (SEQN), exam status and survey design columns kept so every
row can be traced back to CDC's files.

What it does beyond merging the modules:
- Serum creatinine is corrected as CDC's laboratory notes advise: 1999-2000 (LAB18: "highly recommended",
  standard = 1.013 x creatinine + 0.147) and 2005-2006 (BIOPRO_D: standard = -0.016 + 0.978 x creatinine).
  2001-2002 stores creatinine as LBDSCR and needs no correction (L40_B).
- eGFR uses the race-free CKD-EPI 2021 creatinine equation.
- The kidney marker (eGFR below 60 or urine albumin-to-creatinine ratio of 30 mg/g or more) is only set
  when both tests were done at the visit, so people who had just one test are not labelled.
- Implausible values are left blank (BMI outside 10-90, lab values outside wide physiological ranges).

The column rules follow the extract a teammate's friend prepared from the same CDC files. With
--compare <that extract>, the script rebuilds it without the corrections and reports how many rows match
(71,172 of 71,194, 99.97%, when this was written).
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_nhanes import CYCLES, DEFAULT_RAW, file_stem, local_name

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "nhanes" / "nhanes_1999_2023_research.csv.gz"

FIELDS = {
    "DEMO": ["RIDAGEYR", "RIAGENDR", "RIDSTATR", "WTMEC2YR", "WTMEC4YR", "WTMECPRP", "SDMVPSU", "SDMVSTRA"],
    "BMX": ["BMXBMI"],
    "SMQ": ["SMQ020", "SMQ040"],
    "PAQ": ["PAQ650", "PAQ665", "PAD790Q", "PAD810Q"],
    "BPQ": ["BPQ020", "BPQ080"],
    "DIQ": ["DIQ010"],
    "MCQ": ["MCQ160C", "MCQ160E", "MCQ160F"],
    "BIOPRO": ["LBXSGL", "LBXSCR", "LBDSCR", "LBXSBU"],
    "GHB": ["LBXGH"],
    "TCHOL": ["LBXTC"],
    "ALB_CR": ["URDACT", "URXUMA", "URXUCR"],
}
AGE_GROUPS = ["18–29", "30–39", "40–49", "50–59", "60–69", "70–79", "80+"]


def read_module(raw: Path, year: int, stem: str, columns: list[str]) -> pd.DataFrame | None:
    path = raw / local_name(year, stem)
    if not path.exists():
        print(f"  missing {path.name}; its columns stay blank for this cycle")
        return None
    data = pd.read_sas(path, format="xport")
    keep = ["SEQN"] + [column for column in columns if column in data.columns]
    data = data[keep].copy()
    data["SEQN"] = pd.to_numeric(data["SEQN"], errors="coerce")
    return data.dropna(subset=["SEQN"]).drop_duplicates("SEQN")


def merge_cycles(raw: Path) -> pd.DataFrame:
    frames = []
    for cycle, year, suffix in CYCLES:
        base = read_module(raw, year, file_stem("DEMO", year, suffix), FIELDS["DEMO"])
        if base is None:
            raise FileNotFoundError(f"No demographics file for {cycle}; run scripts/fetch_nhanes.py first.")
        for module, columns in FIELDS.items():
            if module == "DEMO":
                continue
            table = read_module(raw, year, file_stem(module, year, suffix), columns)
            if table is not None:
                base = base.merge(table, on="SEQN", how="left", validate="one_to_one")
        base["cycle"] = cycle
        frames.append(base)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    for column in [c for columns in FIELDS.values() for c in columns]:
        if column not in merged:
            merged[column] = np.nan
    return merged[pd.to_numeric(merged["RIDAGEYR"], errors="coerce") >= 18].reset_index(drop=True)


def yes_no(series: pd.Series) -> pd.Series:
    """NHANES 1 = yes, 2 = no; refused (7), don't know (9) and blank stay missing."""
    return series.map({1.0: 1.0, 2.0: 0.0})


def within(series: pd.Series, low: float, high: float) -> pd.Series:
    return series.where(series.between(low, high))


def egfr_ckd_epi_2021(creatinine, age, female) -> pd.Series:
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = creatinine / kappa
    value = (142 * np.minimum(ratio, 1) ** alpha * np.maximum(ratio, 1) ** -1.2 * 0.9938 ** age
             * np.where(female, 1.012, 1.0))
    return pd.Series(value, index=creatinine.index).where(lambda s: s.between(1, 180))


def leisure_activity(data: pd.DataFrame) -> pd.Series:
    """Any moderate or vigorous leisure-time activity. 2007-2018 ask yes/no (PAQ650/PAQ665); 2021-2023 ask
    how often (PAD790Q/PAD810Q, 0 = none). 1999-2006 asked different questions and stay blank."""
    older = pd.Series(np.nan, index=data.index)
    older[data[["PAQ650", "PAQ665"]].eq(1).any(axis=1)] = 1.0
    older[data[["PAQ650", "PAQ665"]].eq(2).all(axis=1)] = 0.0
    moderate, vigorous = within(data["PAD790Q"], 0, 1000), within(data["PAD810Q"], 0, 1000)
    newer = pd.Series(np.nan, index=data.index)
    newer[moderate.gt(0) | vigorous.gt(0)] = 1.0
    newer[moderate.eq(0) & vigorous.eq(0)] = 0.0
    return older.combine_first(newer)


def urine_acr(data: pd.DataFrame) -> pd.Series:
    """CDC's ratio (URDACT) where given, otherwise albumin (ug/mL) / creatinine (mg/dL) x 100 in mg/g."""
    acr = data["URDACT"].copy()
    ratio = 100 * data["URXUMA"] / data["URXUCR"]
    usable = (data["URXUMA"] >= 0) & (data["URXUCR"] > 0) & np.isfinite(ratio)
    acr[acr.isna() & usable] = ratio[acr.isna() & usable]
    return within(acr, 0, 10000)


def kidney_marker(egfr: pd.Series, acr: pd.Series) -> pd.Series:
    both = egfr.notna() & acr.notna()
    marker = pd.Series(np.nan, index=egfr.index)
    marker[both] = ((egfr < 60) | (acr >= 30))[both].astype(float)
    return marker


def base_columns(data: pd.DataFrame, creatinine: pd.Series) -> pd.DataFrame:
    female = data["RIAGENDR"].eq(2)
    smoked, told_bp, told_cholesterol = yes_no(data["SMQ020"]), yes_no(data["BPQ020"]), yes_no(data["BPQ080"])
    activity = leisure_activity(data)
    acr = urine_acr(data)
    egfr = egfr_ckd_epi_2021(creatinine, data["RIDAGEYR"], female)
    return pd.DataFrame({
        "cycle": data["cycle"], "age": data["RIDAGEYR"], "sex": np.where(female, "Female", "Male"),
        "bmi": within(data["BMXBMI"], 10, 90),
        "smoking": smoked.map({1.0: "Ever smoked ≥100 cigarettes", 0.0: "Never smoked ≥100 cigarettes"}),
        "hypertension": told_bp.map({1.0: "Previously told: yes", 0.0: "Previously told: no"}),
        "high_cholesterol": told_cholesterol.map({1.0: "Previously told: yes", 0.0: "Previously told: no"}),
        "diabetes_history": yes_no(data["DIQ010"]),            # 3 = borderline stays blank
        "heart_history": yes_no(data["MCQ160C"]),              # told coronary heart disease (asked 20+)
        "activity": activity.map({1.0: "Leisure activity reported", 0.0: "No leisure activity reported"}),
        "glucose_mg_dl": within(data["LBXSGL"], 30, 900), "hba1c_pct": within(data["LBXGH"], 2, 20),
        "cholesterol_mg_dl": within(data["LBXTC"], 50, 900), "creatinine_mg_dl": creatinine,
        "bun_mg_dl": within(data["LBXSBU"], 1, 200), "uacr_mg_g": acr, "egfr_calculated": egfr,
        "kidney_marker": kidney_marker(egfr, acr),
    })


def research_file(data: pd.DataFrame) -> pd.DataFrame:
    measured = data["LBXSCR"].combine_first(data["LBDSCR"])          # 2001-2002 uses LBDSCR
    corrected = measured.where(~data["cycle"].eq("1999-2000"), 1.013 * measured + 0.147)
    corrected = corrected.where(~data["cycle"].eq("2005-2006"), -0.016 + 0.978 * measured)
    out = base_columns(data, within(corrected, 0.1, 30))
    smoked = yes_no(data["SMQ020"])
    out.insert(0, "SEQN", data["SEQN"].astype("Int64"))
    out["exam_status"] = data["RIDSTATR"]                            # 1 interview only, 2 interview + exam
    out["current_smoker"] = data["SMQ040"].map({1.0: 1.0, 2.0: 1.0, 3.0: 0.0}).where(smoked.eq(1),
                                                                                     np.where(smoked.eq(0), 0.0, np.nan))
    out["heart_attack_history"] = yes_no(data["MCQ160E"])
    out["stroke_history"] = yes_no(data["MCQ160F"])
    out["creatinine_as_measured_mg_dl"] = measured
    out["creatinine_corrected"] = data["cycle"].isin(["1999-2000", "2005-2006"]) & measured.notna()
    out["weight_mec_2yr"] = data["WTMEC2YR"].combine_first(data["WTMECPRP"])
    out["weight_mec_4yr_1999_2002"] = data["WTMEC4YR"]
    out["psu"], out["strata"] = data["SDMVPSU"], data["SDMVSTRA"]
    out["age_group"] = pd.cut(out["age"], [17, 29, 39, 49, 59, 69, 79, 200], labels=AGE_GROUPS).astype(str)
    return out


def compare_with_extract(data: pd.DataFrame, extract: Path) -> None:
    """Rebuild the friend's extract (no creatinine correction, LBXSCR only, eGFR only where ACR exists)."""
    replica = base_columns(data, within(data["LBXSCR"], 0.1, 30))
    replica["egfr_calculated"] = replica["egfr_calculated"].where(replica["uacr_mg_g"].notna())
    friend = pd.read_csv(extract, low_memory=False)
    columns = list(replica.columns)

    def rows(frame):
        values = frame[columns].copy()
        for column in columns:
            if values[column].dtype.kind == "f":
                values[column] = values[column].round(3)
        return Counter(map(tuple, values.astype(str).values))

    matched = sum((rows(friend) & rows(replica)).values())
    same_cycles = friend["cycle"].value_counts().equals(replica["cycle"].value_counts().reindex(friend["cycle"].value_counts().index))
    print(f"Compared with {extract.name}: {matched:,} of {len(friend):,} rows identical on all {len(columns)} columns "
          f"({matched / len(friend):.2%}); people per cycle identical: {same_cycles}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--compare", type=Path, help="the friend's nhanes_1999_2023_analytics.csv.gz")
    args = parser.parse_args()

    data = merge_cycles(args.raw)
    out = research_file(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out):,} adults)")
    print(out.groupby("cycle", sort=False).agg(adults=("age", "size"), hba1c=("hba1c_pct", "count"),
                                              creatinine=("creatinine_mg_dl", "count"),
                                              kidney_marker=("kidney_marker", "count")).to_string())
    if args.compare:
        compare_with_extract(data, args.compare)


if __name__ == "__main__":
    main()
