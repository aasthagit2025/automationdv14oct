# app.py
import streamlit as st
import pandas as pd
import pyreadstat
import io
import re
from typing import List, Dict

st.set_page_config(layout="wide")
st.title("📊 Auto Validation Rules + Failed Checks Generator")

# --- Uploads ---
data_file = st.file_uploader("Upload raw survey data (CSV / XLSX / SAV)", type=["csv", "xlsx", "sav"])
skip_file = st.file_uploader("Upload skip rules (CSV or XLSX) — optional", type=["csv", "xlsx"])
constructed_txt = st.file_uploader("Upload Constructed List export (text file) — optional", type=["txt"])

if not data_file:
    st.info("Upload raw data to start.")
    st.stop()

# --- Load data ---
def load_data(f):
    if f.name.endswith(".csv"):
        return pd.read_csv(f, encoding_errors="ignore")
    elif f.name.endswith(".xlsx"):
        return pd.read_excel(f)
    elif f.name.endswith(".sav"):
        df, meta = pyreadstat.read_sav(f)
        return df
    else:
        raise ValueError("Unsupported data type")

try:
    df = load_data(data_file)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Detect id column (flexible)
id_candidates = ["RespondentID", "Password", "RespID", "RID", "sys_RespNum"]
id_col = next((c for c in id_candidates if c in df.columns), df.columns[0] if len(df.columns)>0 else None)
if id_col is None:
    st.error("No columns found in data.")
    st.stop()

# --- Helpers ---
def treat_as_blank(series: pd.Series):
    """Blank = NaN or empty string. BUT literal 'NA' (case-insensitive) should be considered as answered per user."""
    s = series.astype(str)
    blank = series.isna() | (s.str.strip() == "")
    return blank

def is_multiselect_group(cols: List[str], df: pd.DataFrame):
    """Heuristic: if majority of values are 0/1 and there are >=2 columns, treat as multiselect group."""
    if len(cols) < 2:
        return False
    counts = []
    for c in cols:
        ser = df[c].dropna()
        if ser.empty:
            counts.append(1.0)  # empty -> treat OK
            continue
        prop01 = ser.astype(str).isin(["0","1","0.0","1.0"]).mean()
        counts.append(prop01)
    return sum(1 for x in counts if x>=0.9) >= max(1, len(cols)//2)

def group_prefixes(columns: List[str]):
    """Group names by common prefix ending with '_' or base like Q2_1..Q2_6 -> prefix 'Q2_' or 'Q2' """
    groups = {}
    for c in columns:
        m = re.match(r"^(.+?[_]?)\d+$", c)  # prefix with trailing underscore optional
        if m:
            pref = m.group(1)
            groups.setdefault(pref, []).append(c)
        else:
            # try suffix pattern like _r1, _r2 etc.
            m2 = re.match(r"^(.+?_r)\d+$", c)
            if m2:
                pref = m2.group(1)
                groups.setdefault(pref, []).append(c)
            else:
                groups.setdefault(c, []).append(c)
    return groups

def parse_skip_rows_from_file(f):
    """Try to parse skip rules file (flexible columns). Return list of dicts with keys: condition_text, target_text, raw."""
    rows = []
    try:
        if f.name.endswith(".csv"):
            sk = pd.read_csv(f, dtype=str, encoding_errors="ignore")
        else:
            sk = pd.read_excel(f, dtype=str)
    except Exception:
        return rows
    # common variations: columns like ['Skip From','Logic','Skip To'], or ['Question','Condition']
    cols = [c.lower() for c in sk.columns]
    # find logic-like column
    possible_logic_cols = [c for c in sk.columns if any(k in c.lower() for k in ("logic","condition","if","then"))]
    possible_target_cols = [c for c in sk.columns if any(k in c.lower() for k in ("skip to","skipto","skip_to","skip","to","question","target"))]
    if possible_logic_cols:
        logic_col = possible_logic_cols[0]
    else:
        logic_col = sk.columns[0]
    if possible_target_cols:
        target_col = possible_target_cols[0]
    else:
        target_col = sk.columns[-1]
    for _, r in sk.iterrows():
        logic = str(r.get(logic_col,"")).strip()
        target = str(r.get(target_col,"")).strip()
        if logic and logic.lower()!="nan":
            rows.append({"logic": logic, "target": target, "raw": r.to_dict()})
    return rows

def parse_constructed_text(txt_bytes: bytes):
    """Extract simple constructs: ADD(ParentListName(),start,end) -> map ParentListName -> Range start-end
       Also extract constructs that use VALUE(...) -> complex; we include the raw block for manual review.
    """
    text = txt_bytes.decode("utf-8", errors="ignore")
    results = []
    # blocks: List Name: <name> ... Type: Constructed ... [Logic]: ... End Unverified
    blocks = re.split(r"={3,}", text)  # rough split
    # simpler approach: find 'List Name:' and subsequent 'Type:' and '[Logic]:' block
    list_blocks = re.findall(r"List Name:\s*(.+?)\n(.+?)(?=(?:\nList Name:|\Z))", text, flags=re.S)
    for name, block in list_blocks:
        if "Constructed" in block or "Constructed" in block.splitlines()[0]:
            # find ADD calls
            adds = re.findall(r"ADD\s*\(\s*ParentListName\(\)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", block, flags=re.I)
            for start, end in adds:
                results.append({"list_name": name.strip(), "type": "add_range", "start": int(start), "end": int(end), "raw": block.strip()})
            # find Add(ParentListName(),1,5) style
            adds2 = re.findall(r"Add\s*\(\s*ParentListName\(\)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", block, flags=re.I)
            for start, end in adds2:
                results.append({"list_name": name.strip(), "type": "add_range", "start": int(start), "end": int(end), "raw": block.strip()})
            # also capture simple VALUE(...) based blocks for review
            if "VALUE(" in block.upper() or "VALUE " in block.upper():
                results.append({"list_name": name.strip(), "type": "complex_logic", "raw": block.strip()})
    return results

def get_condition_mask(cond_text, df):
    """Robust parsing: supports 'If A=1', 'If Not(A=1 or A=2)', AND, OR, operators <= >= < > != ="""
    if not cond_text:
        return pd.Series(True, index=df.index)
    s = cond_text.strip()
    if s.lower().startswith("if"):
        s = s[2:].strip()
    # handle Not(...) forms
    neg = False
    s = s.strip()
    if s.lower().startswith("not(") and s.endswith(")"):
        neg = True
        s = s[4:-1]
    # split OR groups
    or_groups = re.split(r"\s+or\s+", s, flags=re.I)
    mask = pd.Series(False, index=df.index)
    for g in or_groups:
        and_parts = re.split(r"\s+and\s+", g, flags=re.I)
        sub = pd.Series(True, index=df.index)
        for p in and_parts:
            p = p.strip()
            p = p.replace("<>", "!=")
            matched = False
            for op in ["<=", ">=", "!=", "<", ">", "="]:
                if op in p:
                    left, right = [x.strip() for x in p.split(op,1)]
                    if left not in df.columns:
                        sub &= False
                        matched = True
                        break
                    colvals = df[left]
                    # try numeric compare
                    try:
                        rval = float(right)
                        coln = pd.to_numeric(colvals, errors="coerce")
                        if op == "<=": sub &= coln <= rval
                        elif op == ">=": sub &= coln >= rval
                        elif op == "<": sub &= coln < rval
                        elif op == ">": sub &= coln > rval
                        elif op == "=": sub &= coln == rval
                    except Exception:
                        # string compare (strip)
                        if op in ("!=",):
                            sub &= colvals.astype(str).str.strip() != right
                        else:
                            sub &= colvals.astype(str).str.strip() == right
                    matched = True
                    break
            if not matched:
                sub &= False
        mask |= sub
    if neg:
        return ~mask
    return mask

# --- Parse skip file & constructed if provided ---
skip_rules_parsed = []
if skip_file:
    skip_rules_parsed = parse_skip_rows_from_file(skip_file)

constructed_parsed = []
if constructed_txt:
    try:
        constructed_parsed = parse_constructed_text(constructed_txt.read())
    except Exception as e:
        st.warning(f"Couldn't parse constructed list file: {e}")

# --- Detect groups & types ---
cols = list(df.columns)
groups = group_prefixes(cols)  # prefix -> columns list

# Build a canonical list of variables to generate rules for, in data column order
vars_in_order = cols[:]  # we will generate rules per column, for multiselect we will only emit one rule for prefix (see below)

# Decide which prefixes are multi-select
multiselect_prefixes = {}
for prefix, clist in groups.items():
    if len(clist) > 1 and is_multiselect_group(clist, df):
        multiselect_prefixes[prefix] = clist

# Utility: find constructed list-based ranges (map list_name -> (start,end))
constructed_ranges = {}
for item in constructed_parsed:
    if item.get("type") == "add_range":
        # try to match list_name to variable prefixes present
        name = item["list_name"]
        # common mapping: ForQ1ColList -> Q1 etc. We'll keep mapping flexible: look for any column starting with name or name lowercased variants
        constructed_ranges[name] = (item["start"], item["end"])

# --- Generate validation rules for each variable (in df order) ---
rules = []
seen_multiselect_prefixes = set()

for col in vars_in_order:
    # skip synthetic columns (if user wants) - we include all
    # If this col is part of a multiselect prefix we've already emitted, skip emitting duplicates
    pref = None
    for p, clist in multiselect_prefixes.items():
        if col in clist:
            pref = p
            break
    if pref and pref in seen_multiselect_prefixes:
        continue

    # For multi-select, we emit one rule using the prefix (without index if prefix ends with _ keep it)
    if pref:
        qname = pref.rstrip("_")
        check_types = ["Multi-Select"]
        conds = ["Only 0/1; At least one selected"]
        source = "Auto (multiselect detected)"
        rules.append({"Question": qname, "Check_Type": ";".join(check_types), "Condition": ";".join(conds), "Source": source})
        seen_multiselect_prefixes.add(pref)
        continue

    # Not multi-select column: decide type
    ser = df[col]
    ser_nonnull = ser.dropna().astype(str)
    unique_vals = ser_nonnull.str.strip().unique().tolist() if not ser_nonnull.empty else []
    # open-end detection (text) - many non-numeric and mean length > 10 or dtype object and many unique strings
    nonnum_ratio = 0.0
    try:
        nonnum_ratio = (~ser.dropna().apply(lambda x: str(x).replace(".","",1).lstrip("-").isdigit())).mean()
    except Exception:
        nonnum_ratio = ser.dropna().apply(lambda x: isinstance(x, str)).mean() if len(ser.dropna())>0 else 1.0

    # Rating detection: numeric ints with small domain (2-10) and many repeats
    is_numeric = pd.to_numeric(ser, errors="coerce").notna().mean() > 0.5
    rating_like = False
    if is_numeric:
        nums = pd.to_numeric(ser.dropna(), errors="coerce")
        if nums.dropna().size > 0:
            uniq = nums.dropna().unique()
            if 2 <= len(uniq) <= 10:
                rating_like = True

    # Choose checks
    check_types = []
    conditions = []
    source = "Auto (inferred)"

    if rating_like:
        # try to get range from constructed lists by matching prefix or column name
        # e.g. forITQ1Col -> forITQ1Col mapping might exist in constructed_ranges
        # fallback to actual min/max
        minv = pd.to_numeric(ser, errors="coerce").min()
        maxv = pd.to_numeric(ser, errors="coerce").max()
        # round to int if they are ints
        try:
            minv_i = int(minv) if pd.notna(minv) else None
            maxv_i = int(maxv) if pd.notna(maxv) else None
        except Exception:
            minv_i, maxv_i = minv, maxv
        rng = None
        # look for constructed range mapping using column name or common variants
        for lname, rngvals in constructed_ranges.items():
            # simple match: if lname lower in col lower
            if lname.lower() in col.lower() or col.lower() in lname.lower():
                rng = f"{rngvals[0]}-{rngvals[1]}"
                break
        if rng is None and minv_i is not None and maxv_i is not None:
            rng = f"{minv_i}-{maxv_i}"
        if rng:
            check_types.append("Range")
            conditions.append(rng)
        # always check straightliner for rating type (but groupable)
        check_types.append("Straightliner")
        # for Straightliner we store a hint using prefix (group detection above will use prefixes)
        conditions.append(f"Group({col}_prefix?)")
    elif nonnum_ratio > 0.9 and ser.dropna().astype(str).str.len().mean() > 3:
        # open-end text
        check_types.append("Missing")
        conditions.append("")  # missing has no param
        check_types.append("OpenEnd_Junk")
        conditions.append("MinLen(3)")
    elif is_numeric:
        # numeric question that isn't rating-like -> range could be specified by constructed list, or skip range
        minv = pd.to_numeric(ser, errors="coerce").min()
        maxv = pd.to_numeric(ser, errors="coerce").max()
        if pd.notna(minv) and pd.notna(maxv):
            check_types.append("Range")
            conditions.append(f"{int(minv)}-{int(maxv)}")
        check_types.append("Missing")
        conditions.append("")
    else:
        # treat as single-select / categorical
        # if few unique values (<=10) then missing & maybe range using numeric-coded values
        if len(unique_vals) <= 10 and len(unique_vals)>0 and all(re.match(r"^\d+$", str(x)) for x in unique_vals):
            # numeric-coded single select
            mn = min(int(x) for x in unique_vals)
            mx = max(int(x) for x in unique_vals)
            check_types.append("Range")
            conditions.append(f"{mn}-{mx}")
        check_types.append("Missing")
        conditions.append("")

    rules.append({"Question": col, "Check_Type": ";".join(check_types), "Condition": ";".join(conditions), "Source": source})

# --- Merge explicit skip rules (from skip CSV/XLSX) into generated rules where target matches data columns/prefixes ---
def add_skip_to_rules(rules_list: List[Dict], parsed_skips: List[Dict], df_cols: List[str]):
    for s in parsed_skips:
        logic = s.get("logic","")
        target = s.get("target","")
        # Try to extract "If <cond> then <target> should be ..." pattern
        m = re.search(r"if\s+(.+?)\s+then\s+(.+)", logic, flags=re.I)
        if m:
            cond = m.group(1).strip()
            then_text = m.group(2).strip()
            # target detection: if user included target column in "then" text use that, else use provided target
            tmatch = re.search(r"([A-Za-z0-9_]+)", then_text)
            if tmatch:
                targ = tmatch.group(1)
            else:
                targ = target or ""
            if not targ:
                continue
            # find rule for targ or prefix variant
            matched_idx = None
            for i, r in enumerate(rules_list):
                # match exact question name or prefix name
                if r["Question"].lower() == targ.lower() or r["Question"].lower().startswith(targ.lower()):
                    matched_idx = i
                    break
            cond_text = f"If {cond} then {targ} should be answered" if "then" in logic.lower() else logic
            if matched_idx is None:
                # if target not in rules (maybe prefix), create new rule entry (only if variable exists in df columns)
                if any(c.lower().startswith(targ.lower()) for c in df_cols):
                    rules_list.append({"Question": targ, "Check_Type": "Skip", "Condition": cond_text, "Source": "Skips file"})
            else:
                # append Skip to existing Check_Type if not present
                existing = rules_list[matched_idx]["Check_Type"].split(";")
                existing_cond = rules_list[matched_idx]["Condition"].split(";") if rules_list[matched_idx]["Condition"] else ["" for _ in existing]
                if "Skip" not in [e.strip() for e in existing]:
                    existing.append("Skip")
                    existing_cond.append(cond_text)
                    rules_list[matched_idx]["Check_Type"] = ";".join(existing)
                    rules_list[matched_idx]["Condition"] = ";".join(existing_cond)
                else:
                    # Merge skip condition into existing skip (append)
                    # find skip pos and append if blank
                    existing = [e.strip() for e in existing]
                    try:
                        idx_skip = existing.index("Skip")
                        conds = rules_list[matched_idx]["Condition"].split(";")
                        if cond_text not in conds[idx_skip]:
                            conds[idx_skip] = (conds[idx_skip] + " | " + cond_text).strip("| ").strip()
                            rules_list[matched_idx]["Condition"] = ";".join(conds)
                    except ValueError:
                        pass
    return rules_list

rules = add_skip_to_rules(rules, skip_rules_parsed, cols)

# --- Convert rules list to DataFrame (ordered by original data columns) ---
# We'll ensure all data columns appear (if some prefixes were emitted, keep them in appropriate order)
final_rules = []
emitted = set()
for c in cols:
    # if a multiselect prefix rule exists for this col, include it when we first encounter the prefix
    pref = None
    for p in multiselect_prefixes:
        if c in multiselect_prefixes[p]:
            pref = p.rstrip("_")
            if pref in emitted:
                pref = None
            break
    if pref:
        # find rule for prefix
        row = next((r for r in rules if r["Question"].lower()==pref.lower() or r["Question"]==pref), None)
        if row:
            final_rules.append(row)
            emitted.add(pref)
        else:
            # fallback create
            final_rules.append({"Question": pref, "Check_Type":"Multi-Select", "Condition":"Only 0/1; At least one selected", "Source":"Auto"})
            emitted.add(pref)
    # now ensure the single col rule is present (if not part of multiselect handled above)
    if c in emitted:
        continue
    row = next((r for r in rules if r["Question"]==c), None)
    if row:
        final_rules.append(row)
        emitted.add(c)
    else:
        # fallback: create minimal missing check
        final_rules.append({"Question": c, "Check_Type": "Missing", "Condition":"", "Source":"Auto(fallback)"})
        emitted.add(c)

rules_df = pd.DataFrame(final_rules)

st.write("### Preview: Generated Validation Rules (first 200 rows)")
st.dataframe(rules_df.head(200))

# --- Apply rules to produce Failed Checks sheet ---
report = []

# Helper to find related columns for a rule question name (for prefixes)
def related_cols_for(qname, df_cols):
    # if exact match in df_cols -> return [qname]
    if qname in df_cols:
        return [qname]
    # otherwise find columns starting with qname (prefix)
    pref = qname if qname.endswith("_") else qname + "_"
    matches = [c for c in df_cols if c.startswith(qname) or c.startswith(pref)]
    return matches if matches else ([qname] if qname in df_cols else [])

for _, r in rules_df.iterrows():
    q = r["Question"]
    checks = [x.strip() for x in str(r["Check_Type"]).split(";") if x.strip()]
    conds = [x.strip() for x in str(r["Condition"]).split(";")] if pd.notna(r["Condition"]) else ["" for _ in checks]
    related = related_cols_for(q, cols)
    # determine skip mask if skip present
    skip_mask = None
    if "Skip" in checks:
        idx = [i for i,ch in enumerate(checks) if ch=="Skip"][0]
        cond_text = conds[idx] if idx < len(conds) else ""
        # try to locate condition inside cond_text (may already be full "If A=1 then Q should be answered")
        # Extract condition part before 'then' if present
        if "then" in cond_text.lower():
            parts = re.split(r'(?i)then', cond_text, maxsplit=1)
            cond_part = parts[0]
        else:
            cond_part = cond_text
        skip_mask = get_condition_mask(cond_part, df)

        # apply skip: target columns defined by then-part if present
        # We'll just use related columns (since rules_df target is that q)
        for col in related:
            if col not in df.columns:
                report.append({id_col: None, "Question": q, "Check_Type": "Skip", "Issue": f"Target '{col}' not in data"})
                continue
            # blank definition: NaN or empty string only (literal "NA" considered answered)
            blank_mask = (df[col].isna()) | (df[col].astype(str).str.strip() == "")
            # but if value is "NA" or "na" or "Na" treat as answered (user wanted that)
            answered_but_na = df[col].astype(str).str.strip().str.upper() == "NA"
            blank_mask = blank_mask & (~answered_but_na)
            # offenders: skip_mask True & blank -> should be answered
            offenders = df.loc[skip_mask & blank_mask, id_col]
            for rid in offenders:
                report.append({id_col: rid, "Question": col, "Check_Type": "Skip", "Issue": "Blank but should be answered"})
            # also check reverse: NOT skip_mask & answered -> answered but should be blank
            offenders2 = df.loc[(~skip_mask) & (~blank_mask), id_col]
            for rid in offenders2:
                report.append({id_col: rid, "Question": col, "Check_Type": "Skip", "Issue": "Answered but should be blank"})

    # For other checks evaluate only for respondents who should answer
    rows_to_check = (~skip_mask) if skip_mask is not None else pd.Series(True, index=df.index)
    # However: if skip_mask is True meaning should answer, we should use skip_mask as rows_to_check
    if skip_mask is not None:
        rows_to_check = skip_mask

    for i, ch in enumerate(checks):
        if ch == "Skip":
            continue
        cond = conds[i] if i < len(conds) else ""
        if ch == "Range":
            # parse min-max
            # apply only on numeric-convertible values (coerce)
            m = re.search(r"(\d+)\s*[-to]+\s*(\d+)", cond)
            if m:
                minv, maxv = float(m.group(1)), float(m.group(2))
            else:
                # fallback: compute from data for first related col
                vals = pd.to_numeric(df[related[0]], errors="coerce")
                minv, maxv = float(vals.min()) if pd.notna(vals.min()) else None, float(vals.max()) if pd.notna(vals.max()) else None
            for col in related:
                if col not in df.columns:
                    report.append({id_col: None, "Question": col, "Check_Type": "Range", "Issue": "Column not found"})
                    continue
                if minv is None or maxv is None:
                    continue
                col_vals = pd.to_numeric(df[col], errors="coerce")
                # don't treat literal "NA" as missing; it will be NaN after to_numeric -> so treat specially:
                # we'll use original str check to exclude 'NA' from being considered blank
                bad_mask = ~col_vals.between(minv, maxv)
                # exclude blank (empty) values from range (they will be caught by Missing check instead)
                blank_mask_col = treat_as_blank(df[col]) & (~df[col].astype(str).str.upper().eq("NA"))
                offenders = df.loc[rows_to_check & bad_mask & (~blank_mask_col), id_col]
                for rid in offenders:
                    report.append({id_col: rid, "Question": col, "Check_Type": "Range", "Issue": f"Value out of range ({minv}-{maxv})"})
        elif ch == "Missing":
            for col in related:
                if col not in df.columns:
                    report.append({id_col: None, "Question": col, "Check_Type": "Missing", "Issue": "Column not found"})
                    continue
                # Blank definition: NaN or empty string; but literal "NA" is NOT blank per user request
                blank_mask_col = treat_as_blank(df[col]) & (~df[col].astype(str).str.upper().eq("NA"))
                offenders = df.loc[rows_to_check & blank_mask_col, id_col]
                for rid in offenders:
                    report.append({id_col: rid, "Question": col, "Check_Type": "Missing", "Issue": "Value is missing"})
        elif ch == "Straightliner":
            # If related >1 columns, look for same response across them (equal and not blank)
            # If only one column specified, try to expand by prefix
            cols_rel = related
            if len(cols_rel) == 1:
                # try expand by prefix
                base = cols_rel[0]
                possible = [c for c in cols if c.startswith(base + "_") or c.startswith(base + "r") or c.startswith(base+"_r")]
                if possible:
                    cols_rel = possible
            if len(cols_rel) > 1:
                same = df[cols_rel].nunique(axis=1) == 1
                # exclude cases where all are blank
                all_blank = df[cols_rel].apply(lambda row: all((str(x).strip()=="" or pd.isna(x)) for x in row), axis=1)
                offenders = df.loc[rows_to_check & same & (~all_blank), id_col]
                for rid in offenders:
                    report.append({id_col: rid, "Question": ",".join(cols_rel), "Check_Type": "Straightliner", "Issue": "Same response across all items"})
        elif ch == "Multi-Select":
            # related will be prefix name; expand
            cols_rel = related_cols_for(q, cols)
            if all(c in df.columns for c in cols_rel):
                summed = df[cols_rel].fillna(0).astype(float).sum(axis=1)
                offenders = df.loc[rows_to_check & (summed == 0), id_col]
                for rid in offenders:
                    report.append({id_col: rid, "Question": q, "Check_Type": "Multi-Select", "Issue": "No options selected"})
                # also check 0/1 validity
                for c in cols_rel:
                    badvals = ~df[c].astype(str).isin(["0","1","0.0","1.0","nan","NaN"])
                    offenders2 = df.loc[rows_to_check & badvals, id_col]
                    for rid in offenders2:
                        report.append({id_col: rid, "Question": c, "Check_Type": "Multi-Select", "Issue": "Invalid value (not 0/1)"})
        elif ch == "OpenEnd_Junk":
            for col in related:
                if col not in df.columns:
                    continue
                junk = df[col].astype(str).str.len() < 3
                offenders = df.loc[rows_to_check & junk, id_col]
                for rid in offenders:
                    report.append({id_col: rid, "Question": col, "Check_Type": "OpenEnd_Junk", "Issue": "Open-end looks like junk/low-effort"})
        # (Duplicate and other checks could be added similarly)

# --- Build final failed checks df ---
if report:
    failed_df = pd.DataFrame(report)
else:
    failed_df = pd.DataFrame(columns=[id_col, "Question", "Check_Type", "Issue"])

st.write("### Failed checks (first 200 rows)")
st.dataframe(failed_df.head(200))

# --- Download: rules + failed checks as Excel with two sheets ---
def to_excel_bytes(df_rules, df_failed):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_rules.to_excel(writer, index=False, sheet_name="Validation_Rules")
        df_failed.to_excel(writer, index=False, sheet_name="Failed_Checks")
    return out.getvalue()

excel_bytes = to_excel_bytes(rules_df, failed_df)

st.download_button(
    label="📥 Download generated Validation Rules + Failed Checks (Excel)",
    data=excel_bytes,
    file_name="validation_rules_and_failed_checks.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.success("Done — rules auto-generated from data + skips/constructed (if provided). You can review and edit the rules in the downloaded Excel.")
