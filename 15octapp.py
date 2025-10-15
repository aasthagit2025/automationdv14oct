import streamlit as st
import pandas as pd
import pyreadstat
import io
import re

st.set_page_config(layout="wide", page_title="Survey DV Tool (Auto rules + Preview)")
st.title("📊 Survey Data Validation Tool — Auto-generate rules & preview")

# -------------------------
# Helpers
# -------------------------
NA_VALUES = {"NA", "N/A", "NONE", "NAN", "NULL"}

def is_blank_series(s: pd.Series):
    """True where value is blank (NaN or empty string). Treat 'NA' etc as answered (not blank)."""
    s_str = s.astype(str).fillna("").str.strip()
    return s.isna() | (s_str == "")

def is_answered_series(s: pd.Series):
    """True where considered answered (not blank). Treat 'NA' strings as answered."""
    s_str = s.astype(str).fillna("").str.strip()
    return ~(s.isna() | (s_str == ""))

def find_id_col(df: pd.DataFrame):
    candidates = ["RespondentID", "Password", "RespID", "RID", "Sys_RespNum"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: first column if no standard id
    return df.columns[0]

def expand_prefix(prefix, cols):
    """If prefix ends with underscore or other convention, return matching columns."""
    prefix = str(prefix)
    # if prefix exactly exists, return it
    if prefix in cols:
        return [prefix]
    # if prefix ends with underscore or ends with no number then find startswith
    return [c for c in cols if c.startswith(prefix)]

def detect_multiselect_groups(cols):
    """
    Identify candidate multi-select groups by shared prefix with suffix like _1, _01, r1, _r1 etc.
    Returns dict prefix -> [cols...]
    """
    groups = {}
    for c in cols:
        m = re.match(r"^(.+?)([_\.]?r?_?\d+)$", c, flags=re.IGNORECASE)
        if m:
            prefix = m.group(1)
            groups.setdefault(prefix, []).append(c)
    # filter groups with more than 1 member
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}

def detect_rating_questions(df, sample_n=200):
    """
    Heuristic: numeric column, small integer domain (e.g., 1..5) and many distinct values <= 10 -> rating
    """
    rating_cols = []
    for c in df.columns:
        ser = df[c]
        if pd.api.types.is_numeric_dtype(ser) or ser.dropna().apply(lambda x: str(x).isdigit()).any():
            vals = pd.to_numeric(ser, errors="coerce").dropna().unique()
            if len(vals) > 0 and len(vals) <= 10:
                minv, maxv = vals.min(), vals.max()
                # typical rating ranges: 1..5,1..4,1..3,1..10
                if (minv >= 0 and maxv <= 10) or (minv >= 1 and maxv <= 10):
                    rating_cols.append(c)
    return rating_cols

def make_skip_mask(condition_text, df):
    """
    Parse a skip condition (the part after 'If' and before 'then') and return boolean mask.
    Supports AND/OR/NOT and operators =, !=, <>, <=, >=, <, >.
    If parsing fails, returns None.
    """
    if not isinstance(condition_text, str) or condition_text.strip() == "":
        return None
    text = condition_text.strip()
    if text.lower().startswith("if"):
        text = text[2:].strip()

    # support Not(...) wrapping at top-level
    try:
        # convert constructs like Not(A=1 or B=2) -> use regex to transform to python-evaluable conditions
        # We'll not eval Python; instead build mask using logic parsing similar to earlier functions
        or_groups = re.split(r'\s+or\s+', text, flags=re.IGNORECASE)
        mask = pd.Series(False, index=df.index)
        for or_group in or_groups:
            and_parts = re.split(r'\s+and\s+', or_group, flags=re.IGNORECASE)
            submask = pd.Series(True, index=df.index)
            for part in and_parts:
                p = part.strip()
                # handle Not( ... ) constructs inside
                neg = False
                if p.lower().startswith("not(") and p.endswith(")"):
                    neg = True
                    p = p[4:-1].strip()
                # normalize operators
                p = p.replace("<>", "!=")
                matched = False
                for op in ["<=", ">=", "!=", "<", ">", "="]:
                    if op in p:
                        col, val = [x.strip() for x in p.split(op, 1)]
                        if col not in df.columns:
                            # unknown column -> submask becomes False for all
                            submask &= False
                            matched = True
                            break
                        # numeric comparison if possible
                        col_vals = df[col]
                        # try numeric first
                        try:
                            val_num = float(val)
                            col_num = pd.to_numeric(col_vals, errors="coerce")
                            if op == "<=":
                                cur = col_num <= val_num
                            elif op == ">=":
                                cur = col_num >= val_num
                            elif op == "<":
                                cur = col_num < val_num
                            elif op == ">":
                                cur = col_num > val_num
                            elif op in ("=", "=="):
                                cur = col_num == val_num
                            else:
                                cur = pd.Series(False, index=df.index)
                        except ValueError:
                            # string compare (strip)
                            if op in ("!=", "!="):
                                cur = col_vals.astype(str).str.strip().str.upper() != val.strip().upper()
                            else:
                                cur = col_vals.astype(str).str.strip().str.upper() == val.strip().upper()
                        if neg:
                            cur = ~cur
                        submask &= cur.fillna(False)
                        matched = True
                        break
                if not matched:
                    # unknown format => set submask False
                    submask &= False
            mask |= submask
        return mask
    except Exception:
        return None

# -------------------------
# UI: Inputs
# -------------------------
st.sidebar.header("Step 1: Upload files")
data_file = st.sidebar.file_uploader("1) Upload survey data (CSV, XLSX, SAV)", type=["csv", "xlsx", "sav"])
skips_file = st.sidebar.file_uploader("2) Upload Skip logic (Excel - Skips sheet)", type=["xlsx"])
constructed_file = st.sidebar.file_uploader("3) (Optional) Constructed lists / PrintStudy (TXT)", type=["txt"])

run_button = st.sidebar.button("Generate rules & preview")

if not data_file or not skips_file:
    st.info("Upload both survey data and skip-logic Excel to proceed. Constructed lists (TXT) are optional.")
    st.stop()

# -------------------------
# Load data and skips
# -------------------------
# Data
try:
    if data_file.name.lower().endswith(".csv"):
        df = pd.read_csv(data_file, encoding_errors="ignore")
    elif data_file.name.lower().endswith(".xlsx"):
        df = pd.read_excel(data_file)
    elif data_file.name.lower().endswith(".sav"):
        df, meta = pyreadstat.read_sav(data_file)
    else:
        st.error("Unsupported data file type")
        st.stop()
except Exception as e:
    st.error(f"Failed reading data file: {e}")
    st.stop()

# Skips
try:
    skips_df = pd.read_excel(skips_file, sheet_name=None)  # read all sheets
    # prefer sheet named 'Skips' or take first sheet
    if "Skips" in skips_df:
        skips_sheet = skips_df["Skips"]
    else:
        # take first sheet
        skips_sheet = list(skips_df.values())[0]
except Exception as e:
    st.error(f"Failed reading skips file: {e}")
    st.stop()

# Constructed lists (optional)
constructed_text = None
if constructed_file:
    try:
        constructed_text = constructed_file.read().decode("utf-8", errors="ignore")
    except Exception:
        try:
            constructed_text = constructed_file.getvalue().decode("utf-8", errors="ignore")
        except Exception:
            constructed_text = None

id_col = find_id_col(df)

# -------------------------
# Auto-generate validation rules
# -------------------------
def generate_rules(df, skips_sheet, constructed_text=None):
    cols = list(df.columns)
    ms_groups = detect_multiselect_groups(cols)  # prefix -> members
    rating_cols = set(detect_rating_questions(df))
    rules = []  # list of dicts: Question, Check_Type, Condition, Source

    # 1) iterate columns in data order and propose rules
    for col in cols:
        # skip respondent id column
        if col == id_col:
            continue

        # if part of a multi-select group, generate a single Multi-Select rule keyed by prefix if not yet added
        ms_prefix = None
        for prefix, members in ms_groups.items():
            if col in members:
                ms_prefix = prefix
                break

        if ms_prefix:
            # add multi-select rule with source "Auto (multi-select)"
            if not any(r["Question"] == ms_prefix and r["Check_Type"] == "Multi-Select" for r in rules):
                rules.append({
                    "Question": ms_prefix,
                    "Check_Type": "Multi-Select",
                    "Condition": "Only 0/1; At least one selected",
                    "Source": "Auto (multiselect detected)"
                })
            continue  # don't add per-item rules here (we will still check per member in validation)

        # rating/range
        if col in rating_cols:
            # detect likely range (min..max)
            try:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals) > 0:
                    minv, maxv = int(vals.min()), int(vals.max())
                    cond = f"{minv}-{maxv}"
                else:
                    cond = ""
            except Exception:
                cond = ""
            rules.append({"Question": col, "Check_Type": "Range;Straightliner", "Condition": f"{cond};Group({col}_prefix?)", "Source": "Auto (rating)"})
            continue

        # single-select (low cardinality)
        nunq = df[col].dropna().astype(str).str.strip().nunique()
        if nunq > 0 and nunq <= 10 and not pd.api.types.is_float_dtype(df[col]):
            rules.append({"Question": col, "Check_Type": "Missing", "Condition": "", "Source": "Auto (single-select)"})
            continue

        # open-end text
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            rules.append({"Question": col, "Check_Type": "Missing;OpenEnd_Junk", "Condition": ";MinLen(3)", "Source": "Auto (open-end/text)"})
            continue

        # fallback numeric -> Range
        rules.append({"Question": col, "Check_Type": "Range", "Condition": "", "Source": "Auto (numeric)"})

    # 2) incorporate skip rules from skips_sheet
    # we expect skips_sheet to contain columns: Skip From / Logic / Skip To OR similar. We'll search for text patterns.
    # Try common columns names
    common_cols = [c.lower() for c in skips_sheet.columns]
    logic_col = None
    target_col = None
    for candidate in ["logic", "skip logic", "logic ", "logic_text", "rule", "logicstring"]:
        if candidate in common_cols:
            idx = common_cols.index(candidate)
            logic_col = skips_sheet.columns[idx]
            break
    for candidate in ["skip to", "skipto", "skip_to", "skipto", "skip to ", "skip_to "]:
        if candidate in common_cols:
            idx = common_cols.index(candidate)
            target_col = skips_sheet.columns[idx]
            break
    # fallback try "Logic" and "Skip To"
    if "Logic" in skips_sheet.columns:
        logic_col = "Logic"
    if "Skip To" in skips_sheet.columns:
        target_col = "Skip To"
    # If none found, try to use any column that contains 'If' text
    if logic_col is None:
        for c in skips_sheet.columns:
            if skips_sheet[c].astype(str).str.contains("If", case=False, na=False).any():
                logic_col = c
                break

    if logic_col:
        for _, row in skips_sheet.iterrows():
            logic_text = str(row.get(logic_col, "")).strip()
            # attempt to extract then target variable(s)
            # common pattern: If <condition> then <target> should be blank/answered/Next Question...
            if "then" in logic_text.lower():
                parts = re.split(r'(?i)then', logic_text, maxsplit=1)
                if_part = parts[0].strip()
                then_part = parts[1].strip()
                # extract first token as target variable (may be "Next Question" or actual var)
                then_token = then_part.split()[0]
                # if token looks like "Next" or "Next Question", attempt to map via Skip From
                # we will simply add a Skip rule for the target token if it's a variable name or if Skip To column exists
                if target_col and pd.notna(row.get(target_col)):
                    target_var = str(row.get(target_col)).strip()
                else:
                    target_var = then_token
                # only add if target_var likely refers to variable (present in df or as prefix)
                # allow prefixes ending with underscore to match groups
                if target_var and (target_var in df.columns or any(c.startswith(target_var) for c in df.columns) or target_var.endswith("_")):
                    # find existing rule entry for that question; append Skip if exists, else add
                    found = False
                    for r in rules:
                        if r["Question"] == target_var or (target_var.endswith("_") and r["Question"] == target_var):
                            # append Skip into Check_Type if not already present
                            if "Skip" not in r["Check_Type"]:
                                r["Check_Type"] = r["Check_Type"] + ";" + "Skip"
                                r["Condition"] = (r.get("Condition","") + ";" + logic_text).strip(";")
                            found = True
                            break
                    if not found:
                        rules.append({"Question": target_var, "Check_Type": "Skip", "Condition": logic_text, "Source": "Skips file"})
    # 3) de-duplicate and keep data order
    # create dict keyed by question keeping the first occurrence but merging Check_Type/Condition when needed
    final_rules = []
    seen = {}
    for r in rules:
        q = r["Question"]
        if q in seen:
            # merge
            prev = seen[q]
            # merge check types
            prev_types = set([x.strip() for x in prev["Check_Type"].split(";") if x.strip()]) if prev["Check_Type"] else set()
            new_types = set([x.strip() for x in r.get("Check_Type","").split(";") if x.strip()])
            merged_types = ";".join(sorted(prev_types.union(new_types), key=lambda x: x))
            prev["Check_Type"] = merged_types
            # merge conditions concatenating with semicolon (preserve source)
            prev["Condition"] = ";".join([c for c in [prev.get("Condition",""), r.get("Condition","")] if c])
            prev["Source"] = prev.get("Source","") + "|" + r.get("Source","")
        else:
            seen[q] = r

    # keep in data order: iterate df columns & then any remaining
    ordered = []
    added = set()
    for c in df.columns:
        if c == id_col:
            continue
        if c in seen:
            ordered.append(seen[c])
            added.add(c)
        else:
            # also check prefixes: if any seen rule matches prefix
            for q in list(seen.keys()):
                if q.endswith("_") and c.startswith(q) and q not in added:
                    ordered.append(seen[q])
                    added.add(q)
    # add remaining rules (constructed lists etc.)
    for q, r in seen.items():
        if q not in added:
            ordered.append(r)
            added.add(q)

    # ensure Question column name consistent
    for r in ordered:
        if "Question" not in r:
            r["Question"] = r.get("Question", "")
        if "Check_Type" not in r:
            r["Check_Type"] = r.get("Check_Type", "")
        if "Condition" not in r:
            r["Condition"] = r.get("Condition", "")
        if "Source" not in r:
            r["Source"] = r.get("Source", "")

    return pd.DataFrame(ordered)

# -------------------------
# Generate & preview rules
# -------------------------
if run_button:
    with st.spinner("Generating rules..."):
        rules_df = generate_rules(df, skips_sheet, constructed_text)
    st.success(f"Generated {len(rules_df)} rules (preview below). Verify and click 'Run Validation' to execute checks.")
    st.subheader("Preview: Auto-generated validation rules (you can download this file)")
    st.dataframe(rules_df.head(500))
    # download
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        rules_df.to_excel(writer, index=False, sheet_name="validation_rules")
    st.download_button("Download validation_rules.xlsx", data=buf.getvalue(),
                       file_name="validation_rules.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Allow user to run validation now
    if st.button("Run validation with these generated rules"):
        st.info("Running validation — only failed checks will be shown.")
        # perform validation using the generated rules
        report = []
        skip_excluded_ids = set()  # respondents to exclude from Range/Missing when skip says should be blank
        for _, rule in rules_df.iterrows():
            q_raw = str(rule["Question"]).strip()
            check_types = [x.strip() for x in str(rule.get("Check_Type","")).split(";") if x.strip()]
            cond_text = str(rule.get("Condition","")).strip()

            # For multiselect rules, the Question likely is a prefix. Expand to actual columns
            related_cols = []
            if q_raw in df.columns:
                related_cols = [q_raw]
            else:
                # try prefix match (endswith underscore), else startswith
                possible = [c for c in df.columns if c.startswith(q_raw)]
                if possible:
                    related_cols = possible
                else:
                    # if question ends with underscore and exists as key in ms detection treat prefix
                    if q_raw.endswith("_"):
                        related_cols = [c for c in df.columns if c.startswith(q_raw)]
            # if still empty, skip but report "Question not found"
            if not related_cols:
                report.append({id_col: None, "Question": q_raw, "Check_Type": ";".join(check_types), "Issue": "Question not found in dataset"})
                continue

            # Process Skip first if present (to set exclusion mask)
            if "Skip" in check_types:
                # cond_text probably contains the skip expression, maybe many semicolon-separated; pick the part containing 'If'
                skip_cond = None
                # sometimes condition contains many pieces; try to find the semicolon segment that contains 'If' or 'then'
                for part in cond_text.split(";"):
                    if "if" in part.lower() or "then" in part.lower():
                        skip_cond = part.strip()
                        break
                if skip_cond is None:
                    skip_cond = cond_text
                # parse skip: split 'then' part
                if "then" not in skip_cond.lower():
                    # invalid skip expression; report and continue
                    report.append({id_col: None, "Question": q_raw, "Check_Type": "Skip", "Issue": f"Invalid skip format ({skip_cond})"})
                else:
                    parts = re.split(r'(?i)then', skip_cond, maxsplit=1)
                    if_part = parts[0].strip()
                    then_part = parts[1].strip()
                    mask = make_skip_mask(if_part, df)
                    if mask is None:
                        report.append({id_col: None, "Question": q_raw, "Check_Type": "Skip", "Issue": f"Unable to parse skip condition ({if_part})"})
                    else:
                        # determine target columns from then_part: may say "Q3 should be blank" or "Q3_1 to Q3_5" or "Next Question"
                        then_tokens = then_part.split()
                        # detect range "Q3_1 to Q3_5"
                        if "to" in then_part.lower():
                            target_cols = []
                            # find token like X to Y
                            m = re.search(r'([A-Za-z0-9_]+)\s+to\s+([A-Za-z0-9_]+)', then_part, flags=re.IGNORECASE)
                            if m:
                                start, end = m.group(1), m.group(2)
                                # expand numeric suffix
                                base1 = re.match(r"(.+?)(\d+)$", start)
                                base2 = re.match(r"(.+?)(\d+)$", end)
                                if base1 and base2 and base1.group(1) == base2.group(1):
                                    prefix = base1.group(1)
                                    for i in range(int(base1.group(2)), int(base2.group(2))+1):
                                        cname = f"{prefix}{i}"
                                        if cname in df.columns:
                                            target_cols.append(cname)
                        else:
                            token = then_tokens[0]
                            if token.lower() in ["next", "nextquestion", "nextquestion", "next_question"]:
                                # best-effort: treat next in list -> find the column that comes after the skip-from column if provided
                                # skip this sophisticated mapping for now
                                target_cols = []
                            else:
                                # token may be prefix, exact var or var with underscore; expand
                                if token in df.columns:
                                    target_cols = [token]
                                else:
                                    # prefix match
                                    target_cols = [c for c in df.columns if c.startswith(token)]
                        if not target_cols:
                            # fallback: apply skip rule to related_cols (the question we generated rule for)
                            target_cols = related_cols

                        # then check for "blank" vs "answered" in then_part
                        should_be_blank = "blank" in then_part.lower() or "should be blank" in then_part.lower()
                        for col in target_cols:
                            blank_mask = is_blank_series(df[col])
                            answered_mask = is_answered_series(df[col])
                            # if should be blank -> flag answered rows where mask True
                            if should_be_blank:
                                offenders = df.loc[mask & answered_mask, id_col]
                                for rid in offenders:
                                    report.append({id_col: rid, "Question": col, "Check_Type": "Skip", "Issue": "Answered but should be blank"})
                                # mark those who satisfy the skip (blank) to exclude from Range/Missing later
                                satisfied = df.loc[mask & blank_mask, id_col].tolist()
                                skip_excluded_ids.update(satisfied)
                            else:
                                # should be answered -> flag blanks when mask True
                                offenders = df.loc[mask & ~is_answered_series(df[col]), id_col]
                                for rid in offenders:
                                    report.append({id_col: rid, "Question": col, "Check_Type": "Skip", "Issue": "Blank but should be answered"})
                                # those who answered when mask True are fine; nothing to exclude
            # END Skip handling

            # For other checks, apply only to respondents who should answer.
            # Build rows_to_check mask: exclude those where skip_excluded_ids indicate they should be blank
            rows_to_check = ~df[id_col].isin(skip_excluded_ids)

            # Range
            if "Range" in check_types:
                # cond_text may include many segments; pick numeric range like '1-5' or '1 to 5'
                range_part = None
                for part in cond_text.split(";"):
                    if re.search(r'\d+\s*[-to]\s*\d+', part):
                        range_part = part.strip()
                        break
                if range_part is None:
                    # maybe default no range specified -> skip
                    for col in related_cols:
                        report.append({id_col: None, "Question": col, "Check_Type": "Range", "Issue": "No range specified"})
                else:
                    # normalize "to" to "-"
                    rp = range_part.replace("to", "-").replace(" ", "")
                    try:
                        minv, maxv = [float(x) for x in re.split(r'[-]', rp)[:2]]
                        for col in related_cols:
                            col_num = pd.to_numeric(df[col], errors="coerce")
                            # only check rows_to_check
                            bad_mask = rows_to_check & ~col_num.between(minv, maxv)
                            # exclude blanks: if blank but not answered, treat as missing vs out-of-range will be handled in Missing check
                            bad_mask = bad_mask & is_answered_series(df[col])
                            offenders = df.loc[bad_mask, id_col]
                            for rid in offenders:
                                report.append({id_col: rid, "Question": col, "Check_Type": "Range", "Issue": f"Value out of range ({minv}-{maxv})"})
                    except Exception:
                        for col in related_cols:
                            report.append({id_col: None, "Question": col, "Check_Type": "Range", "Issue": f"Invalid range ({range_part})"})

            # Missing
            if "Missing" in check_types:
                for col in related_cols:
                    # blank definition: NaN or empty string only. Strings 'NA', 'N/A', 'NONE' are treated as answered.
                    blank_mask = is_blank_series(df[col])
                    offenders = df.loc[rows_to_check & blank_mask, id_col]
                    for rid in offenders:
                        report.append({id_col: rid, "Question": col, "Check_Type": "Missing", "Issue": "Value is missing"})

            # Multi-Select
            if "Multi-Select" in check_types:
                # Question often is prefix; expand to actual columns
                ms_cols = []
                for rc in related_cols:
                    ms_cols.extend([c for c in df.columns if c.startswith(rc)])
                ms_cols = sorted(set(ms_cols))
                # check each multi-select member for invalid values (not 0/1) and overall none selected
                for col in ms_cols:
                    offenders = df.loc[rows_to_check & (~df[col].isin([0,1]) & ~df[col].isin(["0","1"])) , id_col]
                    for rid in offenders:
                        report.append({id_col: rid, "Question": col, "Check_Type": "Multi-Select", "Issue": "Invalid value (not 0/1)"})
                if ms_cols:
                    none_selected = df.loc[rows_to_check & (df[ms_cols].fillna(0).sum(axis=1) == 0), id_col]
                    for rid in none_selected:
                        report.append({id_col: rid, "Question": ";".join(ms_cols), "Check_Type": "Multi-Select", "Issue": "No options selected"})

            # Straightliner
            if "Straightliner" in check_types:
                # if user gave a single column name, try to expand by prefix
                sc = related_cols
                if len(sc) == 1:
                    sc = [c for c in df.columns if c.startswith(sc[0])]
                if len(sc) > 1:
                    same_resp = df[sc].nunique(axis=1) == 1
                    offenders = df.loc[rows_to_check & same_resp, id_col]
                    for rid in offenders:
                        report.append({id_col: rid, "Question": ",".join(sc), "Check_Type": "Straightliner", "Issue": "Same response across all items"})

            # OpenEnd_Junk
            if "OpenEnd_Junk" in check_types:
                for col in related_cols:
                    # treat very short responses (<3) as junk, but ignore "NA" etc - because is_answered_series treats 'NA' as answered.
                    s = df[col].astype(str).fillna("").str.strip()
                    junk_mask = rows_to_check & (s.str.len() < 3) & (~s.str.upper().isin(NA_VALUES))
                    offenders = df.loc[junk_mask, id_col]
                    for rid in offenders:
                        report.append({id_col: rid, "Question": col, "Check_Type": "OpenEnd_Junk", "Issue": "Open-end looks like junk/low-effort"})

            # Duplicate
            if "Duplicate" in check_types:
                for col in related_cols:
                    dupes = df.loc[rows_to_check & df.duplicated(subset=[col], keep=False), id_col]
                    for rid in dupes:
                        report.append({id_col: rid, "Question": col, "Check_Type": "Duplicate", "Issue": "Duplicate value found"})

        # After loop end
        report_df = pd.DataFrame(report)
        if report_df.empty:
            st.success("Validation completed — no failures found.")
        else:
            st.success(f"Validation completed — {len(report_df)} failed checks found.")
            st.dataframe(report_df)
            # download report
            outbuf = io.BytesIO()
            with pd.ExcelWriter(outbuf, engine="openpyxl") as writer:
                report_df.to_excel(writer, index=False, sheet_name="DV_Report")
            st.download_button("Download DV_Report.xlsx", data=outbuf.getvalue(), file_name="DV_Report.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

