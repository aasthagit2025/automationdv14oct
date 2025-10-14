# app.py
import streamlit as st
import pandas as pd
import pyreadstat
import io
import re
from typing import List, Dict

st.set_page_config(page_title="DV Rules Generator", layout="wide")
st.title("🔧 Auto-generate Validation Rules from Raw Data + Skips + Constructed Lists")

st.markdown(
    """
Upload these three files:
1. Raw Data (CSV / XLSX / SPSS `.sav`) — variable (column) order is preserved.
2. Skips file (CSV / XLSX) — should contain skip logic (a `Logic` or `Condition` column is fine).
3. Constructed List text (Sawtooth 'Print Study' `.txt`) — used for range extraction.
"""
)

# --- Uploads ---
col1, col2, col3 = st.columns(3)
with col1:
    data_file = st.file_uploader("1) Raw data (CSV / XLSX / SAV)", type=["csv", "xlsx", "sav"])
with col2:
    skips_file = st.file_uploader("2) Skips file (CSV / XLSX) — or export from Sawtooth", type=["csv", "xlsx"])
with col3:
    constructed_txt = st.file_uploader("3) Constructed list (Print Study .txt)", type=["txt"])

# helpers
def load_data(f):
    if f is None:
        return None
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f, encoding_errors="ignore")
    if name.endswith(".xlsx"):
        return pd.read_excel(f)
    if name.endswith(".sav"):
        df, meta = pyreadstat.read_sav(f)
        return df
    raise ValueError("Unsupported data file")

def load_skips(f):
    if f is None:
        return pd.DataFrame()
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f, encoding_errors="ignore")
    if name.endswith(".xlsx"):
        return pd.read_excel(f)
    return pd.DataFrame()

def parse_constructed_text(txt: str) -> Dict[str, List[str]]:
    """
    Parse constructed list text to map list names (or related question prefixes) -> range strings
    Returns mapping: canonical_prefix -> list of range strings (like "1-5")
    Heuristics:
    - Look for lines containing ADD(...,X,Y) or ADD(ParentListName(),1,5)
    - Use nearby 'List Name' or 'ForXXX' patterns to map
    """
    lines = txt.splitlines()
    mapping = {}  # prefix -> list of ranges
    current_list = None
    for i, ln in enumerate(lines):
        ln_strip = ln.strip()
        # detect "List Name: <name>"
        m = re.match(r"List Name:\s*(\S+)", ln_strip)
        if m:
            current_list = m.group(1).strip()
            continue
        # detect ADD(ParentListName(),a,b) or ADD(...,1,5)
        add_matches = re.findall(r"ADD\([^)]+\)", ln_strip, flags=re.IGNORECASE)
        for am in add_matches:
            # extract numbers
            nums = re.findall(r"(\d+)\s*,\s*(\d+)", am)
            if nums:
                a, b = nums[0]
                rng = f"{int(a)}-{int(b)}"
                if current_list:
                    mapping.setdefault(current_list, []).append(rng)
                else:
                    # also try to find nearest 'List Name' upwards
                    j = i-1
                    found = None
                    while j >= 0 and j > i-6:
                        mm = re.match(r"List Name:\s*(\S+)", lines[j].strip())
                        if mm:
                            found = mm.group(1)
                            break
                        j -= 1
                    if found:
                        mapping.setdefault(found, []).append(rng)
    return mapping

def extract_skip_conditions_from_df(skips_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Heuristic: find any column that looks like logic text, e.g. 'Logic', 'Condition', 'SkipCondition'
    Extract 'If ... then ...' strings and map target variable(s) (then part) -> list of full conditions text.
    """
    mapping = {}
    if skips_df is None or skips_df.shape[0] == 0:
        return mapping
    # find candidate text columns
    text_cols = [c for c in skips_df.columns if skips_df[c].dtype == object or skips_df[c].dtype == "string"]
    # find likely logic columns (names)
    candidates = [c for c in text_cols if re.search(r'logic|condition|if|then|skip', c, flags=re.IGNORECASE)]
    if not candidates:
        candidates = text_cols[:1]  # fallback
    for c in candidates:
        for val in skips_df[c].dropna().astype(str):
            # find if ... then ... (allow variations)
            m = re.search(r"(if\s+.+?\s+then\s+.+)", val, flags=re.IGNORECASE)
            if not m:
                # maybe the logic is the whole cell without explicit 'If'
                candidate = val.strip()
            else:
                candidate = m.group(1).strip()
            # extract then target(s)
            then_m = re.search(r"then\s+(.+)$", candidate, flags=re.IGNORECASE)
            if then_m:
                then_part = then_m.group(1).strip()
                # Normalize: remove trailing words like 'should be blank/answered' but keep text
                # Determine main variable token (first token that looks like variable)
                tok = then_part.split()[0]
                # Remove punctuation
                tok = tok.strip(" ,.;:")
                mapping.setdefault(tok, []).append(candidate)
            else:
                # no then found — ignore or store under unmapped
                mapping.setdefault("UNMAPPED", []).append(candidate)
    return mapping

def find_prefix_for_listname(listname: str) -> str:
    """
    Heuristic: For constructed list names like 'ForITQ1' or 'ITQ1ColList' return probable variable prefix like 'ITQ1' or 'ITQ1_'.
    """
    s = listname
    # common patterns
    s = re.sub(r'^(For|for|for_)?', '', s)
    s = re.sub(r'(ColList|RowList|List|List1|List\d+)$', '', s, flags=re.IGNORECASE)
    s = s.strip('_')
    return s

def build_rules_from_inputs(df: pd.DataFrame, skips_df: pd.DataFrame, constructed_txt: str) -> pd.DataFrame:
    # parse constructed lists
    constructed_map = {}
    if constructed_txt:
        constructed_map = parse_constructed_text(constructed_txt)
    # build mapping from prefix -> ranges
    prefix_range_map = {}
    for name, ranges in constructed_map.items():
        prefix = find_prefix_for_listname(name)
        if prefix:
            prefix_range_map.setdefault(prefix, []).extend(ranges)

    # parse skips
    skip_map_raw = extract_skip_conditions_from_df(skips_df)

    # now iterate columns in df order and assemble rules
    cols = list(df.columns)
    rules = []
    handled_multiselect_prefixes = set()

    # detect multi-select groups by common prefix (like Q2_1, Q2_2, Q2_3)
    prefix_groups = {}
    for c in cols:
        if '_' in c:
            p = c.split('_')[0] + '_'
            prefix_groups.setdefault(p, []).append(c)
    # also try prefixes that end with r1/r2 pattern: Q4r1? keep simpler with underscore

    for c in cols:
        # Skip respondent id columns in rule generation
        if c.lower() in ["respondentid", "password", "sys_respnum", "sys_resp_id"]:
            continue

        check_types = []
        conditions = []

        # 1) Multi-select group: if this column is the prefix base and group size>1, generate single Multi-Select rule
        ms_prefix = None
        for p, members in prefix_groups.items():
            if c == p.rstrip('_') or c.startswith(p):
                ms_prefix = p.rstrip('_')
                break

        if ms_prefix and ms_prefix + '_' in prefix_groups and ms_prefix not in handled_multiselect_prefixes:
            # create Multi-Select rule for this prefix
            handled_multiselect_prefixes.add(ms_prefix)
            check_types.append("Multi-Select")
            conditions.append("Only 0/1 allowed;At least one option should be selected")
            rules.append({"Question": ms_prefix, "Check_Type": ";".join(check_types), "Condition": ";".join(conditions)})
            continue

        # 2) Range from constructed lists: match prefix keys (exact or startswith)
        applied_range = None
        for pref, rngs in prefix_range_map.items():
            # exact match or prefix match ignoring underscores and case
            if c.lower().startswith(pref.lower()) or pref.lower().startswith(c.lower()) or pref.lower() in c.lower():
                # pick first range (if multiple, join with '|')
                applied_range = "|".join(sorted(set(rngs), key=lambda x: x))
                break
        if applied_range:
            check_types.append("Range")
            conditions.append(applied_range)

        # 3) Skip rules targeting this variable (try exact then prefix)
        skip_conditions_for_col = []
        # exact
        if c in skip_map_raw:
            skip_conditions_for_col.extend(skip_map_raw[c])
        # try prefix matches (e.g. target 'ITQ1' when col is 'ITQ1_r1')
        for target, conds in skip_map_raw.items():
            if target == "UNMAPPED":
                continue
            # map 'Intro1' to 'Intro1' etc, accept case-insensitive & prefix
            if c.lower().startswith(target.lower()) or target.lower().startswith(c.lower()) or ('_' in c and c.split('_')[0].lower() == target.lower()):
                skip_conditions_for_col.extend(conds)
        if skip_conditions_for_col:
            check_types.append("Skip")
            # dedupe and join with ' || ' between multiple skip conditions, keep full text
            cond_join = " || ".join(pd.unique(skip_conditions_for_col).tolist())
            conditions.append(cond_join)

        # 4) Missing: if variable seems single-select (small unique set) or numeric without range, then add Missing
        # Heuristics:
        col_ser = df[c]
        # identify open-text: many unique values and object dtype
        try:
            nunique = col_ser.nunique(dropna=True)
        except Exception:
            nunique = 0
        is_text = pd.api.types.is_string_dtype(col_ser) or pd.api.types.is_object_dtype(col_ser)
        is_numeric = pd.api.types.is_numeric_dtype(col_ser)
        # treat literal NA/N/A/None as answers — note: rules still say Missing when applicable; evaluator will treat those as answered
        # Determine single-select: small unique (<=20) and not long text
        if "Range" not in check_types and "Multi-Select" not in check_types:
            if (is_numeric and nunique > 0) or (is_text and nunique <= 20):
                # candidate for Missing
                check_types.append("Missing")
                conditions.append("Should not be blank (treat NA/N/A/NONE as answered)")

        # 5) If nothing found -> still include as Unmapped but with Missing (per requirement)
        if not check_types:
            check_types = ["Missing"]
            conditions = ["Should not be blank (treat NA/N/A/NONE as answered)"]

        # finalize row
        rules.append({"Question": c, "Check_Type": ";".join(check_types), "Condition": ";".join(conditions)})

    # preserve data order and return DataFrame
    rules_df = pd.DataFrame(rules)
    # keep only unique rows by Question (some prefix handling could introduce duplicates)
    rules_df = rules_df.drop_duplicates(subset=["Question"], keep="first").reset_index(drop=True)
    return rules_df

# --- UI flow ---
if st.button("Generate rules (preview)"):
    if data_file is None:
        st.error("Please upload Raw Data first.")
    else:
        try:
            df = load_data(data_file)
        except Exception as e:
            st.error(f"Cannot read Raw Data: {e}")
            st.stop()

        skips_df = load_skips(skips_file) if skips_file else pd.DataFrame()
        constructed_txt_val = None
        if constructed_txt:
            constructed_txt_val = constructed_txt.read().decode("utf-8", errors="ignore")

        with st.spinner("Generating rules..."):
            rules_df = build_rules_from_inputs(df, skips_df, constructed_txt_val)

        st.success(f"Generated {len(rules_df)} rules (in data order). You can edit them below.")
        st.write("Preview (editable):")
        # streamlit provides data_editor (editable)
        edited = st.data_editor(rules_df, num_rows="dynamic", use_container_width=True)
        st.markdown("---")

        # Download
        to_download = edited.copy()
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            to_download.to_excel(writer, sheet_name="validation_rules", index=False)
        st.download_button("📥 Download validation_rules.xlsx", data=buffer.getvalue(),
                           file_name="validation_rules.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Save session variables for later (if user wants to generate DV report)
        st.session_state["generated_rules"] = edited
        st.session_state["raw_data"] = df
        st.session_state["skips"] = skips_df

else:
    st.info("Click 'Generate rules (preview)' to produce validation rules from the uploaded files.")

# Helpful note about NA
st.markdown(
    """
**Notes**
- The generator treats literal values `NA`, `N/A`, `NONE`, `nan` (case-insensitive) in open-end fields as **answered** (not missing).  
- Multi-select groups are detected by suffix pattern like `Q2_1`, `Q2_2` and a single `Multi-Select` rule is produced for the prefix (e.g. `Q2`).  
- You can edit rules in the table then download. These rules are formatted as:
  - `Question | Check_Type` e.g. `ITQ1_r1 | Range;Skip`  
  - `Condition` matches (e.g. `1-5;If Segment_7=1 then ITQ1_r1 should be answered`)
"""
)
