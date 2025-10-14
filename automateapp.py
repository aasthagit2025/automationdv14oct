# app.py
import streamlit as st
import pandas as pd
import pyreadstat
import io
import re
from typing import List, Dict

st.set_page_config(layout="wide")
st.title("📊 DV Tool — Rule Generator + Validator (Combined)")

# -------------------------
# Helpers
# -------------------------
def read_data_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded, encoding_errors="ignore")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    if name.endswith(".sav"):
        df, meta = pyreadstat.read_sav(uploaded)
        return df
    raise ValueError("Unsupported data file type")

def is_literal_na(x):
    """Return True if value is the literal string 'NA' (case-insensitive)."""
    try:
        return str(x).strip().upper() == "NA"
    except Exception:
        return False

def is_blank_series(s: pd.Series):
    """Blank = NaN or empty string. 'NA' (literal) is NOT blank."""
    return s.isna() | (s.astype(str).str.strip() == "")

def expand_prefix(prefix: str, cols: List[str]) -> List[str]:
    return [c for c in cols if c.startswith(prefix)]

def expand_range_expr(expr: str, cols: List[str]) -> List[str]:
    # support "Q3_1 to Q3_13" or "Q3_1-Q3_13" patterns
    expr = expr.strip()
    if "to" in expr.lower():
        a, b = re.split(r'\s+to\s+', expr, flags=re.I)
    elif "-" in expr and re.search(r'\d+\s*-\s*\d+', expr):
        a, b = [p.strip() for p in expr.split("-", 1)]
    else:
        return [expr] if expr in cols else []
    m1 = re.match(r"([A-Za-z0-9_]+?)(\d+)$", a.strip())
    m2 = re.match(r"([A-Za-z0-9_]+?)(\d+)$", b.strip())
    if m1 and m2 and m1.group(1) == m2.group(1):
        prefix = m1.group(1)
        start, end = int(m1.group(2)), int(m2.group(2))
        return [f"{prefix}{i}" for i in range(start, end+1) if f"{prefix}{i}" in cols]
    return [expr] if expr in cols else []

def parse_skip_logic_text(cond_text: str, df: pd.DataFrame) -> pd.Series:
    """
    Parse conditions like:
      If Q4_r1<=7
      If Not(Q4_r1=1 or Q4_r1=2)
      If Q2_1=1 and Q2_2=0 or Q5=3
    Return boolean mask of respondents where condition is True.
    """
    if cond_text is None:
        return pd.Series(True, index=df.index)
    text = cond_text.strip()
    if text.lower().startswith("if"):
        text = text[2:].strip()
    # Normalize Not(...) -> we will treat 'Not(' as wrapper
    # We'll split top-level OR groups then AND parts
    or_groups = re.split(r'\s+or\s+', text, flags=re.I)
    mask = pd.Series(False, index=df.index)
    for or_grp in or_groups:
        and_parts = re.split(r'\s+and\s+', or_grp, flags=re.I)
        sub_mask = pd.Series(True, index=df.index)
        for part in and_parts:
            part = part.strip()
            # handle Not(...) wrapper
            if part.lower().startswith("not(") and part.endswith(")"):
                inner = part[4:-1].strip()
                inner_mask = parse_skip_logic_text("If " + inner, df)
                sub_mask &= ~inner_mask
                continue
            # handle `( ... )` around expression
            if part.startswith("(") and part.endswith(")"):
                inner = part[1:-1].strip()
                sub_mask &= parse_skip_logic_text("If " + inner, df)
                continue
            # normalize != operators
            part = part.replace("<>", "!=")
            matched = False
            for op in ["<=", ">=", "!=", "<", ">", "="]:
                if op in part:
                    col, val = [p.strip() for p in part.split(op, 1)]
                    if col not in df.columns:
                        # if column doesn't exist, treat expression as False for all rows
                        sub_mask &= False
                        matched = True
                        break
                    # numeric compare if possible
                    if op in ["<=", ">=", "<", ">"]:
                        col_vals = pd.to_numeric(df[col], errors="coerce")
                        try:
                            vnum = float(val)
                        except:
                            vnum = float('nan')
                        if op == "<=":
                            sub_mask &= col_vals <= vnum
                        elif op == ">=":
                            sub_mask &= col_vals >= vnum
                        elif op == "<":
                            sub_mask &= col_vals < vnum
                        elif op == ">":
                            sub_mask &= col_vals > vnum
                    elif op in ["!=", "<>"]:
                        sub_mask &= df[col].astype(str).str.strip() != str(val)
                    elif op == "=":
                        sub_mask &= df[col].astype(str).str.strip() == str(val)
                    matched = True
                    break
            if not matched:
                # unsupported fragment -> be conservative (False)
                sub_mask &= False
        mask |= sub_mask
    return mask

# -------------------------
# UI: Generator & Validator Tabs
# -------------------------
tab = st.tabs(["1 - Generate Rules", "2 - Validate Data"])[0]

with st.sidebar:
    st.markdown("### Quick settings")
    treat_literal_na_as_answer = st.checkbox("Treat literal 'NA' as answered (not blank)", value=True)
    auto_rating_range_force_1_5 = st.checkbox("Force rating checks to 1-5 when ambiguous", value=True)
    show_preview_limit = st.number_input("Rules preview rows limit", min_value=5, max_value=500, value=50)

# -------------------------
# Tab 1: Generate rules
# -------------------------
with st.container():
    st.header("1 — Rule Generator (Skips + Constructed list -> Validation rules)")
    st.info("Upload your survey dataset and the Sawtooth Constructed List text export and Skips Excel/CSV. The tool will produce a rules Excel.")

    col1, col2, col3 = st.columns(3)
    with col1:
        upload_data_for_gen = st.file_uploader("Upload data file (for generating rules) — CSV/XLSX/SAV", key="gen_data")
    with col2:
        upload_skips = st.file_uploader("Upload Skips file (CSV/XLSX) — optional", key="skips")
    with col3:
        upload_constructed_txt = st.file_uploader("Upload Constructed List export (text file) — optional", key="constructed")

    if st.button("Generate rules from inputs"):
        if not upload_data_for_gen:
            st.error("Please upload data file first.")
        else:
            try:
                df_gen = read_data_file(upload_data_for_gen)
            except Exception as e:
                st.error(f"Error reading data file: {e}")
                df_gen = None

            if df_gen is not None:
                cols = list(df_gen.columns)
                # identify respondent id
                id_candidates = [c for c in ["RespondentID", "Password", "RespID", "RID", "Sys_RespNum"] if c in cols]
                id_col = id_candidates[0] if id_candidates else None

                # start building rules list (rows as dicts)
                rules_rows = []

                # 1) Use Skips (if provided)
                skips_df = None
                if upload_skips is not None:
                    try:
                        if upload_skips.name.lower().endswith(".csv"):
                            skips_df = pd.read_csv(upload_skips)
                        else:
                            skips_df = pd.read_excel(upload_skips)
                        st.success("Skips file loaded")
                    except Exception as e:
                        st.warning(f"Could not load skips file: {e}")

                if skips_df is not None:
                    # Expecting columns: Skip From, Skip Type, Always Skip, Logic, Skip To  (but be flexible)
                    # We'll read rows where Skip To refers to an actual variable in data (or Next Question -> skip_to is inferred)
                    for _, r in skips_df.iterrows():
                        # get skip logic text
                        logic = str(r.get("Logic", r.get("Condition", r.get("Logic/Condition", "")))).strip()
                        skip_from = str(r.get("Skip From", r.get("From", ""))).strip()
                        skip_to = str(r.get("Skip To", r.get("Skip To", ""))).strip()
                        # If skip_to is like "Next Question" we attempt to map later (skip)
                        # We'll output skip rules for skip_to when skip_to is present in data columns or if skip_from in data -> create rule for skip_to candidate
                        if skip_to and skip_to in cols:
                            rules_rows.append({"Question": skip_to, "Check_Type": "Skip", "Condition": logic, "Source": "Skips"})
                        else:
                            # Try to map skip_to to variable in data with same name (common case)
                            if skip_from in cols and skip_to in cols:
                                rules_rows.append({"Question": skip_to, "Check_Type": "Skip", "Condition": logic, "Source": "Skips"})
                            else:
                                # keep as constructed skip for manual review
                                rules_rows.append({"Question": skip_to or skip_from, "Check_Type": "Skip", "Condition": logic, "Source": "Skips (unmapped)"})

                # 2) Constructed list parsing (optional)
                if upload_constructed_txt is not None:
                    try:
                        raw_text = upload_constructed_txt.read().decode("utf-8", errors="ignore") if isinstance(upload_constructed_txt.read(), bytes) else upload_constructed_txt.getvalue()
                    except Exception:
                        upload_constructed_txt.seek(0)
                        raw_text = upload_constructed_txt.read()
                    # Find blocks for constructed lists that mention ADD(ParentListName(),x,y) or logic lines mapping to a parent
                    # We'll add Range rules for constructed lists that map to rating scales (1-5)
                    constructed_blocks = re.split(r"={10,}", raw_text)
                    # simple heuristic: any "Add(ParentListName(),1,5)" -> create Range 1-5 for related variables if present
                    if "Add(ParentListName(),1,5)" in raw_text or re.search(r"Add\(ParentListName\(\),\s*1,\s*5\)", raw_text, flags=re.I):
                        # find likely prefixes in data that match constructed names -> we won't over-guess; we'll add Constructed rules as separate rows for review
                        rules_rows.append({"Question": "ConstructedLists", "Check_Type": "Constructed", "Condition": "Add(ParentListName(),1,5)", "Source": "ConstructedText"})
                    st.success("Constructed text processed (heuristic)")

                # 3) Data-driven default rules
                # - For numeric columns: add Range check: min-max with some guard (if many distinct values, leave to user; we use common heuristics)
                for col in cols:
                    # skip id col
                    if id_col and col == id_col:
                        continue
                    ser = df_gen[col]
                    # detect multi-select pattern: if there are sibling columns with same prefix 'QX_' then treat as multi-select parent
                    m = re.match(r"^([A-Za-z0-9_]+?_)\d+$", col)
                    if m:
                        prefix = m.group(1)
                        # we'll add multi-select rule once per prefix (only once)
                        if not any(r["Question"] == prefix for r in rules_rows):
                            # check numeric nature 0/1 presence
                            related = expand_prefix(prefix, cols)
                            # if at least 2 related columns -> multi-select
                            if len(related) >= 2:
                                rules_rows.append({"Question": prefix, "Check_Type": "Multi-Select", "Condition": "only01;atleast1", "Source": "DataDetection"})
                        continue

                    # rating detection: small integer unique values 1..5/1..7 - add straightliner for these group prefixes
                    if pd.api.types.is_integer_dtype(ser) or pd.api.types.is_float_dtype(ser) or ser.dropna().apply(lambda x: str(x).isdigit()).all():
                        uniq = sorted(set([int(x) for x in ser.dropna().astype(str).unique() if str(x).strip().lstrip("-").isdigit()])) if len(ser.dropna())>0 else []
                        # If values mostly between 1 and 5 and count distinct >=3 assume rating
                        if len(uniq) >= 3 and min(uniq) >= 0 and max(uniq) <= 7 and max(uniq) <= 7:
                            # add range rule
                            # If ambiguous and force_1_5 is on and 5 in uniq -> set 1-5 else set actual min-max
                            if auto_rating_range_force_1_5 and (1 in uniq and 5 in uniq):
                                rules_rows.append({"Question": col, "Check_Type": "Range;Straightliner", "Condition": "1-5;", "Source": "AutoDetectedRating"})
                            else:
                                rules_rows.append({"Question": col, "Check_Type": "Range;Straightliner", "Condition": f"{min(uniq)}-{max(uniq)};", "Source": "AutoDetectedRating"})
                            continue

                    # for non-multi single select (categorical) add Missing check by default
                    # Heuristic: if string/object dtype or small unique count -> treat as single-select / categorical => missing check
                    if pd.api.types.is_object_dtype(ser) or ser.nunique(dropna=True) <= 20:
                        rules_rows.append({"Question": col, "Check_Type": "Missing", "Condition": "", "Source": "DefaultMissing"})

                # Deduplicate rules_rows preserving order
                seen = set()
                final_rows = []
                for r in rules_rows:
                    key = (r.get("Question"), r.get("Check_Type"), r.get("Condition"))
                    if key not in seen:
                        final_rows.append(r)
                        seen.add(key)
                rules_df_out = pd.DataFrame(final_rows)
                if rules_df_out.empty:
                    st.warning("No rules generated — check inputs.")
                else:
                    st.success(f"Generated {len(rules_df_out)} rules")
                    st.dataframe(rules_df_out.head(show_preview_limit))
                    # prepare Excel download
                    out = io.BytesIO()
                    with pd.ExcelWriter(out, engine="openpyxl") as writer:
                        rules_df_out.to_excel(writer, index=False, sheet_name="validation_rules")
                        # put a brief readme sheet
                        pd.DataFrame([{"Note":"Rules generated heuristically. Review before validating."}]).to_excel(writer, index=False, sheet_name="README")
                    st.download_button("Download generated rules.xlsx", data=out.getvalue(), file_name="validation_rules.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------------
# Tab 2: Validator
# -------------------------
st.markdown("---")
st.header("2 — Validate Data (use generated rules or upload your own)")

colA, colB = st.columns(2)
with colA:
    data_file = st.file_uploader("Upload survey data to validate (CSV/XLSX/SAV)", key="val_data")
with colB:
    rules_file = st.file_uploader("Upload validation rules (Excel) — optional (if omitted, use generator)", key="val_rules")

if data_file:
    try:
        df = read_data_file(data_file)
    except Exception as e:
        st.error(f"Could not read data file: {e}")
        df = None

    if df is not None:
        cols = list(df.columns)
        # choose respondent id col (flexible)
        id_candidates = [c for c in ["RespondentID", "Password", "RespID", "RID", "Sys_RespNum"] if c in cols]
        id_col = st.selectbox("Respondent ID column", options=(id_candidates + ["<none>"]), index=0 if id_candidates else 0)
        if id_col == "<none>":
            st.warning("No RespondentID selected — report will use row index.")
            df["_row_index_for_dv"] = df.index.astype(str)
            id_col = "_row_index_for_dv"

        # load rules
        if rules_file is not None:
            try:
                rules_df = pd.read_excel(rules_file)
                st.success("Rules loaded from uploaded file")
            except Exception as e:
                st.error(f"Could not read rules file: {e}")
                rules_df = None
        else:
            st.info("No rules file uploaded: attempting to auto-generate rules (same heuristics as generator).")
            # very light auto-generation: missing for non-id, range for numeric (1-99999 if numeric wide)
            nodes = []
            for c in cols:
                if c == id_col:
                    continue
                ser = df[c]
                if re.match(r"^[A-Za-z0-9_]+?_\d+$", c):
                    prefix = re.match(r"^([A-Za-z0-9_]+?_)\d+$", c).group(1)
                    # add once
                    if not any(n["Question"] == prefix for n in nodes):
                        # find all related
                        related = expand_prefix(prefix, cols)
                        if len(related) >= 2:
                            nodes.append({"Question": prefix, "Check_Type": "Multi-Select", "Condition": "only01;atleast1", "Source": "Auto"})
                        continue
                # numeric
                if pd.api.types.is_numeric_dtype(ser):
                    nodes.append({"Question": c, "Check_Type": "Range", "Condition": "1-99999", "Source": "Auto"})
                else:
                    nodes.append({"Question": c, "Check_Type": "Missing", "Condition": "", "Source": "Auto"})
            rules_df = pd.DataFrame(nodes)

        if rules_df is None or rules_df.empty:
            st.error("No rules available for validation. Upload rules or generate them first.")
        else:
            st.write("Rules preview (first 200 rows):")
            st.dataframe(rules_df.head(200))

            # Run validation
            run = st.button("Run Validation")
            if run:
                issues = []
                # We'll track respondents exempted from range/missing by skip satisfied cases
                exempted_ids_by_var = {}  # var -> set(ids)  (to allow multiple skip rules)
                # Preprocess rules: convert to unified list where Check_Type may be semicolon-separated
                for _, rule in rules_df.iterrows():
                    q_raw = str(rule.get("Question", "")).strip()
                    check_types_raw = str(rule.get("Check_Type", "")).strip()
                    cond_raw = str(rule.get("Condition", "")).strip()
                    # Normalize: if question seems like a prefix (ends with '_') keep as prefix
                    # We'll expand per check type when needed
                    check_types = [ct.strip() for ct in re.split(r'[;,]', check_types_raw) if ct.strip()]
                    conds = [c.strip() for c in re.split(r'[;,]', cond_raw) if c.strip()]

                    # For each check type apply logic
                    for idx, ct in enumerate(check_types):
                        ct_low = ct.lower()
                        condition = conds[idx] if idx < len(conds) else conds[0] if conds else None

                        # STRAIGHTLINER (allow passing prefix like Q9_ to detect all Q9_* columns)
                        if ct_low == "straightliner":
                            # determine related cols: if q_raw endswith '_' treat as prefix
                            if q_raw.endswith("_"):
                                related = expand_prefix(q_raw, cols)
                            elif "," in q_raw:
                                related = [x.strip() for x in q_raw.split(",") if x.strip() in cols]
                            else:
                                related = [q_raw] if q_raw in cols else expand_prefix(q_raw + "_", cols)
                            if len(related) <= 1:
                                continue
                            same_resp = df[related].nunique(axis=1) == 1
                            offenders = df.loc[same_resp, id_col]
                            for rid in offenders:
                                issues.append({id_col: rid, "Question": ",".join(related), "Check_Type": "Straightliner", "Issue": "Same response across items"})

                        # MULTI-SELECT
                        elif ct_low.startswith("multi"):
                            # q_raw could be prefix 'Q2_' or exact column name; we expect prefix
                            prefix = q_raw if q_raw.endswith("_") else q_raw + "_" if not q_raw.endswith("_") and any(c.startswith(q_raw + "_") for c in cols) else q_raw
                            related = expand_prefix(prefix, cols)
                            if not related:
                                # if direct col provided and there are siblings, expand
                                related = expand_prefix(q_raw + "_", cols)
                            if not related:
                                issues.append({id_col: None, "Question": q_raw, "Check_Type": "Multi-Select", "Issue": "Question not found in dataset"})
                                continue
                            # condition may include tokens like 'only01' and 'atleast1' etc
                            conds_ms = (condition or "").lower().split()
                            # check only 0/1 allowed
                            invalid_mask = pd.Series(False, index=df.index)
                            for col in related:
                                invalid_mask |= ~df[col].isin([0, 1]) & ~df[col].astype(str).str.upper().eq("NA")
                                # Note: 'NA' string considered valid answer and NOT invalid
                            for rid in df.loc[invalid_mask, id_col]:
                                issues.append({id_col: rid, "Question": ",".join(related), "Check_Type": "Multi-Select", "Issue": "Invalid values (not 0/1 or NA)"})
                            # check at least one selected
                            zero_selected = df[related].fillna(0).sum(axis=1) == 0
                            for rid in df.loc[zero_selected, id_col]:
                                issues.append({id_col: rid, "Question": ",".join(related), "Check_Type": "Multi-Select", "Issue": "No options selected"})

                        # SKIP
                        elif ct_low == "skip":
                            # condition expected like: If Q2=1 then Q3 should be blank / should be answered
                            if not condition or "then" not in condition.lower():
                                issues.append({id_col: None, "Question": q_raw, "Check_Type": "Skip", "Issue": f"Invalid skip format ({condition})"})
                                continue
                            if_part, then_part = re.split(r'(?i)then', condition, maxsplit=1)
                            mask = parse_skip_logic_text(if_part, df)
                            # parse then target(s): "Q3 should be blank" or "Q3_1 to Q3_13 should be answered"
                            then_part = then_part.strip()
                            # extract target word(s)
                            # check if 'to' range present
                            targets = []
                            if " to " in then_part.lower() or re.search(r'\d+\s*-\s*\d+', then_part):
                                # try range in then_part
                                targets = expand_range_expr(then_part, cols)
                            else:
                                # first token could be prefix or variable (maybe like Q3_ or Q3)
                                tok = then_part.split()[0].strip()
                                # if token ends with '_' treat prefix
                                if tok.endswith("_"):
                                    targets = expand_prefix(tok, cols)
                                elif tok in cols:
                                    targets = [tok]
                                else:
                                    # try prefix match (Q3 -> Q3_)
                                    candidates = expand_prefix(tok + "_", cols)
                                    targets = candidates if candidates else ([tok] if tok in cols else [])
                            if not targets:
                                issues.append({id_col: None, "Question": q_raw, "Check_Type": "Skip", "Issue": f"Target variable(s) not found in data for then-part '{then_part}'"})
                                continue
                            should_be_blank = "blank" in then_part.lower()
                            for tcol in targets:
                                blank_mask = is_blank_series(df[tcol])
                                # treat literal 'NA' as answered if option set; that is already handled in is_blank_series
                                if should_be_blank:
                                    offenders = df.loc[mask & ~blank_mask, id_col]
                                    for rid in offenders:
                                        issues.append({id_col: rid, "Question": tcol, "Check_Type": "Skip", "Issue": "Answered but should be blank"})
                                    # note: for exemption of range/missing we will mark those rows as not to check further
                                    exempted = df.loc[mask & blank_mask, id_col].tolist()
                                    exempted_ids_by_var.setdefault(tcol, set()).update(set(exempted))
                                else:
                                    # should be answered
                                    offenders = df.loc[mask & blank_mask, id_col]
                                    for rid in offenders:
                                        issues.append({id_col: rid, "Question": tcol, "Check_Type": "Skip", "Issue": "Blank but should be answered"})

                        # RANGE
                        elif ct_low == "range":
                            # condition expected like "1-5" or "1 to 5"
                            try:
                                if condition is None or condition.strip() == "":
                                    raise ValueError("Empty range")
                                rng = condition.replace("to", "-").strip()
                                if "-" not in rng:
                                    raise ValueError("Range must contain '-' or 'to'")
                                minv, maxv = [float(x.strip()) for x in rng.split("-",1)]
                                # related columns: q_raw may be prefix
                                if q_raw.endswith("_"):
                                    related = expand_prefix(q_raw, cols)
                                elif q_raw in cols:
                                    related = [q_raw]
                                else:
                                    # try prefix
                                    related = expand_prefix(q_raw + "_", cols)
                                if not related:
                                    issues.append({id_col: None, "Question": q_raw, "Check_Type": "Range", "Issue": "Question not found in dataset"})
                                    continue
                                for col in related:
                                    # create effective rows to check: exclude exempted ids for this col
                                    rows_mask = pd.Series(True, index=df.index)
                                    if col in exempted_ids_by_var:
                                        rows_mask &= ~df[id_col].isin(exempted_ids_by_var[col])
                                    # numeric compare: coerce to numeric
                                    col_vals = pd.to_numeric(df[col], errors="coerce")
                                    out_mask = ~col_vals.between(minv, maxv) & rows_mask
                                    # treat blanks as not within range? If blank -> it's missing, range shouldn't flag if blank (missing flagged separately)
                                    out_mask &= ~is_blank_series(df[col])
                                    for rid in df.loc[out_mask, id_col]:
                                        issues.append({id_col: rid, "Question": col, "Check_Type": "Range", "Issue": f"Value out of range ({minv}-{maxv})"})
                            except Exception as e:
                                issues.append({id_col: None, "Question": q_raw, "Check_Type": "Range", "Issue": f"Invalid range condition ({condition})"})

                        # MISSING
                        elif ct_low == "missing":
                            # q_raw may be prefix
                            if q_raw.endswith("_"):
                                related = expand_prefix(q_raw, cols)
                            elif q_raw in cols:
                                related = [q_raw]
                            else:
                                related = expand_prefix(q_raw + "_", cols)
                            if not related:
                                issues.append({id_col: None, "Question": q_raw, "Check_Type": "Missing", "Issue": "Question not found in dataset"})
                                continue
                            for col in related:
                                blank_mask = is_blank_series(df[col])
                                # exclude exempted ids if any
                                rows_mask = pd.Series(True, index=df.index)
                                if col in exempted_ids_by_var:
                                    rows_mask &= ~df[id_col].isin(exempted_ids_by_var[col])
                                for rid in df.loc[rows_mask & blank_mask, id_col]:
                                    issues.append({id_col: rid, "Question": col, "Check_Type": "Missing", "Issue": "Value is missing"})

                        # OPENEND_JUNK
                        elif ct_low in ("openend_junk", "openend", "openendjunk"):
                            if q_raw in cols:
                                col = q_raw
                                # treat NA literal as answered (not blank). is_blank_series handled above.
                                # flagged if len < 3 and not blank
                                mask_junk = (~is_blank_series(df[col])) & (df[col].astype(str).str.len() < 3)
                                for rid in df.loc[mask_junk, id_col]:
                                    issues.append({id_col: rid, "Question": col, "Check_Type": "OpenEnd_Junk", "Issue": "Open-end looks like junk/low-effort"})

                        # DUPLICATE
                        elif ct_low == "duplicate":
                            if q_raw in cols:
                                dupes = df[df.duplicated(subset=[q_raw], keep=False)][id_col]
                                for rid in dupes:
                                    issues.append({id_col: rid, "Question": q_raw, "Check_Type": "Duplicate", "Issue": "Duplicate value found"})
                        else:
                            # unrecognized check type — note for review
                            issues.append({id_col: None, "Question": q_raw, "Check_Type": ct, "Issue": "Unknown/unsupported check type (review)"})

                # build report DataFrame
                if not issues:
                    st.success("No issues found!")
                report_df = pd.DataFrame(issues)
                # normalize column name for Respondent
                if id_col not in report_df.columns and not report_df.empty:
                    # find first column name that matches candidate set in report
                    for c in report_df.columns:
                        if c in df.columns:
                            break
                st.dataframe(report_df)
                # download
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as writer:
                    report_df.to_excel(writer, index=False, sheet_name="Validation Report")
                    # also attach original rules for trace
                    try:
                        rules_df.to_excel(writer, index=False, sheet_name="Rules")
                    except Exception:
                        pass
                st.download_button("Download validation_report.xlsx", data=out.getvalue(), file_name="validation_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
