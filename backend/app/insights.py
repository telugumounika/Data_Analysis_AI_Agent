"""Stage 10: insights that are computed first, written up second.

The anti-hallucination rule: Python calculates every fact. The LLM (if used)
may only reword the list of facts - it is never asked to 'find' insights itself.
"""
import pandas as pd

from .config import USE_LLM
from .llm_client import LLMError, chat
from .tools import categorical_columns, date_columns, numeric_columns


def compute_facts(df: pd.DataFrame) -> list[str]:
    facts = [f"The dataset has {len(df):,} rows and {len(df.columns)} columns."]

    missing = int(df.isna().sum().sum())
    if missing:
        pct = missing / (df.shape[0] * df.shape[1]) * 100
        worst = df.isna().sum().idxmax()
        facts.append(f"{pct:.1f}% of all cells are missing; '{worst}' has the most missing values.")
    else:
        facts.append("There are no missing values.")

    dups = int(df.duplicated().sum())
    if dups:
        facts.append(f"{dups} duplicate rows were found.")

    nums = numeric_columns(df)
    # pick the main metric: first numeric column with a business-like name
    main = next((c for w in ["sales", "revenue", "amount", "profit"] for c in nums if w in c.lower()),
                nums[0] if nums else None)
    if main:
        facts.append(f"Total {main} is {df[main].sum():,.2f} (average {df[main].mean():,.2f} per row).")
        for cat in categorical_columns(df):
            if 2 <= df[cat].nunique() <= 30:
                g = df.groupby(cat)[main].sum().sort_values(ascending=False)
                share = g.iloc[0] / g.sum() * 100 if g.sum() else 0
                facts.append(f"'{g.index[0]}' has the highest {main} by {cat} "
                             f"({g.iloc[0]:,.2f}, {share:.1f}% of the total); "
                             f"'{g.index[-1]}' has the lowest.")
        for d in date_columns(df)[:1]:
            m = df.groupby(df[d].dt.to_period("M"))[main].sum()
            if len(m) >= 2:
                facts.append(f"Monthly {main} peaked in {m.idxmax()} and was lowest in {m.idxmin()}.")

    outlier_cols = []
    for c in nums:
        s = df[c].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        if n:
            outlier_cols.append((c, n))
    if outlier_cols:
        c, n = max(outlier_cols, key=lambda t: t[1])
        facts.append(f"Potential outliers exist; '{c}' has the most ({n} values).")

    if len(nums) >= 2:
        corr = df[nums].corr().abs()
        pairs = corr.where(~pd.DataFrame(
            [[i >= j for j in range(len(nums))] for i in range(len(nums))],
            index=corr.index, columns=corr.columns)).stack()
        if not pairs.empty and pairs.max() > 0.5:
            a, b = pairs.idxmax()
            facts.append(f"'{a}' and '{b}' are strongly correlated ({df[a].corr(df[b]):.2f}).")
    return facts


def generate_insights(df: pd.DataFrame) -> dict:
    facts = compute_facts(df)
    text = "Key insights:\n" + "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
    mode = "rules"
    if USE_LLM:
        try:
            msg = chat([
                {"role": "system", "content":
                    "You are a data analyst. Rewrite the verified facts below as 4-7 short, clear "
                    "bullet-point insights. Use ONLY these facts and numbers. Do not add anything new."},
                {"role": "user", "content": "\n".join(f"- {f}" for f in facts)},
            ])
            if msg.get("content"):
                text, mode = "Key insights:\n" + msg["content"].strip(), "llm"
        except LLMError:
            pass
    return {"insights": text, "facts": facts, "mode": mode}
