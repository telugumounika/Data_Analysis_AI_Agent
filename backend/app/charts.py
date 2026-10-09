"""Stage 9: turn data into a PNG chart (returned as base64 so the frontend can show it)."""
import base64
import io

import matplotlib

matplotlib.use("Agg")  # no GUI window - we only need image files
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

PRIMARY = "#4f46e5"


def make_chart(kind: str, title: str, **d) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        if kind == "bar":
            s = d["series"]
            ax.bar([str(i) for i in s.index], s.values, color=PRIMARY)
            ax.set_xlabel(d.get("xlabel", ""))
            ax.set_ylabel(d.get("ylabel", ""))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        elif kind == "line":
            s = d["series"]
            ax.plot([str(i) for i in s.index], s.values, marker="o", color=PRIMARY)
            ax.set_xlabel(d.get("xlabel", ""))
            ax.set_ylabel(d.get("ylabel", ""))
            step = max(1, len(s) // 12)
            ax.set_xticks(range(0, len(s), step))
            ax.set_xticklabels([str(i) for i in s.index][::step], rotation=45, ha="right")
            ax.grid(alpha=0.3)
        elif kind == "pie":
            s = d["series"].head(8)
            ax.pie(s.values, labels=[str(i) for i in s.index], autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
        elif kind == "hist":
            ax.hist(d["values"], bins=30, color=PRIMARY, edgecolor="white")
            ax.set_xlabel(d.get("xlabel", ""))
            ax.set_ylabel("Frequency")
        elif kind == "scatter":
            ax.scatter(d["x"], d["y"], alpha=0.5, color=PRIMARY, s=14)
            ax.set_xlabel(d.get("xlabel", ""))
            ax.set_ylabel(d.get("ylabel", ""))
        elif kind == "heatmap":
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(d["corr"], annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
        ax.set_title(title)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    finally:
        plt.close(fig)
