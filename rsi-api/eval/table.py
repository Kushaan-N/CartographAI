"""
Results table formatter.

Takes eval results dict and formats as:
1. Pretty printed terminal table
2. Markdown table string (for slides)
3. JSON file (for programmatic access)
"""
import json
import os

_LEVEL_ENDPOINTS = {1: 3, 2: 6, 3: 10, 4: 15}


def format_terminal_table(results: dict) -> str:
    """
    Format results dict as terminal table string.

    results format:
    {
      "episodes_trained": int,
      "curriculum_level_reached": int,
      "levels": {
        "1": {
          "endpoints": int,
          "base": {"mean_coverage": float, "success_rate": float, "mean_steps": float, "n_episodes": int},
          "trained": {"mean_coverage": float, "success_rate": float, "mean_steps": float, "n_episodes": int},
          "improvement": float
        },
        ...
      }
    }
    """
    episodes = results.get("episodes_trained", 0)
    level_reached = results.get("curriculum_level_reached", 1)
    levels = results.get("levels", {})

    lines = [
        "=== RSI-API EVALUATION RESULTS ===",
        f"Episodes trained: {episodes:,} | Curriculum level reached: {level_reached}",
        "",
        f"{'':20}{'BASE MODEL':<18}{'TRAINED MODEL'}",
        f"{'':20}{'Coverage':<10}{'Success':<9}{'Coverage':<10}{'Success':<9}{'Steps':<7}Improvement",
    ]

    improvements = []
    for lvl_str in sorted(levels.keys(), key=int):
        lvl = int(lvl_str)
        d = levels[lvl_str]
        n = d["base"]["n_episodes"]
        ep_count = d.get("endpoints") or _LEVEL_ENDPOINTS.get(lvl, "?")

        base_cov = d["base"]["mean_coverage"] * 100
        base_suc = int(round(d["base"]["success_rate"] * n))
        tr_cov = d["trained"]["mean_coverage"] * 100
        tr_suc = int(round(d["trained"]["success_rate"] * n))
        tr_steps = d["trained"]["mean_steps"]
        imp = d.get("improvement", (tr_cov - base_cov) / 100) * 100
        improvements.append(imp)

        label = f"Level {lvl} ({ep_count} ep)"
        suc_base = f"{base_suc}/{n}"
        suc_tr = f"{tr_suc}/{n}"
        lines.append(
            f"{label:<20}{base_cov:>5.1f}%   {suc_base:<8}{tr_cov:>5.1f}%   {suc_tr:<8}{tr_steps:>5.1f}   +{imp:.1f}%"
        )

    mean_imp = sum(improvements) / len(improvements) if improvements else 0.0
    lines += [
        "",
        f"Mean improvement: +{mean_imp:.1f}% coverage over base model",
        "=====================================",
    ]
    return "\n".join(lines)


def format_markdown_table(results: dict) -> str:
    """
    Format results as markdown table for slides/README.

    | Level | Base Coverage | Trained Coverage | Improvement | Steps to Map |
    |-------|--------------|-----------------|-------------|--------------|
    | 1 (3 endpoints) | 31.2% | 94.1% | +62.9% | 8.3 |
    ...
    """
    levels = results.get("levels", {})
    lines = [
        "| Level | Base Coverage | Trained Coverage | Improvement | Steps to Map |",
        "|-------|--------------|-----------------|-------------|--------------|",
    ]
    for lvl_str in sorted(levels.keys(), key=int):
        lvl = int(lvl_str)
        d = levels[lvl_str]
        ep_count = d.get("endpoints") or _LEVEL_ENDPOINTS.get(lvl, "?")
        base_cov = d["base"]["mean_coverage"] * 100
        tr_cov = d["trained"]["mean_coverage"] * 100
        imp = d.get("improvement", (tr_cov - base_cov) / 100) * 100
        tr_steps = d["trained"]["mean_steps"]
        lines.append(
            f"| {lvl} ({ep_count} endpoints) | {base_cov:.1f}% | {tr_cov:.1f}% | +{imp:.1f}% | {tr_steps:.1f} |"
        )
    return "\n".join(lines)


def save_results(results: dict, path: str = "eval/results.json"):
    """Save results dict to JSON file. Create parent dirs if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(path: str = "eval/results.json") -> dict:
    """Load results dict from JSON file."""
    with open(path) as f:
        return json.load(f)
