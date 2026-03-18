#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict

def human_bytes(n: int) -> str:
    if n is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024.0

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

class Node:
    """
    We maintain:
      - self_* : metrics only for this exact module name (exclusive sums)
      - subtree_* : metrics for this node + all descendants (exclusive sums)  <-- no double count
    """
    __slots__ = (
        "name", "full", "children",
        "self_count",
        "self_gpu_excl_sum", "self_gpu_incl_sum",
        "self_cpu_excl_sum", "self_cpu_incl_sum",
        "self_in_max", "self_out_max",
        "types",
        "subtree_count",
        "subtree_gpu_excl_sum",
        "subtree_cpu_excl_sum",
        "subtree_in_max", "subtree_out_max",
    )

    def __init__(self, name, full):
        self.name = name
        self.full = full
        self.children = {}

        self.self_count = 0
        self.self_gpu_excl_sum = 0.0
        self.self_gpu_incl_sum = 0.0
        self.self_cpu_excl_sum = 0.0
        self.self_cpu_incl_sum = 0.0
        self.self_in_max = 0
        self.self_out_max = 0
        self.types = defaultdict(int)

        self.subtree_count = 0
        self.subtree_gpu_excl_sum = 0.0
        self.subtree_cpu_excl_sum = 0.0
        self.subtree_in_max = 0
        self.subtree_out_max = 0

    def add_self_record(self, r, use_gpu=True):
        """
        Add only this module's record into self_* metrics.
        For subtree totals, we will post-aggregate via DFS.
        """
        # prefer exclusive fields if present; fall back to gpu_ms/cpu_ms (old format)
        g_excl = safe_float(r.get("gpu_ms_exclusive"))
        g_incl = safe_float(r.get("gpu_ms_inclusive"))
        c_excl = safe_float(r.get("cpu_ms_exclusive"))
        c_incl = safe_float(r.get("cpu_ms_inclusive"))

        # fallback to old keys if needed
        if g_excl is None:
            g_excl = safe_float(r.get("gpu_ms"))
        if g_incl is None:
            g_incl = safe_float(r.get("gpu_ms"))
        if c_excl is None:
            c_excl = safe_float(r.get("cpu_ms"))
        if c_incl is None:
            c_incl = safe_float(r.get("cpu_ms"))

        if use_gpu and g_excl is None:
            return  # skip non-cuda rows

        self.self_count += 1
        if g_excl is not None:
            self.self_gpu_excl_sum += g_excl
        if g_incl is not None:
            self.self_gpu_incl_sum += g_incl
        if c_excl is not None:
            self.self_cpu_excl_sum += c_excl
        if c_incl is not None:
            self.self_cpu_incl_sum += c_incl

        self.self_in_max = max(self.self_in_max, int(r.get("in_bytes", 0) or 0))
        self.self_out_max = max(self.self_out_max, int(r.get("out_bytes", 0) or 0))

        t = r.get("type")
        if t:
            self.types[t] += 1


def insert_path(root: Node, module: str) -> Node:
    parts = module.split(".")
    cur = root
    full = []
    for p in parts:
        full.append(p)
        if p not in cur.children:
            cur.children[p] = Node(name=p, full=".".join(full))
        cur = cur.children[p]
    return cur

def build_tree(records):
    root = Node(name="(root)", full="(root)")

    # 1) add self records to leaf nodes (exact module)
    for r in records:
        m = r.get("module")
        if not m:
            continue
        node = insert_path(root, m)
        node.add_self_record(r, use_gpu=True)

    # 2) post-order aggregate subtree totals from children's subtree + self
    def dfs(node: Node):
        node.subtree_count = node.self_count
        node.subtree_gpu_excl_sum = node.self_gpu_excl_sum
        node.subtree_cpu_excl_sum = node.self_cpu_excl_sum
        node.subtree_in_max = node.self_in_max
        node.subtree_out_max = node.self_out_max

        for ch in node.children.values():
            dfs(ch)
            node.subtree_count += ch.subtree_count
            node.subtree_gpu_excl_sum += ch.subtree_gpu_excl_sum
            node.subtree_cpu_excl_sum += ch.subtree_cpu_excl_sum
            node.subtree_in_max = max(node.subtree_in_max, ch.subtree_in_max)
            node.subtree_out_max = max(node.subtree_out_max, ch.subtree_out_max)

    dfs(root)
    return root

def to_html(root: Node, min_share_pct: float, max_depth: int):
    total_excl_gpu = root.subtree_gpu_excl_sum if root else 0.0

    def most_common_type(n: Node):
        if not n.types:
            return "—"
        return max(n.types.items(), key=lambda kv: kv[1])[0]

    def share(n: Node):
        if total_excl_gpu <= 0:
            return 0.0
        return (n.subtree_gpu_excl_sum / total_excl_gpu) * 100.0

    def mean_excl(n: Node):
        if n.self_count <= 0:
            return 0.0
        return n.self_gpu_excl_sum / n.self_count if n.self_gpu_excl_sum else 0.0

    def should_show(n: Node, depth: int):
        if depth == 0:
            return True
        if max_depth is not None and depth > max_depth:
            return False
        # keep top 2 levels always; else apply share threshold
        return depth <= 2 or share(n) >= min_share_pct

    def render(n: Node, depth: int):
        if not should_show(n, depth):
            return ""

        has_children = len(n.children) > 0
        children_sorted = sorted(n.children.values(), key=lambda x: x.subtree_gpu_excl_sum, reverse=True)

        row = f"""
        <div class="row" data-full="{n.full}">
          <span class="caret {'caret-down' if depth==0 else ''}">{'' if has_children else '•'}</span>
          <span class="name">{n.name}</span>
          <span class="meta">
            <span class="pill">{most_common_type(n)}</span>

            <span class="kv">subtree_excl_gpu <b>{n.subtree_gpu_excl_sum:.2f} ms</b></span>
            <span class="kv">share <b>{share(n):.2f}%</b></span>

            <span class="kv">self_excl_gpu <b>{n.self_gpu_excl_sum:.2f} ms</b></span>
            <span class="kv">self_incl_gpu <b>{n.self_gpu_incl_sum:.2f} ms</b></span>

            <span class="kv">calls <b>{n.self_count}</b></span>
            <span class="kv">mean_self_excl <b>{mean_excl(n):.4f} ms</b></span>

            <span class="kv">in_max <b>{human_bytes(n.subtree_in_max)}</b></span>
            <span class="kv">out_max <b>{human_bytes(n.subtree_out_max)}</b></span>
          </span>
        </div>
        """

        if not has_children:
            return f"<li>{row}</li>"

        kids_html = "".join(render(ch, depth+1) for ch in children_sorted)
        ul_class = "nested active" if depth == 0 else "nested"
        return f"<li>{row}<ul class='{ul_class}'>{kids_html}</ul></li>"

    body = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Alpamayo Layer Tree (exclusive)</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 16px; }}
  .topbar {{ display:flex; gap:12px; align-items:center; flex-wrap: wrap; }}
  input {{ padding:8px 10px; width: 520px; max-width: 92vw; border:1px solid #ccc; border-radius: 8px; }}
  .btn {{ padding:8px 10px; border:1px solid #ccc; border-radius: 8px; background:#fafafa; cursor:pointer; }}
  .hint {{ color:#555; font-size: 13px; }}
  ul, #tree {{ list-style-type: none; padding-left: 18px; }}
  .row {{ display:flex; align-items:center; gap:10px; padding: 4px 6px; border-radius: 8px; }}
  .row:hover {{ background: #f3f5f7; }}
  .name {{ font-weight: 600; min-width: 220px; }}
  .meta {{ display:flex; flex-wrap: wrap; gap:10px; font-size: 12.5px; color: #333; }}
  .kv {{ color:#333; }}
  .pill {{ display:inline-block; padding:2px 8px; border:1px solid #ddd; border-radius: 999px; background:#fff; font-size: 12px; }}
  .caret {{ cursor: pointer; user-select:none; width:14px; display:inline-block; text-align:center; }}
  .caret::before {{ content: "\\25B6"; color: #666; display: inline-block; }}
  .caret-down::before {{ transform: rotate(90deg); }}
  .nested {{ display: none; margin-left: 10px; }}
  .active {{ display: block; }}
  .muted {{ color:#666; font-size: 13px; }}
  .warn {{ color:#8a3b12; }}
</style>
</head>
<body>
  <h2>Alpamayo Layer Tree (exclusive, no-double-count subtree totals)</h2>

  <div class="topbar">
    <input id="q" placeholder="Filter by module path substring, e.g. 'visual.blocks.0' or 'language_model' ..."/>
    <button class="btn" onclick="expandAll()">Expand all</button>
    <button class="btn" onclick="collapseAll()">Collapse all</button>
    <button class="btn" onclick="clearFilter()">Clear</button>
  </div>

  <div class="muted">
    Root subtree_excl_gpu total (sum of exclusive across all nodes): <b>{total_excl_gpu:.2f} ms</b>
    <span class="warn">(This is much closer to end-to-end than inclusive sums, but can still differ if work happens outside modules you recorded.)</span>
  </div>

  <div class="hint">Click the triangle to expand/collapse. “subtree_excl_gpu” is your no-double-count total for that subtree.</div>

  <ul id="tree">
    {render(root, 0)}
  </ul>

<script>
function setNested(li, on) {{
  const nested = li.querySelector(":scope > ul.nested");
  const caret = li.querySelector(":scope > .row > .caret");
  if (!nested || !caret) return;
  if (on) {{
    nested.classList.add("active");
    caret.classList.add("caret-down");
  }} else {{
    nested.classList.remove("active");
    caret.classList.remove("caret-down");
  }}
}}

document.addEventListener("click", function(e) {{
  if (e.target.classList.contains("caret")) {{
    const li = e.target.closest("li");
    const nested = li.querySelector(":scope > ul.nested");
    if (!nested) return;
    nested.classList.toggle("active");
    e.target.classList.toggle("caret-down");
  }}
}});

function expandAll() {{
  document.querySelectorAll("#tree li").forEach(li => setNested(li, true));
}}
function collapseAll() {{
  document.querySelectorAll("#tree li").forEach(li => setNested(li, false));
  const rootLi = document.querySelector("#tree > li");
  if (rootLi) setNested(rootLi, true);
}}
function clearFilter() {{
  document.getElementById("q").value = "";
  applyFilter("");
}}
function applyFilter(q) {{
  q = q.trim().toLowerCase();
  const lis = document.querySelectorAll("#tree li");
  lis.forEach(li => li.style.display = "");
  if (!q) return;

  // Hide rows that don't match. Also keep ancestors of matches visible.
  lis.forEach(li => {{
    const row = li.querySelector(":scope > .row");
    if (!row) return;
    const full = (row.getAttribute("data-full") || "").toLowerCase();
    if (full.includes(q)) {{
      // keep this and ancestors
      let cur = li;
      while (cur) {{
        cur.style.display = "";
        const parentLi = cur.parentElement ? cur.parentElement.closest("li") : null;
        cur = parentLi;
      }}
    }} else {{
      li.style.display = "none";
    }}
  }});
}}
document.getElementById("q").addEventListener("input", (e) => applyFilter(e.target.value));
</script>
</body>
</html>
"""
    return body

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_html", default="layer_tree_exclusive.html")
    ap.add_argument("--clip_id", default=None)
    ap.add_argument("--min_share_pct", type=float, default=0.05)
    ap.add_argument("--max_depth", type=int, default=50)
    args = ap.parse_args()

    records = []
    with open(args.in_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if args.clip_id and r.get("clip_id") != args.clip_id:
                continue
            # keep only rows with some GPU info (exclusive or fallback)
            if r.get("gpu_ms_exclusive") is None and r.get("gpu_ms") is None:
                continue
            records.append(r)

    root = build_tree(records)
    html = to_html(root, min_share_pct=args.min_share_pct, max_depth=args.max_depth)

    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Read records: {len(records)}")
    print(f"Wrote HTML: {args.out_html}")
    print("Open it with your browser as a local file.")

if __name__ == "__main__":
    main()
