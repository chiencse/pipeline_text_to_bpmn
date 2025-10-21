# tools/run_with_pretty_log.py
import json, time
from datetime import datetime
from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.json import JSON
from rich.text import Text
from rich.live import Live
from rich import box

console = Console()
node_start_ts = {}
node_durations = defaultdict(float)

def fmt_ts(ts=None):
    dt = datetime.fromtimestamp(ts or time.time())
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def color_for_node(node: str) -> str:
    # tuỳ biến theo nhóm node của bạn
    if "preprocess" in node: return "cyan"
    if "ner" in node: return "magenta"
    if "retrieve" in node: return "yellow"
    if "bpmn" in node: return "green"
    if "validate" in node: return "red"
    if "render" in node: return "blue"
    return "white"

def pretty_value(value):
    # dùng rich.JSON cho dict/list, còn lại hiển thị as-is
    if isinstance(value, (dict, list)):
        try:
            return JSON.from_data(value, indent=2)
        except Exception:
            return Text(json.dumps(value, indent=2, ensure_ascii=False), style="white")
    return Text(str(value), style="white")

def print_event(node_name: str, node_data: dict):
    start = node_start_ts.get(node_name, time.time())
    end = time.time()
    elapsed = end - start
    node_durations[node_name] += elapsed  # tích luỹ nếu node lặp

    header = Text.assemble(
        ("🧩 NODE ", "bold white"),
        (node_name, f"bold {color_for_node(node_name)}"),
        ("  "), ("•", "dim"), ("  "),
        ("started ", "dim"), (fmt_ts(start), "italic dim"),
        ("  "), ("•", "dim"), ("  "),
        ("finished ", "dim"), (fmt_ts(end), "italic dim"),
        ("  "), ("•", "dim"), ("  "),
        ("Δ ", "bold"), (f"{elapsed*1000:.1f} ms", "bold")
    )

    # build table of keys
    tbl = Table.grid(expand=True)
    tbl.add_column(justify="left", ratio=1)
    for key, val in (node_data.items() if isinstance(node_data, dict) else [("value", node_data)]):
        tbl.add_row(Text(f"🔸 {key}", style="bold"))
        tbl.add_row(pretty_value(val))
        tbl.add_row(Text(""))

    console.print(Panel(tbl, title=header, border_style=color_for_node(node_name), box=box.ROUNDED))

def print_summary():
    if not node_durations:
        return
    t = Table(title="Run Summary", box=box.SIMPLE_HEAVY, show_edge=True, expand=True)
    t.add_column("Node", style="bold")
    t.add_column("Total Time (ms)", justify="right")
    t.add_column("Color", justify="center")
    total = 0.0
    for node, sec in node_durations.items():
        total += sec
        t.add_row(node, f"{sec*1000:.1f}", f"[{color_for_node(node)}]■[/]")
    t.add_row("—", "—", "—")
    t.add_row("[bold]TOTAL[/]", f"[bold]{total*1000:.1f}[/]", "")
    console.print(t)


