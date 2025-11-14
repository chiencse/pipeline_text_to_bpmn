def visualize_state(state, step=None, log_path=None):
    """
    Stub for visualizing state. Extend as needed for graph rendering.
    Currently prints summary to console.
    """
    print(f"[VISUALIZE] Step: {step}, Log: {log_path}")
    print(f"State keys: {list(state.keys())}")
    # Add more visualization logic as needed
# tools/viz_graphs.py
import os
from pathlib import Path

OUT = Path("viz")
OUT.mkdir(exist_ok=True)

# import 2 scenario đã có
from app.scenario_a import build_graph_a
from app.scenario_b import build_graph_b

def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    print(f"✔ Wrote: {path}")

def try_draw(compiled_graph, basename: str):
    """
    Tùy phiên bản LangGraph, compiled_graph có thể có:
      - .get_graph().draw_png(...), .draw_svg(...)
      - .get_graph().to_mermaid() hoặc .to_mermaid()
    Mình thử theo thứ tự an toàn.
    """
    g = None

    # 1) Mermaid (an toàn, thường có)
    mermaid = None
    for attr in ("to_mermaid",):
        if hasattr(compiled_graph, attr):
            mermaid = getattr(compiled_graph, attr)()
            break
    if mermaid is None:
        # thử qua get_graph
        if hasattr(compiled_graph, "get_graph"):
            g = compiled_graph.get_graph()
            if hasattr(g, "to_mermaid"):
                mermaid = g.to_mermaid()
    if mermaid:
        save_text(OUT / f"{basename}.mmd", mermaid)
    else:
        print(f"⚠ Mermaid not available for {basename}")

    # 2) PNG/SVG (cần graphviz)
    try:
        if g is None and hasattr(compiled_graph, "get_graph"):
            g = compiled_graph.get_graph()

        drew = False
        if g is not None and hasattr(g, "draw_png"):
            g.draw_png(str(OUT / f"{basename}.png"))
            print(f"✔ PNG: {OUT / f'{basename}.png'}")
            drew = True
        if g is not None and hasattr(g, "draw_svg"):
            g.draw_svg(str(OUT / f"{basename}.svg"))
            print(f"✔ SVG: {OUT / f'{basename}.svg'}")
            drew = True
        if not drew:
            print(f"ℹ No draw_png/draw_svg methods for {basename} (Graphviz path or binding may be missing)")
    except Exception as e:
        print(f"⚠ Failed to draw PNG/SVG for {basename}: {e}")
def main():
    ga = build_graph_a()  
    gb = build_graph_b()
    from IPython.display import Image, display

    try:
        display(Image(ga.get_graph().draw_mermaid_png()))
        png_bytes = ga.get_graph().draw_mermaid_png()
        with open("app/tools/graph_a.png", "wb") as f:
            f.write(png_bytes)
        print("Saved graph_a.png")
    except Exception:
        pass
    try:
        display(Image(gb.get_graph().draw_mermaid_png()))
        png_bytes = gb.get_graph().draw_mermaid_png()
        with open("app/tools/graph_b.png", "wb") as f:
            f.write(png_bytes)
        print("Saved graph_b.png")
    except Exception as e:
        print(f"Failed to save graph_b.png: {e}")
    # try_draw(ga, "scenario_a")
    # try_draw(gb, "scenario_b")

if __name__ == "__main__":
    main()
