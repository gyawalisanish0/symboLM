import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

def render_diagram(output_path):
    # Set high DPI for crisp presentation
    dpi = 200
    fig_width = 16
    fig_height = 9
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor='#0b0f19')
    ax = fig.add_axes([0, 0, 1, 1], facecolor='#0b0f19')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Color Palette (Dark Theme / GitHub & Terminal Inspired)
    bg_card = '#161b26'
    border_card = '#30363d'
    cyan = '#58a6ff'
    emerald = '#3fb950'
    amber = '#d29922'
    purple = '#bc8cff'
    text_white = '#f0f6fc'
    text_muted = '#8b949e'
    text_dim = '#6e7681'
    highlight_red = '#f85149'

    # 1. HEADER
    ax.text(0.8, 8.4, "SymboLM: 2nd-Gen Test-Time Compute Architecture for Edge LLMs", 
            fontsize=20, weight='bold', color=text_white, fontfamily='sans-serif')
    ax.text(0.8, 8.05, "Fixing 1st-Gen Reasoning Bottlenecks: Overthinking, Unstructured Monologue & Edge Budget-Choking", 
            fontsize=11, color=cyan, fontfamily='sans-serif')
    ax.text(0.8, 7.75, "Architect: Sanish Gyawali (Rupandehi, Nepal)  |  Base: DeepSeek-R1-Distill-1.5B  |  Open Weights (Apache 2.0)", 
            fontsize=9.5, color=text_muted, fontfamily='monospace')

    # Status Pill (Top Right)
    pill = patches.FancyBboxPatch((12.0, 7.8), 3.2, 0.7, boxstyle="round,pad=0.1,rounding_size=0.15",
                                 facecolor='#1f2430', edgecolor='#388bfd', linewidth=1.2)
    ax.add_patch(pill)
    ax.text(13.6, 8.2, "TEST-TIME COMPUTE AUDIT", fontsize=8, weight='bold', color=cyan, 
            ha='center', va='center', fontfamily='monospace')
    ax.text(13.6, 7.95, "NVIDIA Tesla T4 (NVML 100ms Polled)", fontsize=7.5, color=text_white, 
            ha='center', va='center', fontfamily='sans-serif')

    # 2. THREE ARCHITECTURAL MODULES (Columns)
    col_y = 3.65
    col_h = 3.8
    col_w = 4.4

    # --- COLUMN 1: SYSTEM 1 REFLEX ---
    c1_x = 0.8
    p1 = patches.FancyBboxPatch((c1_x, col_y), col_w, col_h, boxstyle="round,pad=0.1,rounding_size=0.2",
                                facecolor=bg_card, edgecolor='#238636', linewidth=1.5)
    ax.add_patch(p1)
    
    # Header 1
    ax.text(c1_x + 0.3, col_y + col_h - 0.4, "1. ADAPTIVE TEST-TIME COMPUTE", fontsize=11, weight='bold', color=emerald, fontfamily='sans-serif')
    ax.text(c1_x + 0.3, col_y + col_h - 0.7, "Subcortical Reflex (<reg:bypass>)", fontsize=8.5, color=text_muted, fontfamily='monospace')
    
    # Concept box
    desc1 = (
        "• Problem: Uniform compute allocation. 1st-gen\n"
        "  reasoning models (o1, R1) spend thousands of\n"
        "  thinking tokens on trivial/atomic queries.\n\n"
        "• Solution: Token-level dynamic difficulty gating.\n"
        "  Bypasses test-time compute loops completely\n"
        "  for atomic factual retrieval.\n\n"
        "• Trace: Q: Atomic number of Carbon?\n"
        "  <reg:bypass> 6 </reg:bypass>"
    )
    ax.text(c1_x + 0.3, col_y + col_h - 1.0, desc1, fontsize=8.5, color=text_white, va='top', fontfamily='sans-serif', linespacing=1.2)

    # Metric Box 1
    m1 = patches.FancyBboxPatch((c1_x + 0.25, col_y + 0.25), col_w - 0.5, 0.95, boxstyle="round,pad=0.08,rounding_size=0.1",
                               facecolor='#102318', edgecolor=emerald, linewidth=1)
    ax.add_patch(m1)
    ax.text(c1_x + 0.4, col_y + 0.9, "80× Dynamic Energy Reduction", fontsize=9.5, weight='bold', color=emerald)
    ax.text(c1_x + 0.4, col_y + 0.65, "3 tokens  |  25.8 Joules / solve", fontsize=9, color=text_white, fontfamily='monospace')
    ax.text(c1_x + 0.4, col_y + 0.42, "vs. 1st-Gen Monologue: 167 tok | 2,052 Joules", fontsize=7.5, color=text_muted)

    # --- COLUMN 2: SYSTEM 2 WORKING MEMORY ---
    c2_x = 5.8
    p2 = patches.FancyBboxPatch((c2_x, col_y), col_w, col_h, boxstyle="round,pad=0.1,rounding_size=0.2",
                                facecolor=bg_card, edgecolor='#8957e5', linewidth=1.5)
    ax.add_patch(p2)
    
    # Header 2
    ax.text(c2_x + 0.3, col_y + col_h - 0.4, "2. STRUCTURED STATE MACHINE", fontsize=11, weight='bold', color=purple, fontfamily='sans-serif')
    ax.text(c2_x + 0.3, col_y + col_h - 0.7, "Cognitive Registers (<reg:plan>, <reg:deduce>)", fontsize=8.5, color=text_muted, fontfamily='monospace')

    desc2 = (
        "• Problem: The \"Overthinking Trap\" (ArXiv 2024-25).\n"
        "  Unstructured natural language monologue drifts\n"
        "  into second-guessing loops and token bloat.\n\n"
        "• Solution: Bounded working memory registers\n"
        "  with counterfactual branching and pruning:\n"
        "  hyp [...] -> verify(...) -> [ ✗ | ✓ ].\n\n"
        "• Enforces strict formal state invariants."
    )
    ax.text(c2_x + 0.3, col_y + col_h - 1.0, desc2, fontsize=8.5, color=text_white, va='top', fontfamily='sans-serif', linespacing=1.2)

    # Metric Box 2
    m2 = patches.FancyBboxPatch((c2_x + 0.25, col_y + 0.25), col_w - 0.5, 0.95, boxstyle="round,pad=0.08,rounding_size=0.1",
                               facecolor='#1e1533', edgecolor=purple, linewidth=1)
    ax.add_patch(m2)
    ax.text(c2_x + 0.4, col_y + 0.9, "38.3% Lower Energy vs Monologue", fontsize=9.5, weight='bold', color=purple)
    ax.text(c2_x + 0.4, col_y + 0.65, "125.0 tokens  |  1,265.7 Joules / solve", fontsize=9, color=text_white, fontfamily='monospace')
    ax.text(c2_x + 0.4, col_y + 0.42, "Prunes wandering counterfactual paths", fontsize=7.5, color=text_muted)

    # --- COLUMN 3: TINY TERMINAL ---
    c3_x = 10.8
    p3 = patches.FancyBboxPatch((c3_x, col_y), col_w, col_h, boxstyle="round,pad=0.1,rounding_size=0.2",
                                facecolor=bg_card, edgecolor='#1f6feb', linewidth=1.5)
    ax.add_patch(p3)

    # Header 3
    ax.text(c3_x + 0.3, col_y + col_h - 0.4, "3. DETERMINISTIC REPL OFFLOAD", fontsize=11, weight='bold', color=cyan, fontfamily='sans-serif')
    ax.text(c3_x + 0.3, col_y + col_h - 0.7, "Neural-Symbolic Sandbox (<reg:term>)", fontsize=8.5, color=text_muted, fontfamily='monospace')

    desc3 = (
        "• Problem: Arithmetic Hallucination Delusion.\n"
        "  Transformer weights make terrible ALUs;\n"
        "  simulating division in text compounds errors.\n\n"
        "• Solution: Isolated, sandboxed SymPy/Python\n"
        "  execution (<5ms). Weights guide high-level\n"
        "  intent; silicon handles exact arithmetic.\n\n"
        "• Eliminates token-wasting arithmetic monologue."
    )
    ax.text(c3_x + 0.3, col_y + col_h - 1.0, desc3, fontsize=8.5, color=text_white, va='top', fontfamily='sans-serif', linespacing=1.2)

    # Metric Box 3
    m3 = patches.FancyBboxPatch((c3_x + 0.25, col_y + 0.25), col_w - 0.5, 0.95, boxstyle="round,pad=0.08,rounding_size=0.1",
                               facecolor='#101d36', edgecolor=cyan, linewidth=1)
    ax.add_patch(m3)
    ax.text(c3_x + 0.4, col_y + 0.9, "3.3× Accuracy Boost | 78.4% Energy Cut", fontsize=9.5, weight='bold', color=cyan)
    ax.text(c3_x + 0.4, col_y + 0.65, "23.0% Pass@1  |  444.3 Joules / solve", fontsize=9, color=text_white, fontfamily='monospace')
    ax.text(c3_x + 0.4, col_y + 0.42, "vs. 1st-Gen Monologue: 7.0% Pass@1 | 2,052 J", fontsize=7.5, color=text_muted)

    # 3. BOTTOM PANEL: REAL SILICON PARETO FRONTIER & COMPARISON
    bot_y = 0.5
    bot_h = 2.8
    bot_w = 14.4
    p_bot = patches.FancyBboxPatch((0.8, bot_y), bot_w, bot_h, boxstyle="round,pad=0.1,rounding_size=0.2",
                                   facecolor=bg_card, edgecolor=border_card, linewidth=1)
    ax.add_patch(p_bot)

    # Bottom Header
    ax.text(1.1, bot_y + bot_h - 0.35, "HARDWARE SILICON BENCHMARK (Held-Out Clean 200 GSM8K/Math Evaluation on NVIDIA Tesla T4)", 
            fontsize=11, weight='bold', color=text_white, fontfamily='sans-serif')

    # Sub-table / Metrics representation
    # Row headers
    ax.text(1.1, bot_y + 1.85, "EVALUATION ARM", fontsize=8, weight='bold', color=text_muted, fontfamily='monospace')
    ax.text(4.2, bot_y + 1.85, "STRATEGY", fontsize=8, weight='bold', color=text_muted, fontfamily='monospace')
    ax.text(7.6, bot_y + 1.85, "PASS@1 ACCURACY", fontsize=8, weight='bold', color=text_muted, fontfamily='monospace')
    ax.text(10.2, bot_y + 1.85, "ENERGY / SOLVE", fontsize=8, weight='bold', color=text_muted, fontfamily='monospace')
    ax.text(12.8, bot_y + 1.85, "256-TOKEN BUDGET", fontsize=8, weight='bold', color=text_muted, fontfamily='monospace')

    # Horizontal divider
    ax.plot([1.1, 14.8], [bot_y + 1.72, bot_y + 1.72], color=border_card, linewidth=1)

    rows = [
        ("Arm A: Vanilla CoT", "Unstructured Monologue (<think>)", "7.0%", "2,052.4 Joules", "0.0% (Choked / Truncated)", text_muted),
        ("Arm B: Answer-Only Reflex", "Gated System 1 (<reg:bypass>)", "15.0%", "25.8 Joules (-98.7%)", "15.0% (Fast Complete)", emerald),
        ("Arm C: Full Registers", "Plan + Deduce + Backtrack", "10.0%", "1,265.8 Joules (-38.3%)", "10.0% (Structured)", purple),
        ("Arm E: Tiny Terminal", "CoT + Sandboxed SymPy/Python", "23.0% (3.3× Boost)", "444.3 Joules (-78.4%)", "33.3% (Clean Solve)", cyan),
    ]

    curr_y = bot_y + 1.4
    for arm, strat, acc, energy, budget, col in rows:
        ax.text(1.1, curr_y, arm, fontsize=8.5, weight='bold', color=text_white, fontfamily='sans-serif')
        ax.text(4.2, curr_y, strat, fontsize=8.5, color=text_muted, fontfamily='sans-serif')
        ax.text(7.6, curr_y, acc, fontsize=9, weight='bold', color=col, fontfamily='monospace')
        ax.text(10.2, curr_y, energy, fontsize=9, weight='bold', color=col, fontfamily='monospace')
        ax.text(12.8, curr_y, budget, fontsize=8.5, color=text_white, fontfamily='monospace')
        curr_y -= 0.35

    # Footer note
    ax.text(1.1, bot_y + 0.18, 
            "*Physical measurement: NVML 100ms hardware power polling minus idle GPU baseline (P_active = P_total - P_idle). No regex approximations.",
            fontsize=7.5, color=text_dim, fontstyle='italic', fontfamily='sans-serif')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, facecolor='#0b0f19', edgecolor='none')
    plt.close()
    print(f"Diagram successfully saved to {output_path}")

if __name__ == '__main__':
    out = r"C:\Users\user\.gemini\antigravity\brain\72b5b54b-0070-4669-a126-775ccb9b027d\symbolm_grounded_architecture_diagram.png"
    render_diagram(out)
