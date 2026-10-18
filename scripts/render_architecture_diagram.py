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
    ax.text(0.8, 8.4, "SymboLM: Nature-Inspired Cognitive Architecture for Edge LLMs", 
            fontsize=20, weight='bold', color=text_white, fontfamily='sans-serif')
    ax.text(0.8, 8.05, "Replacing Brute-Force Token Monologue with Biological Cognitive Modularity & Hardware-Measured Efficiency", 
            fontsize=11, color=cyan, fontfamily='sans-serif')
    ax.text(0.8, 7.75, "Architect: Sanish Gyawali (Rupandehi, Nepal)  |  Base: DeepSeek-R1-Distill-1.5B  |  Open Weights (Apache 2.0)", 
            fontsize=9.5, color=text_muted, fontfamily='monospace')

    # Status Pill (Top Right)
    pill = patches.FancyBboxPatch((12.2, 7.8), 3.0, 0.7, boxstyle="round,pad=0.1,rounding_size=0.15",
                                 facecolor='#1f2430', edgecolor='#388bfd', linewidth=1.2)
    ax.add_patch(pill)
    ax.text(13.7, 8.2, "PHYSICAL SILICON AUDIT", fontsize=8, weight='bold', color=cyan, 
            ha='center', va='center', fontfamily='monospace')
    ax.text(13.7, 7.95, "NVIDIA Tesla T4 (NVML 100ms Polled)", fontsize=7.5, color=text_white, 
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
    ax.text(c1_x + 0.3, col_y + col_h - 0.4, "SYSTEM 1: THE REFLEX", fontsize=12, weight='bold', color=emerald, fontfamily='sans-serif')
    ax.text(c1_x + 0.3, col_y + col_h - 0.7, "Subcortical Automaticity (<reg:bypass>)", fontsize=9, color=text_muted, fontfamily='monospace')
    
    # Concept box
    desc1 = (
        "• Human Analogy: Direct retrieval without inner voice.\n"
        "  Recalling \"Carbon = 6\" doesn't need 200 tokens.\n\n"
        "• Execution: Gated Router bypasses reasoning\n"
        "  chains directly for atomic scientific facts.\n\n"
        "• Trace:\n"
        "  Q: Atomic number of Carbon?\n"
        "  <reg:bypass> 6 </reg:bypass>"
    )
    ax.text(c1_x + 0.3, col_y + col_h - 1.0, desc1, fontsize=8.5, color=text_white, va='top', fontfamily='sans-serif', linespacing=1.2)

    # Metric Box 1
    m1 = patches.FancyBboxPatch((c1_x + 0.25, col_y + 0.25), col_w - 0.5, 0.95, boxstyle="round,pad=0.08,rounding_size=0.1",
                               facecolor='#102318', edgecolor=emerald, linewidth=1)
    ax.add_patch(m1)
    ax.text(c1_x + 0.4, col_y + 0.9, "80× Dynamic Energy Reduction", fontsize=9.5, weight='bold', color=emerald)
    ax.text(c1_x + 0.4, col_y + 0.65, "3 tokens  |  25.8 Joules / solve", fontsize=9, color=text_white, fontfamily='monospace')
    ax.text(c1_x + 0.4, col_y + 0.42, "vs. Vanilla CoT: 167 tokens | 2,052 Joules", fontsize=7.5, color=text_muted)

    # --- COLUMN 2: SYSTEM 2 WORKING MEMORY ---
    c2_x = 5.8
    p2 = patches.FancyBboxPatch((c2_x, col_y), col_w, col_h, boxstyle="round,pad=0.1,rounding_size=0.2",
                                facecolor=bg_card, edgecolor='#8957e5', linewidth=1.5)
    ax.add_patch(p2)
    
    # Header 2
    ax.text(c2_x + 0.3, col_y + col_h - 0.4, "SYSTEM 2: MENTAL SCRATCHPAD", fontsize=12, weight='bold', color=purple, fontfamily='sans-serif')
    ax.text(c2_x + 0.3, col_y + col_h - 0.7, "Prefrontal Working Memory (<reg:plan>, <reg:deduce>)", fontsize=8.5, color=text_muted, fontfamily='monospace')

    desc2 = (
        "• Human Analogy: Controlled multi-step derivation.\n"
        "  Enforces bounded mental scratchpad discipline.\n\n"
        "• Execution: Formal registers structure reasoning;\n"
        "  Counterfactual branches with explicit verification:\n"
        "  hyp [...] -> verify(...) -> [ ✗ | ✓ ].\n\n"
        "• Eliminates runaway token loops & hallucination."
    )
    ax.text(c2_x + 0.3, col_y + col_h - 1.0, desc2, fontsize=8.5, color=text_white, va='top', fontfamily='sans-serif', linespacing=1.2)

    # Metric Box 2
    m2 = patches.FancyBboxPatch((c2_x + 0.25, col_y + 0.25), col_w - 0.5, 0.95, boxstyle="round,pad=0.08,rounding_size=0.1",
                               facecolor='#1e1533', edgecolor=purple, linewidth=1)
    ax.add_patch(m2)
    ax.text(c2_x + 0.4, col_y + 0.9, "38.3% Lower Energy vs CoT", fontsize=9.5, weight='bold', color=purple)
    ax.text(c2_x + 0.4, col_y + 0.65, "125.0 tokens  |  1,265.7 Joules / solve", fontsize=9, color=text_white, fontfamily='monospace')
    ax.text(c2_x + 0.4, col_y + 0.42, "Structured counterfactual derivation", fontsize=7.5, color=text_muted)

    # --- COLUMN 3: TINY TERMINAL ---
    c3_x = 10.8
    p3 = patches.FancyBboxPatch((c3_x, col_y), col_w, col_h, boxstyle="round,pad=0.1,rounding_size=0.2",
                                facecolor=bg_card, edgecolor='#1f6feb', linewidth=1.5)
    ax.add_patch(p3)

    # Header 3
    ax.text(c3_x + 0.3, col_y + col_h - 0.4, "EXTENDED MIND: TINY TERMINAL", fontsize=12, weight='bold', color=cyan, fontfamily='sans-serif')
    ax.text(c3_x + 0.3, col_y + col_h - 0.7, "Offloaded Symbolic Computation (<reg:term>)", fontsize=8.5, color=text_muted, fontfamily='monospace')

    desc3 = (
        "• Human Analogy: Pencil & paper / calculator.\n"
        "  Humans don't calculate 384 * 17 purely in head;\n"
        "  we write it down or punch into a tool.\n\n"
        "• Execution: Deterministic math expressions offloaded\n"
        "  to sandboxed Python/SymPy runtime (<5ms).\n\n"
        "• Weights guide strategy; silicon handles exact math."
    )
    ax.text(c3_x + 0.3, col_y + col_h - 1.0, desc3, fontsize=8.5, color=text_white, va='top', fontfamily='sans-serif', linespacing=1.2)

    # Metric Box 3
    m3 = patches.FancyBboxPatch((c3_x + 0.25, col_y + 0.25), col_w - 0.5, 0.95, boxstyle="round,pad=0.08,rounding_size=0.1",
                               facecolor='#101d36', edgecolor=cyan, linewidth=1)
    ax.add_patch(m3)
    ax.text(c3_x + 0.4, col_y + 0.9, "3.3× Accuracy Boost | 78.4% Energy Cut", fontsize=9.5, weight='bold', color=cyan)
    ax.text(c3_x + 0.4, col_y + 0.65, "23.0% Pass@1  |  444.3 Joules / solve", fontsize=9, color=text_white, fontfamily='monospace')
    ax.text(c3_x + 0.4, col_y + 0.42, "vs. Vanilla CoT: 7.0% Pass@1 | 2,052 Joules", fontsize=7.5, color=text_muted)

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
