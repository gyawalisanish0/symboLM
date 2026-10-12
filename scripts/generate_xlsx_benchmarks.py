import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

def create_workbook():
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # Styles
    navy_fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    bold_font = Font(name="Calibri", size=11, bold=True, color="0F172A")
    regular_font = Font(name="Calibri", size=10, color="1E293B")
    accent_font = Font(name="Calibri", size=11, bold=True, color="1D4ED8")
    zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    highlight_fill = PatternFill(start_color="EFF6FF", end_color="EFF6FF", fill_type="solid")
    success_fill = PatternFill(start_color="F0FDF4", end_color="F0FDF4", fill_type="solid")

    thin_border = Border(
        left=Side(style='thin', color='CBD5E1'),
        right=Side(style='thin', color='CBD5E1'),
        top=Side(style='thin', color='CBD5E1'),
        bottom=Side(style='thin', color='CBD5E1')
    )

    # ---------------- SHEET 1: 200 Cross-Domain Benchmark ----------------
    ws1 = wb.create_sheet(title="200_Cross_Domain_Benchmark")
    ws1.views.sheetView[0].showGridLines = True
    
    ws1["A1"] = "SymboLM Cross-Domain 200-Question Empirical Benchmark (October 12, 2026)"
    ws1["A1"].font = Font(name="Calibri", size=14, bold=True, color="0F172A")
    ws1["A2"] = "Author & Principal Architect: Sanish Gyawali | Operational Exoskeleton: Antigravity (Google DeepMind)"
    ws1["A2"].font = Font(name="Calibri", size=10, italic=True, color="64748B")

    b_headers = ["Domain Category", "Questions (N)", "Accuracy (%)", "Correct", "SymboLM Mean Tok", "Baseline Mean Tok", "Token Compression (%)", "Register Routing Fidelity", "Primary Failure Mode"]
    for col_idx, h in enumerate(b_headers, start=1):
        cell = ws1.cell(row=4, column=col_idx, value=h)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    b_data = [
        ["Strategic Intent & Pragmatics", 25, 1.000, 25, 127.0, 200.0, 0.365, "100% Intent Routing (25/25)", "None (Flawless communicative tone calibration)"],
        ["Scientific & Multi-Hop Factual QA", 25, 0.280, 7, 94.1, 200.0, 0.529, "100% Bypass Routing (25/25)", "Knowledge retrieval boundary in 1.5B weights"],
        ["Pure & Competition Mathematics", 40, 0.175, 7, 128.0, 200.0, 0.360, "75% Plan Routing (30/40)", "128-token Truncation Bottleneck before <ans>"],
        ["Computer Science & Discrete Systems", 30, 0.133, 4, 128.0, 200.0, 0.360, "97% Plan Routing (29/30)", "128-token Truncation Bottleneck before <ans>"],
        ["Chemistry & Quantitative Biology", 25, 0.120, 3, 115.2, 200.0, 0.424, "Mixed Bypass (10) & Plan (11)", "Stoichiometric AST arithmetic precision"],
        ["Probability, Statistics & Finance", 25, 0.080, 2, 127.3, 200.0, 0.364, "92% Plan Routing (23/25)", "Multi-variable combinatorial branching"],
        ["Physics & Engineering", 30, 0.067, 2, 126.3, 200.0, 0.369, "80% Plan Routing (24/30)", "Dimensional analysis truncation at 128 tokens"]
    ]

    for r_idx, row in enumerate(b_data, start=5):
        for c_idx, val in enumerate(row, start=1):
            cell = ws1.cell(row=r_idx, column=c_idx, value=val)
            cell.font = regular_font
            cell.border = thin_border
            if r_idx % 2 == 0:
                cell.fill = zebra_fill
            if c_idx in [2, 4]:
                cell.alignment = Alignment(horizontal="center")
            elif c_idx in [3, 7]:
                cell.number_format = "0.0%"
                cell.alignment = Alignment(horizontal="right")
            elif c_idx in [5, 6]:
                cell.number_format = "0.0"
                cell.alignment = Alignment(horizontal="right")

    # Composite Total Row
    tot_row = len(b_data) + 5
    ws1.cell(row=tot_row, column=1, value="OVERALL COMPOSITE BENCHMARK").font = bold_font
    ws1.cell(row=tot_row, column=2, value="=SUM(B5:B11)").font = bold_font
    ws1.cell(row=tot_row, column=3, value="=D12/B12").font = accent_font
    ws1.cell(row=tot_row, column=3).number_format = "0.0%"
    ws1.cell(row=tot_row, column=4, value="=SUM(D5:D11)").font = bold_font
    ws1.cell(row=tot_row, column=5, value="=AVERAGE(E5:E11)").font = bold_font
    ws1.cell(row=tot_row, column=5).number_format = "0.0"
    ws1.cell(row=tot_row, column=6, value="=AVERAGE(F5:F11)").font = bold_font
    ws1.cell(row=tot_row, column=6).number_format = "0.0"
    ws1.cell(row=tot_row, column=7, value="=1-(E12/F12)").font = accent_font
    ws1.cell(row=tot_row, column=7).number_format = "0.0%"
    ws1.cell(row=tot_row, column=8, value="IER Composite: 20.54").font = bold_font
    ws1.cell(row=tot_row, column=9, value="Resolved by Stage 5 Unbounded Parsimony Gradient").font = bold_font

    for c_i in range(1, 10):
        cell = ws1.cell(row=tot_row, column=c_i)
        cell.fill = highlight_fill
        cell.border = thin_border

    # ---------------- SHEET 2: Stages 1 to 5 Progression ----------------
    ws2 = wb.create_sheet(title="Stages_1_to_5_Progress")
    ws2.views.sheetView[0].showGridLines = True

    ws2["A1"] = "SymboLM Reinforcement Learning & Post-Training Milestones (Stages 1–5)"
    ws2["A1"].font = Font(name="Calibri", size=14, bold=True, color="0F172A")

    st_headers = ["Training Stage", "Primary Strategy", "Compute Cluster", "Rollouts / Steps", "Token Compression", "Key Architectural Finding & Empirical Receipt"]
    for c_idx, h in enumerate(st_headers, start=1):
        cell = ws2.cell(row=3, column=c_idx, value=h)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    st_data = [
        ["Stage 1 (SFT)", "Syntactic AST Seeding", "Dual Tesla T4 (Kaggle)", "1,600 samples / 1h", "Established Prior", "Established non-zero logit prior for formal AST operators (→, ∴, |, let)."],
        ["Stage 2 (GRPO)", "Direct Arithmetic RL", "Dual Tesla T4 (Kaggle)", "16,000 rollouts / 3h 17m", "98.3% Arithmetic", "Compressed 3x+6=21 to 2 tokens; 51.9% overall compression across Modern Math."],
        ["Stage 3 (Concept)", "Olympiad Invariant GRPO", "Dual Tesla T4 (Kaggle)", "16,000 rollouts / 3h 31m", "43.5% Mixed", "Discovered Cold-Start Exploration Barrier; multi-register AST seeding must precede RL."],
        ["Stage 4 (SE-HEA)", "Symbolic Efficiency & Hard EOS", "Single Tesla T4 (Kaggle)", "2,000 rollouts / 2h", "52.9% Factual", "Completely eliminated 220-token repetition loop; cut factual QA to 35 tokens; 1% debris."],
        ["Stage 5 (Parsimony)", "Scratchpad 2.0 Unbounded RL", "Tesla T4 (Kaggle)", "2,400 rollouts / 1h 41m", "Monotonic MDL", "Hyperbolic reward eliminated 128-token Truncation Bottleneck; full multi-depth proofs with zero truncation."]
    ]

    for r_idx, row in enumerate(st_data, start=4):
        for c_idx, val in enumerate(row, start=1):
            cell = ws2.cell(row=r_idx, column=c_idx, value=val)
            cell.font = regular_font
            cell.border = thin_border
            if r_idx % 2 == 1:
                cell.fill = zebra_fill

    # ---------------- SHEET 3: Mathematical Complexity ----------------
    ws3 = wb.create_sheet(title="Mathematical_Complexity_Models")
    ws3.views.sheetView[0].showGridLines = True

    ws3["A1"] = "Mathematical Foundations: Reasoning Density, KV-Cache Scaling & Concurrency"
    ws3["A1"].font = Font(name="Calibri", size=14, bold=True, color="0F172A")

    # Table 3A: KV Cache Scaling
    ws3["A3"] = "1. Key-Value (KV) Cache Complexity & Serving Concurrency (4 GB Allocation)"
    ws3["A3"].font = Font(name="Calibri", size=12, bold=True, color="1E3A8A")

    kv_h = ["Serving Model", "Trace Length (N)", "Layers (L)", "Heads (H)", "Head Dim (D)", "Precision", "Bytes / Token", "KV Cache / Stream (MB)", "Max Streams (4GB Pool)", "Throughput Ratio"]
    for c_i, h in enumerate(kv_h, start=1):
        cell = ws3.cell(row=4, column=c_i, value=h)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    kv_data = [
        ["DeepSeek-1.5B (English CoT Baseline)", 650, 28, 12, 128, "bfloat16", 172032, 106.64, 38, "1.0× (Baseline)"],
        ["DeepSeek-1.5B (SymboLM SRL v1.1)", 60, 28, 12, 128, "bfloat16", 172032, 9.84, 416, "10.9× Concurrency"]
    ]
    for r_i, row in enumerate(kv_data, start=5):
        for c_i, val in enumerate(row, start=1):
            cell = ws3.cell(row=r_i, column=c_i, value=val)
            cell.font = regular_font
            cell.border = thin_border
            if r_i == 6:
                cell.fill = success_fill
                cell.font = bold_font

    # Table 3B: Search Economics
    ws3["A8"] = "2. Test-Time Search Economics: Majority Voting Best-of-8 Analysis"
    ws3["A8"].font = Font(name="Calibri", size=12, bold=True, color="1E3A8A")

    se_h = ["Strategy", "Rollouts (K)", "Tokens / Rollout", "Total Sequence Tokens", "Relative Compute Cost", "Economic Viability"]
    for c_i, h in enumerate(se_h, start=1):
        cell = ws3.cell(row=9, column=c_i, value=h)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    se_data = [
        ["Standard English (Greedy Baseline)", 1, 650, 650, 1.00, "Commercial Baseline"],
        ["Standard English (Best-of-8 Ensemble)", 8, 650, 5200, 8.00, "Economically Prohibitive"],
        ["SymboLM (Greedy Mode)", 1, 55, 55, 0.08, "12× Cheaper than Baseline"],
        ["SymboLM (Best-of-8 Parallel Majority)", 8, 55, 440, 0.68, "32.3% Cheaper than Single English Trace!"]
    ]
    for r_i, row in enumerate(se_data, start=10):
        for c_i, val in enumerate(row, start=1):
            cell = ws3.cell(row=r_i, column=c_i, value=val)
            cell.font = regular_font
            cell.border = thin_border
            if r_i == 13:
                cell.fill = success_fill
                cell.font = bold_font

    # ---------------- SHEET 4: DeepSeek 224x Co-Design ----------------
    ws4 = wb.create_sheet(title="DeepSeek_CoDesign_224x")
    ws4.views.sheetView[0].showGridLines = True

    ws4["A1"] = "Frontier Hardware Co-Design: DeepSeek-V4.1-Flash × SymboLM Compound Multiplier"
    ws4["A1"].font = Font(name="Calibri", size=14, bold=True, color="0F172A")

    ds_h = ["Architectural Configuration", "KV Cache Density (B/tok)", "Reasoning Tokens", "Active Sequence Memory (KB)", "Active Sequence Memory (MB)", "Serving Footprint Reduction Multiplier"]
    for c_i, h in enumerate(ds_h, start=1):
        cell = ws4.cell(row=3, column=c_i, value=h)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    ds_data = [
        ["Standard Decoder-Only (Llama / Qwen)", 4000, 2000, 8000.0, 8.000, "1.0× (Baseline)"],
        ["DeepSeek-V4.1-Flash Alone", 890, 2000, 1780.0, 1.780, "4.5× Hardware Reduction"],
        ["SymboLM Alone (Standard Base)", 4000, 40, 160.0, 0.160, "50.0× Sequence Reduction"],
        ["SymboLM + DeepSeek-V4.1-Flash Co-Design", 890, 40, 35.6, 0.0356, "224.7× COMPOUND SERVING REDUCTION!"]
    ]
    for r_i, row in enumerate(ds_data, start=4):
        for c_i, val in enumerate(row, start=1):
            cell = ws4.cell(row=r_i, column=c_i, value=val)
            cell.font = regular_font
            cell.border = thin_border
            if r_i == 7:
                cell.fill = success_fill
                cell.font = bold_font

    # ---------------- SHEET 5: Grant Budget Allocation ----------------
    ws5 = wb.create_sheet(title="Grant_Budget_Allocation")
    ws5.views.sheetView[0].showGridLines = True

    ws5["A1"] = "Emergent Ventures & 1517 Fund Grant Budget Allocation ($20,000 – $25,000)"
    ws5["A1"].font = Font(name="Calibri", size=14, bold=True, color="0F172A")

    bg_h = ["Category", "Allocation ($ USD)", "Share (%)", "Operational Impact & Milestone Deliverables"]
    for c_i, h in enumerate(bg_h, start=1):
        cell = ws5.cell(row=3, column=c_i, value=h)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    bg_data = [
        ["1. Electrical & Power Resilience (Rupandehi, Nepal)", 5000, 0.20, "Pure sine-wave inverter, LiFePO4 battery storage (0ms switchover), and dual-WAN failover router to ensure continuous connection during regional grid blackouts."],
        ["2. Dedicated Cloud Compute Clusters", 12000, 0.48, "Dedicated A100/H100 GPU clusters (Lambda Labs / RunPod) to scale SymboLM from 1.5B parameters to 14B and 32B foundation models, executing multi-thousand sample GRPO rollouts."],
        ["3. Living & Physical Care Runway (12 Months)", 8000, 0.32, "12-month domestic and caregiving assistance runway in Rupandehi, Nepal, providing physical autonomy to conduct full-time independent AI research."]
    ]
    for r_i, row in enumerate(bg_data, start=4):
        for c_i, val in enumerate(row, start=1):
            cell = ws5.cell(row=r_i, column=c_i, value=val)
            cell.font = regular_font
            cell.border = thin_border
            if c_i == 2:
                cell.number_format = "$#,##0"
                cell.alignment = Alignment(horizontal="right")
            elif c_i == 3:
                cell.number_format = "0.0%"
                cell.alignment = Alignment(horizontal="right")

    # Total row
    ws5.cell(row=7, column=1, value="TOTAL UNRESTRICTED FELLOWSHIP").font = bold_font
    ws5.cell(row=7, column=2, value="=SUM(B4:B6)").font = accent_font
    ws5.cell(row=7, column=2).number_format = "$#,##0"
    ws5.cell(row=7, column=3, value="=SUM(C4:C6)").font = bold_font
    ws5.cell(row=7, column=3).number_format = "0.0%"
    ws5.cell(row=7, column=4, value="Permanent elimination of power, compute, and physical bottlenecks.").font = bold_font

    for c_i in range(1, 5):
        cell = ws5.cell(row=7, column=c_i)
        cell.fill = highlight_fill
        cell.border = thin_border

    # Auto-adjust column widths across all sheets
    for ws in wb.worksheets:
        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 48)

    output_path = "c:/Users/user/Dev/symboLM/docs/SymboLM_Benchmark_Data_and_Math_Evaluation.xlsx"
    wb.save(output_path)
    print(f"Successfully generated XLSX at {output_path}")

if __name__ == "__main__":
    create_workbook()
