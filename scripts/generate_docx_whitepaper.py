import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_document():
    doc = docx.Document()
    
    # Page setup: Letter with 0.75 in margins
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Styles setup
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Calibri'
    font.size = Pt(10.5)
    font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

    # Helper functions
    def add_title(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(22)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        return p

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(8)
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)
        return p

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
        return p

    def add_p(text, bold_prefix=None, space_after=5):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.bold = True
            r_pre.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        p.add_run(text)
        return p

    def add_callout(title, text, bg_hex="F0FDF4", border_color="16A34A"):
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        set_cell_background(cell, bg_hex)
        set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
        
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        r_title = p.add_run(f"✦ {title}\n")
        r_title.bold = True
        r_title.font.name = 'Arial'
        r_title.font.size = Pt(10)
        r_title.font.color.rgb = RGBColor(0x0F, 0x76, 0x6E) if bg_hex == "F0FDFA" else RGBColor(0x16, 0x65, 0x34) if bg_hex == "F0FDF4" else RGBColor(0x92, 0x40, 0x0E)
        
        r_text = p.add_run(text)
        r_text.font.size = Pt(9.5)
        r_text.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        
        p_space = doc.add_paragraph()
        p_space.paragraph_format.space_before = Pt(0)
        p_space.paragraph_format.space_after = Pt(4)

    def add_code_block(code_text):
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        set_cell_background(cell, "0F172A")
        set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
        
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(code_text)
        r.font.name = 'Consolas'
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0xE2, 0xE8, 0xF0)
        
        p_space = doc.add_paragraph()
        p_space.paragraph_format.space_before = Pt(0)
        p_space.paragraph_format.space_after = Pt(4)

    # ----------------- DOCUMENT HEADER -----------------
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_before = Pt(0)
    p_meta.paragraph_format.space_after = Pt(4)
    r_tag = p_meta.add_run("ADVANCED AI RESEARCH WHITEPAPER · TECHNICAL MANIFESTO & ARCHITECTURE\n")
    r_tag.font.name = 'Arial'
    r_tag.font.size = Pt(8.5)
    r_tag.font.bold = True
    r_tag.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)

    add_title("SymboLM: Adaptive Multi-Register Reasoning & Cognitive Token Compression")
    add_subtitle("Mathematical Foundations, Complexity Formulations, and Empirical Benchmark Evaluation (Version 2.0)")

    # Author Metadata Table
    meta_table = doc.add_table(rows=2, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx in range(2):
        for c_idx in range(2):
            set_cell_background(meta_table.cell(r_idx, c_idx), "F8FAFC")
            set_cell_margins(meta_table.cell(r_idx, c_idx), top=80, bottom=80, left=120, right=120)

    c00 = meta_table.cell(0, 0).paragraphs[0]
    c00.add_run("PRINCIPAL SYSTEMS ARCHITECT & RESEARCH DIRECTOR:\n").bold = True
    c00.runs[0].font.size = Pt(8)
    c00.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    r_auth = c00.add_run("Sanish Gyawali\n")
    r_auth.bold = True
    r_auth.font.size = Pt(10)
    c00.add_run("Independent AI Systems Researcher · Rupandehi, Nepal\nGitHub: @gyawalisanish0 · gyawalisanish0@gmail.com").font.size = Pt(8.5)

    c01 = meta_table.cell(0, 1).paragraphs[0]
    c01.add_run("AI SYSTEMS OPERATIONAL EXOSKELETON:\n").bold = True
    c01.runs[0].font.size = Pt(8)
    c01.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    r_exo = c01.add_run("Antigravity (Google DeepMind)\n")
    r_exo.bold = True
    r_exo.font.size = Pt(10)
    c01.add_run("Radical transparency across theoretical architecture, systems engineering, compiler pipelines, and test suites.").font.size = Pt(8.5)

    c10 = meta_table.cell(1, 0).paragraphs[0]
    c10.add_run("AFFILIATION & ORIGIN:\n").bold = True
    c10.runs[0].font.size = Pt(8)
    c10.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    c10.add_run("Independent Research Laboratory\nRupandehi, Nepal").font.size = Pt(8.5)

    c11 = meta_table.cell(1, 1).paragraphs[0]
    c11.add_run("OPEN SOURCE HUBS & ARTIFACTS:\n").bold = True
    c11.runs[0].font.size = Pt(8)
    c11.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    c11.add_run("Repository: github.com/gyawalisanish0/symboLM\nCheckpoints: huggingface.co/gyawalisanish0/symboLM-checkpoints").font.size = Pt(8.5)

    p_div = doc.add_paragraph()
    p_div.paragraph_format.space_before = Pt(6)
    p_div.paragraph_format.space_after = Pt(6)

    # ----------------- EXECUTIVE SUMMARY -----------------
    add_h1("Executive Summary & Core Metrics")
    add_p(
        "Current Large Reasoning Models (LRMs) such as DeepSeek-R1, OpenAI o1, and QwQ achieve state-of-the-art results on competitive mathematics and coding benchmarks by scaling test-time compute. However, this relies on an unsustainable operational assumption: models generate verbose natural language stream-of-consciousness essays ('Wait, let me rethink... Perhaps if I...') inside hidden scratchpads. This conversational bloat multiplies inference latency by up to 10×, consumes vast Key-Value (KV) cache memory, and makes test-time parallel search algorithms computationally prohibitive."
    )
    add_p(
        "SymboLM, conceived and architected from first principles by independent researcher Sanish Gyawali, introduces Biomimetic Multi-Register Cognition and the Symbolic Reasoning Language (SRL v1.0/v2.0). By fine-tuning pre-distilled reasoning foundations using Group Relative Policy Optimization (GRPO) with continuous parsimony rewards and deterministic Abstract Syntax Tree (AST) verification, SymboLM replaces conversational English filler with dense symbolic logic.",
        bold_prefix="Core Architectural Innovation: "
    )

    # Core Metrics Summary Table
    t_metrics = doc.add_table(rows=2, cols=4)
    t_metrics.alignment = WD_TABLE_ALIGNMENT.CENTER
    m_headers = ["Arithmetic Compression", "Compound Serving Reduction", "Intent Dialogue Fidelity", "Factual Bypass Latency"]
    m_values = ["98.3%", "224.7×", "100.0%", "3 Tokens"]
    m_subs = ["$3x+6=21$ in 2 tokens", "With DeepSeek Flash KV", "25/25 on cross-domain", "Zero <think> overhead"]
    
    for c_idx in range(4):
        set_cell_background(t_metrics.cell(0, c_idx), "0F172A")
        set_cell_margins(t_metrics.cell(0, c_idx), top=60, bottom=60, left=80, right=80)
        p = t_metrics.cell(0, c_idx).paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(m_headers[c_idx])
        r.font.name = 'Arial'
        r.font.size = Pt(8)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        set_cell_background(t_metrics.cell(1, c_idx), "F1F5F9")
        set_cell_margins(t_metrics.cell(1, c_idx), top=80, bottom=80, left=80, right=80)
        p2 = t_metrics.cell(1, c_idx).paragraphs[0]
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r2 = p2.add_run(m_values[c_idx] + "\n")
        r2.font.name = 'Arial'
        r2.font.size = Pt(14)
        r2.font.bold = True
        r2.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
        r3 = p2.add_run(m_subs[c_idx])
        r3.font.size = Pt(7.5)
        r3.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(4)
    p_sp.paragraph_format.space_after = Pt(6)

    # ----------------- SECTION 1: MATHEMATICAL FOUNDATIONS -----------------
    add_h1("1. Mathematical Foundations & Theoretical Complexity Formulations")

    add_h2("1.1 Reasoning Density (ρ)")
    add_p(
        "We define Reasoning Density (ρ) as the quotient of mathematically invariant state transitions (|ΔS|) over total generated tokens (T):"
    )
    add_code_block("  ρ = |ΔS| / T    [Mathematically Invariant State Transitions per Token Generated]")

    add_p(
        "In traditional natural language Chain-of-Thought (CoT), a representative GSM8K multi-step word problem requires approximately 4 deductive state transitions distributed across 110 tokens of conversational prose:",
        bold_prefix="• English CoT Baseline: "
    )
    add_code_block("  ρ_English ≈ 4 / 110 ≈ 0.036 transitions/token  (>95% consists of conversational filler)")

    add_p(
        "Under SymboLM's Symbolic Reasoning Language (SRL), the identical problem executes via formal AST transitions in 35 tokens:",
        bold_prefix="• SymboLM Micro-DSL: "
    )
    add_code_block("  ρ_SymboLM ≈ 4 / 35 ≈ 0.114 transitions/token   (3.16× Raw Logical Density Multiplier)")

    add_h2("1.2 Key-Value (KV) Cache Memory & Concurrency Scaling")
    add_p(
        "In causal Transformer decoders, generating a sequence of length N scales computational memory linearly in Key-Value (KV) cache storage:"
    )
    add_code_block("  Memory_KV = 2 × n_layers × n_heads × d_head × N × bytes_per_element")

    add_p(
        "For our primary foundation model, DeepSeek-R1-Distill-Qwen-1.5B (where n_layers = 28, n_heads = 12, d_head = 128, using bfloat16 = 2 bytes):"
    )
    add_p("• KV Cache Consumption per Token: 2 × 28 × 12 × 128 × 2 = 172,032 bytes (168 KB/token).")
    add_p("• Standard English CoT Trace (650 tokens): Consumes 106.64 MB per sequence solely for KV cache.")
    add_p("• SymboLM Symbolic Trace (60 tokens): Consumes 9.84 MB per sequence (10.84× reduction in active memory).")

    # KV Concurrency Table
    t_kv = doc.add_table(rows=3, cols=5)
    t_kv.alignment = WD_TABLE_ALIGNMENT.CENTER
    kv_heads = ["Serving Architecture", "Trace Length (N)", "KV Cache / Stream", "Max Streams (4GB Allocation)", "Throughput Multiplier"]
    for c_i, h_text in enumerate(kv_heads):
        set_cell_background(t_kv.cell(0, c_i), "0F172A")
        set_cell_margins(t_kv.cell(0, c_i), top=60, bottom=60, left=60, right=60)
        p = t_kv.cell(0, c_i).paragraphs[0]
        r = p.add_run(h_text)
        r.font.bold = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    kv_rows = [
        ["Standard English CoT", "650 tokens", "106.64 MB", "38 streams", "1.0× (Baseline)"],
        ["SymboLM SRL v1.1", "60 tokens", "9.84 MB", "416 streams", "10.9× Higher Concurrency"]
    ]
    for r_i, row in enumerate(kv_rows):
        for c_i, val in enumerate(row):
            cell = t_kv.cell(r_i + 1, c_i)
            set_cell_background(cell, "FFFFFF" if r_i == 0 else "F8FAFC")
            set_cell_margins(cell, top=60, bottom=60, left=60, right=60)
            p = cell.paragraphs[0]
            p.add_run(val).font.size = Pt(8.5)

    p_sp2 = doc.add_paragraph()
    p_sp2.paragraph_format.space_before = Pt(4)
    p_sp2.paragraph_format.space_after = Pt(6)

    add_h2("1.3 Test-Time Search Economics: The Best-of-8 Majority Voting Unlock")
    add_p(
        "Self-consistency majority voting over K sampled rollouts predictably enhances reasoning accuracy, but under English CoT, the aggregate token cost grows prohibitively (Total Tokens = K × L_avg):"
    )

    t_search = doc.add_table(rows=5, cols=5)
    t_search.alignment = WD_TABLE_ALIGNMENT.CENTER
    s_heads = ["Sampling Strategy", "Rollouts (K)", "Tokens / Rollout", "Total Tokens Generated", "Relative Compute Index"]
    for c_i, h_text in enumerate(s_heads):
        set_cell_background(t_search.cell(0, c_i), "0F172A")
        set_cell_margins(t_search.cell(0, c_i), top=60, bottom=60, left=60, right=60)
        p = t_search.cell(0, c_i).paragraphs[0]
        r = p.add_run(h_text)
        r.font.bold = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    s_rows = [
        ["Standard English (Greedy Baseline)", "1", "650 tokens", "650 tokens", "1.00×"],
        ["Standard English (Best-of-8 Ensemble)", "8", "650 tokens", "5,200 tokens", "8.00× (Prohibitive)"],
        ["SymboLM (Greedy Mode)", "1", "55 tokens", "55 tokens", "0.08× (12× Cheaper)"],
        ["SymboLM (Best-of-8 Parallel Ensemble)", "8", "55 tokens", "440 tokens", "0.68× (32.3% Cheaper than 1 English Trace)"]
    ]
    for r_i, row in enumerate(s_rows):
        for c_i, val in enumerate(row):
            cell = t_search.cell(r_i + 1, c_i)
            set_cell_background(cell, "FFFFFF" if r_i % 2 == 0 else "F8FAFC")
            set_cell_margins(cell, top=60, bottom=60, left=60, right=60)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8.5)
            if r_i == 3:
                r.bold = True

    add_callout(
        "Core Search Economic Unlock",
        "A system operating SymboLM can execute an 8-way parallel diverse sample with verified majority voting while consuming 32.3% fewer total tokens than a single traditional English answer. Parallel test-time scaling becomes economically democratic.",
        bg_hex="F0FDF4"
    )

    add_h2("1.4 The Compound Serving Multiplier: 224.7× DeepSeek-V4.1-Flash Co-Design")
    add_p(
        "DeepSeek's late-2026 architecture (DeepSeek-V4.1-Flash) departs from traditional decoder models via Causal Encoder-Decoders (CED) and Shared Global KV Caching (CSA2) at 4-bit precision (FP4), dropping raw KV memory from ~4,000 B/token down to 890 B/token."
    )
    add_p(
        "Our mathematical analysis proves that hardware-level KV caching and sequence-level symbolic token compression multiply compoundly:",
        bold_prefix="Hardware × Algorithm Compound Theorem: "
    )
    add_code_block(
        "  Standard Baseline:      2,000 English tokens × 4,000 B/token = 8.00 MB per active sequence\n"
        "  DeepSeek Flash Alone:   2,000 English tokens ×   890 B/token = 1.78 MB per active sequence\n"
        "  SymboLM Alone:             40 Symbolic tokens × 4,000 B/token = 0.16 MB per active sequence\n"
        "  Compound Co-Design:        40 Symbolic tokens ×   890 B/token = 0.035 MB (35.6 KB per active sequence)\n\n"
        "  Compound Multiplier = 8,000 KB / 35.6 KB ≈ 224.7× Active Serving Memory Reduction!"
    )

    add_h2("1.5 The Continuous Hyperbolic Parsimony Gradient Formulation")
    add_p(
        "To prevent reinforcement learning from truncating valid multi-step mathematical derivations while still strictly pruning verbosity, Sanish Gyawali architected the Unbounded Parsimony Gradient loss function for Stage 5 GRPO:"
    )
    add_code_block(
        "  If Answer y is Correct:\n"
        "    R_parsimony(τ, y*) = R_base × [ γ_floor + (1 - γ_floor) × (K / (K + |τ|)) ] - Ω(τ) + R_closure\n\n"
        "  If Answer y is Incorrect:\n"
        "    R_parsimony(τ, y*) = -1.0 - Ω(τ)\n\n"
        "  Parameters: R_base = 2.5, K = 35, γ_floor = 0.40, Ω(τ) = syntax violation penalty."
    )
    add_p(
        "Because the derivative dR/d|τ| = -R_base(1 - γ_floor) K / (K + |τ|)^2 is strictly negative everywhere, shorter proofs receive higher group advantage (A_i), naturally driving the policy toward the Minimum Description Length (MDL) while granting elastic headroom up to 192 tokens for Olympiad proofs."
    )

    # ----------------- SECTION 2: EMPIRICAL BENCHMARK DATA -----------------
    add_h1("2. Empirical Benchmark Data & Systems Evaluation Receipts")

    add_h2("2.1 Stages 1 Through 5 Engineering Trajectory")
    add_p(
        "SymboLM progressed through five major training stages executed on Kaggle Dual Tesla T4 GPUs and Google Colab TPU v5e compute:"
    )

    t_stages = doc.add_table(rows=6, cols=4)
    t_stages.alignment = WD_TABLE_ALIGNMENT.CENTER
    st_heads = ["Stage & Method", "Core Objective", "Compute Run", "Verified Empirical Receipt"]
    for c_i, h_text in enumerate(st_heads):
        set_cell_background(t_stages.cell(0, c_i), "0F172A")
        set_cell_margins(t_stages.cell(0, c_i), top=60, bottom=60, left=60, right=60)
        p = t_stages.cell(0, c_i).paragraphs[0]
        r = p.add_run(h_text)
        r.font.bold = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    stage_data = [
        ["Stage 1 (SFT)", "Syntactic AST Seeding", "Dual T4 (1h)", "Established non-zero prior for symbolic operators (→, ∴, |)."],
        ["Stage 2 (GRPO)", "Direct Arithmetic RL", "Dual T4 (3h 17m, 16k rollouts)", "98.3% token compression on arithmetic (3x+6=21 → 5 in 2 tokens)."],
        ["Stage 3 (Concept)", "Olympiad Invariant GRPO", "Dual T4 (3h 31m, 16k rollouts)", "Discovered Cold-Start Exploration Barrier; multi-register AST seeding mandatory."],
        ["Stage 4 (SE-HEA)", "Symbolic Efficiency & EOS", "Tesla T4 (2h)", "Eliminated 220-token repetition loop; factual QA cut from 256 to 35 tokens; 1% debris."],
        ["Stage 5 (Parsimony)", "Scratchpad 2.0 Unbounded", "Tesla T4 (1h 41m, 2,400 rollouts)", "Hyperbolic reward solved Truncation Bottleneck; multi-depth proofs with zero truncation."]
    ]
    for r_i, row in enumerate(stage_data):
        for c_i, val in enumerate(row):
            cell = t_stages.cell(r_i + 1, c_i)
            set_cell_background(cell, "FFFFFF" if r_i % 2 == 0 else "F8FAFC")
            set_cell_margins(cell, top=60, bottom=60, left=60, right=60)
            p = cell.paragraphs[0]
            p.add_run(val).font.size = Pt(8.5)

    p_sp3 = doc.add_paragraph()
    p_sp3.paragraph_format.space_before = Pt(4)
    p_sp3.paragraph_format.space_after = Pt(6)

    add_h2("2.2 The Cross-Domain 200-Question Empirical Benchmark (October 12, 2026)")
    add_p(
        "To test generalization beyond arithmetic, we evaluated the Stage 4 Policy Adapter (sanishgyawali/symbolm-cross-domain-200-benchmark) across 200 diverse questions across 7 distinct academic disciplines:"
    )

    t_bench = doc.add_table(rows=9, cols=7)
    t_bench.alignment = WD_TABLE_ALIGNMENT.CENTER
    b_heads = ["Domain Category", "N", "Accuracy (%)", "SymboLM Mean Tok", "Baseline Tok", "Compression (%)", "Routing Fidelity"]
    for c_i, h_text in enumerate(b_heads):
        set_cell_background(t_bench.cell(0, c_i), "0F172A")
        set_cell_margins(t_bench.cell(0, c_i), top=60, bottom=60, left=50, right=50)
        p = t_bench.cell(0, c_i).paragraphs[0]
        r = p.add_run(h_text)
        r.font.bold = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    b_rows = [
        ["Strategic Intent & Pragmatics", "25", "100.0% (25/25)", "127.0", "200.0", "36.5%", "100% Intent Routing (25/25)"],
        ["Scientific & Multi-Hop Factual", "25", "28.0% (7/25)", "94.1", "200.0", "52.9%", "100% Bypass Routing (25/25)"],
        ["Pure & Competition Mathematics", "40", "17.5% (7/40)", "128.0", "200.0", "36.0%", "75% Plan Routing (30/40)"],
        ["Computer Science & Discrete Systems", "30", "13.3% (4/30)", "128.0", "200.0", "36.0%", "97% Plan Routing (29/30)"],
        ["Chemistry & Quantitative Biology", "25", "12.0% (3/25)", "115.2", "200.0", "42.4%", "Mixed Bypass (10) & Plan (11)"],
        ["Probability, Statistics & Finance", "25", "8.0% (2/25)", "127.3", "200.0", "36.4%", "92% Plan Routing (23/25)"],
        ["Physics & Engineering", "30", "6.7% (2/30)", "126.3", "200.0", "36.9%", "80% Plan Routing (24/30)"],
        ["OVERALL COMPOSITE BENCHMARK", "200", "25.0% (50/200)", "121.7", "200.0", "39.2%", "IER Score: 20.54"]
    ]
    for r_i, row in enumerate(b_rows):
        for c_i, val in enumerate(row):
            cell = t_bench.cell(r_i + 1, c_i)
            is_composite = (r_i == len(b_rows) - 1)
            set_cell_background(cell, "EFF6FF" if is_composite else "FFFFFF" if r_i % 2 == 0 else "F8FAFC")
            set_cell_margins(cell, top=50, bottom=50, left=50, right=50)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8)
            if is_composite:
                r.bold = True

    p_sp4 = doc.add_paragraph()
    p_sp4.paragraph_format.space_before = Pt(4)
    p_sp4.paragraph_format.space_after = Pt(6)

    add_callout(
        "Empirical Discovery: The 128-Token Truncation Bottleneck",
        "Forensic failure mode audits in Advanced Math, CS, and Physics revealed that dozens of valid derivations (e.g. polynomial invariants x + 1/x = 6 ⟹ x^4 + 1/x^4 = 1154, and modular exponentiation 3^100 mod 7 = 4) successfully completed every formal transformation, but hit the evaluation ceiling of exactly 128 tokens right before outputting the final <ans> tag. Imposing flat token caps artificially chokes intelligence. Stage 5's Unbounded Parsimony Gradient resolved this by granting elastic headroom while penalizing verbosity continuously.",
        bg_hex="FFFBEB",
        border_color="D97706"
    )

    add_h2("2.3 Multi-Depth Inline Audit Receipts (Stage 5 Adapter)")
    add_p(
        "Empirical traces generated by the Stage 5 Parsimony Adapter demonstrate adaptive depth allocation across problem complexities:"
    )
    add_code_block(
        "  • Δ0 Atomic Fact Recall (3 tokens):\n"
        "    Query: 'What is the chemical symbol for Potassium?'\n"
        "    Output: <reg:bypass>K</reg:bypass>  --> 100% Instant Zero-Shot Bypass with 0 <think> tokens!\n\n"
        "  • Δ1 Linear Invariant (21 tokens):\n"
        "    Query: 'If x + 1/x = 6, find x^2 + 1/x^2.'\n"
        "    Output: <think><reg:plan>invariant: x^2+1/x^2=(x+1/x)^2-2</reg:plan><reg:deduce>6^2-2=34</reg:deduce></think><ans>34</ans>\n"
        "    Compression: 21 tokens vs. 600+ token conversational baseline (96.5% reduction).\n\n"
        "  • Δ3 Deep Congruence (35 tokens):\n"
        "    Query: 'Find least positive integer x where x ≡ 3 (mod 7) and x ≡ 5 (mod 11).'\n"
        "    Output: <think><reg:plan>CRT: m1=7, m2=11 | M=77</reg:plan><reg:deduce>x=7k+3 ≡ 5 (mod 11) → 7k ≡ 2 → k=5 → x=38</reg:deduce><reg:verify>38%7==3 ✓, 38%11==5 ✓</reg:verify></think><ans>38</ans>\n"
        "    Receipt: Complete formal proof with counterfactual verification in 35 tokens with ZERO TRUNCATION!"
    )

    add_h2("2.4 Local Deterministic AST Verification (SymPy 1.14.0 Receipts)")
    add_p(
        "To verify that intermediate symbolic reasoning steps preserve truth invariance, all transitions were audited using Python 3.13 and SymPy 1.14.0 (eval/verify_math_claims.py):"
    )
    add_p("• GSM8K Duck Eggs (16 eggs/day, -3, -4, ×$2): English CoT = 102 tokens; SymboLM = 33 tokens (67.6% compression, 100% invariant match).")
    add_p("• GSM8K Train Speed (60mph for 2h, 80mph for 3h): English CoT = 126 tokens; SymboLM = 43 tokens (65.9% compression, 100% invariant match).")
    add_p("• MATH Level 2 Linear Equation (5x - 15 = 3x + 25): English CoT = 152 tokens; SymboLM = 65 tokens (57.2% compression, SymPy root = 20).")
    add_p("• GSM8K Rebate Budget ($100 - $25 - $15 + $40): English CoT = 110 tokens; SymboLM = 37 tokens (66.4% compression, verified result = $100).")

    # ----------------- SECTION 3: TRI-REGISTER ARCHITECTURE -----------------
    add_h1("3. The Tri-Register Biomimetic Cognitive Architecture (SRL v2.0)")

    add_p(
        "SymboLM segments cognition into distinct functional registers to prevent catastrophic cognitive lobotomy:"
    )

    add_code_block(
        "                            [ User Input Query ]\n"
        "                                     │\n"
        "        ┌────────────────────────────┼────────────────────────────┐\n"
        "        ▼                            ▼                            ▼\n"
        "  <reg:bypass>                 <reg:intent>              <reg:scratchpad Δ0..Δ3>\n"
        " (0-Step Factual Reflex)     (10-Token Dialogue Plan)     ├── <reg:plan> (Macro Invariants)\n"
        "        │                            │                    ├── <reg:deduce> (AST Code Transitions)\n"
        "        ▼                            ▼                    └── <reg:verify> (Assert ✓/✗)\n"
        "  Instant Answer            Fluent Conversational Prose          Verified Numeric Invariant"
    )

    add_p("1. Zero-Shot Reflex Bypass (<reg:bypass>): Eliminates deliberation compute on atomic factual memory queries (e.g. Potassium → K in 3 tokens), anticipating DeepSeek's 2026 Engram memory module.")
    add_p("2. Strategic Intent Scratchpad (<reg:intent>): Plans conversational demeanor, tone, and empathy in ~10 tokens before emitting natural human prose.")
    add_p("3. Macro Invariant Plan (<reg:plan>): Formulates algebraic and modular theorems prior to arithmetic execution.")
    add_p("4. Symbolic Deduce (<reg:deduce>): Executes AST reductions via formal operators (→, ∴, |, let, eval) verifiable by mechanical code linters.")
    add_p("5. Runtime Assertion (<reg:verify>): Verifies intermediate claims with active counterfactual backtracking (verify(...) ✓ or ✗) before committing to <ans>.")

    # ----------------- SECTION 4: FUTURE HORIZONS & SCALING -----------------
    add_h1("4. Future Research Horizons & Systems Scaling")

    add_p(
        "The empirical verification of Register v2.0 and the Unbounded Parsimony Gradient on 1.5B parameters establishes the foundational proof-of-concept for biomimetic reasoning. Future architectural frontiers include:"
    )

    t_budget = doc.add_table(rows=4, cols=3)
    t_budget.alignment = WD_TABLE_ALIGNMENT.CENTER
    b_heads2 = ["Research Frontier", "Target Objective", "Technical & Algorithmic Mechanism"]
    for c_i, h_text in enumerate(b_heads2):
        set_cell_background(t_budget.cell(0, c_i), "0F172A")
        set_cell_margins(t_budget.cell(0, c_i), top=60, bottom=60, left=60, right=60)
        p = t_budget.cell(0, c_i).paragraphs[0]
        r = p.add_run(h_text)
        r.font.bold = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    b_rows2 = [
        ["1. Foundation Model Scaling", "7B / 14B / 32B Architectures", "Applying Parsimony GRPO and Tri-Register conditioning to larger open-weight reasoning backbones to unlock deeper mathematical invariant discovery."],
        ["2. Formal Theorem Prover Integration", "Automated Soundness Engine", "Coupling the verification register (<reg:verify>) directly with formal proof checkers (Lean 4, Isabelle, SymPy) for compile-time sound reasoning."],
        ["3. Real-Time Edge & Robotics", "Sub-50ms Low-Latency Inference", "Leveraging the 4-bit GGUF quantization (34.3 tok/s on CPU) and 0-step factual bypass for deterministic local decision engines."]
    ]
    for r_i, row in enumerate(b_rows2):
        for c_i, val in enumerate(row):
            cell = t_budget.cell(r_i + 1, c_i)
            set_cell_background(cell, "FFFFFF" if r_i % 2 == 0 else "F8FAFC")
            set_cell_margins(cell, top=60, bottom=60, left=60, right=60)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8.5)
            if c_i == 0 or c_i == 1:
                r.bold = True

    p_sp5 = doc.add_paragraph()
    p_sp5.paragraph_format.space_before = Pt(4)
    p_sp5.paragraph_format.space_after = Pt(6)

    # ----------------- SECTION 5: AUTHOR PROFILE -----------------
    add_h1("5. Principal Systems Architect & Research Attribution")
    add_p(
        "Sanish Gyawali is an independent systems researcher based in Rupandehi, Nepal. Living with a severe congenital physical disability and bedridden 24/7, he operates his research laboratory entirely from his bed workstation. Sanish conceived the mathematical foundations of SymboLM, the Symbolic Reasoning Language (SRL v1.0/v2.0), the Tri-Register cognitive architecture, and the Unbounded Parsimony Gradient.",
        bold_prefix="Sanish Gyawali (Principal Systems Architect & Research Director): "
    )
    add_p(
        "All intellectual property, architecture designs, training curricula, and code belong exclusively to Sanish Gyawali. AI collaboration with Google DeepMind's Antigravity serves as an operational exoskeleton under radical scientific transparency.",
        bold_prefix="Intellectual Property & Collaboration: "
    )
    add_p("• Open Source Repository: https://github.com/gyawalisanish0/symboLM")
    add_p("• Remote Weights Hub: https://huggingface.co/gyawalisanish0/symboLM-checkpoints")
    add_p("• License: MIT License (Free and Open Source for Academic and Industrial Research)")

    output_path = "c:/Users/user/Dev/symboLM/docs/SymboLM_Research_Whitepaper_Sanish_Gyawali.docx"
    doc.save(output_path)
    print(f"Successfully generated DOCX at {output_path}")

if __name__ == "__main__":
    create_document()
