# SymboLM Benchmark Contamination & Leakage Audit Report (Task 4)

**Audit Protocol:** Claude Haiku 5.5 Empirical Auditing Protocol  
**Author & Principal Systems Architect:** Sanish Gyawali (Rupandehi, Nepal)  
**AI Systems Collaborator:** Antigravity (Google DeepMind)  
**Date:** October 17, 2026  

---

## 1. Executive Summary

| Partition | Question Count | Proportion of Benchmark | Honest Rescored Accuracy |
| :--- | :---: | :---: | :---: |
| **Exact Verbatim Prompt Matches** | **47** | 4.7% | **34.0%** (16/47) |
| **8-Gram Partial Overlap (≥ 2 shared)** | **49** | 4.9% | **0.0%** (0/49) |
| **Strictly Clean (Out-of-Distribution)** | **904** | **90.4%** | **1.8%** (16/904) |

---

## 2. Key Findings

1. **Verbatim Leaks Identified:** Exactly **47 benchmark prompts** appeared verbatim in the curriculum data, directly confirming Claude's audit finding #7.
2. **Strictly Clean Majority:** **904 out of 1,000 questions (90.4%)** have zero exact matches and zero multi-token n-gram overlap with the training curriculum.
3. **Remediation Action:** All future evaluation partitions for Task 2 and beyond will strictly filter out the identified overlapping items to ensure zero contamination.
