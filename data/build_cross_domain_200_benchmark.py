"""
SymboLM Cross-Domain 200-Question Randomized Benchmark Generator
================================================================
Curates 200 diverse, high-rigor evaluation questions spanning multiple fields:
  1. Classical & Modern Physics (Mechanics, Circuits, Thermo, Relativity) - 30 questions
  2. Computer Science & Discrete Mathematics (Algorithms, Graphs, Trees, Logic) - 30 questions
  3. Chemistry & Quantitative Biology (Stoichiometry, Solutions, Genetics) - 25 questions
  4. Probability, Statistics & Financial Math (Bayes, Expected Value, Compounding) - 25 questions
  5. Pure & Competition Mathematics (Number Theory, Polynomials, Geometry) - 40 questions
  6. Scientific & Multi-Hop Factual Retrieval (Bypass vs Reasoning routing) - 25 questions
  7. Strategic Intent & Pragmatics (High-stakes dialogue, de-escalation) - 25 questions

Incorporates Cognitive Depth Spectrum annotations (Delta-0 to Delta-3)
to evaluate Sanish Gyawali's Gradient Register & Scratchpad 2.0 architecture.

Author: Sanish Gyawali
AI Systems Exoskeleton: Antigravity (Google DeepMind)
"""

import json
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT_DIR / "data" / "eval_200_cross_domain_benchmark.jsonl"


def generate_benchmark():
    items = []

    # =========================================================================
    # 1. Classical & Modern Physics (30 questions)
    # =========================================================================
    physics_questions = [
        # Mechanics & Kinematics
        {"prompt": "A 1200 kg car accelerates from rest at a constant rate of 3 m/s^2 for 8 seconds. Find the work done on the car in joules.",
         "ground_truth": "345600", "depth": "Delta-1", "budget": 40, "baseline": 220},
        {"prompt": "A projectile is launched from ground level with velocity 40 m/s at an angle of 30 degrees above the horizontal. Assuming g = 9.8 m/s^2, find its maximum height in meters (round to two decimal places).",
         "ground_truth": "20.41", "depth": "Delta-1", "budget": 45, "baseline": 240},
        {"prompt": "A block of mass 5 kg slides down a frictionless 30-degree incline of length 10 meters. Assuming g = 9.8 m/s^2, what is its speed at the bottom in m/s (round to two decimal places)?",
         "ground_truth": "9.90", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "A 2 kg mass attached to a spring with k = 200 N/m undergoes simple harmonic motion. What is its angular frequency omega in rad/s?",
         "ground_truth": "10", "depth": "Delta-0", "budget": 20, "baseline": 150},
        {"prompt": "Calculate the escape velocity from Earth in km/s (assuming G = 6.674e-11, M = 5.972e24 kg, R = 6.371e6 m; round to one decimal place).",
         "ground_truth": "11.2", "depth": "Delta-1", "budget": 40, "baseline": 260},
        {"prompt": "Two masses of 3 kg and 7 kg undergo a completely inelastic head-on collision. Mass 1 moves at +10 m/s and mass 2 moves at -5 m/s. What is their final common velocity in m/s?",
         "ground_truth": "-0.5", "depth": "Delta-1", "budget": 35, "baseline": 220},
        {"prompt": "A satellite moves in a circular orbit around a planet of mass M = 4e24 kg at radius r = 1e7 m. If G = 6.67e-11 N m^2/kg^2, calculate its orbital speed in m/s (to nearest integer).",
         "ground_truth": "5165", "depth": "Delta-1", "budget": 45, "baseline": 250},
        {"prompt": "A wheel of moment of inertia I = 0.5 kg m^2 is acted upon by a torque tau = 4 N m for 6 seconds from rest. What is its final angular momentum in kg m^2/s?",
         "ground_truth": "24", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "A 0.5 kg ball drops from height 20 m and rebounds to height 5 m. Assuming g = 9.8 m/s^2, what is the impulse delivered to the ball by the ground in N s (round to two decimal places)?",
         "ground_truth": "14.85", "depth": "Delta-2", "budget": 55, "baseline": 280},
        {"prompt": "Find the period of a simple pendulum of length 2.45 meters on Earth where g = 9.8 m/s^2 in seconds (round to two decimal places).",
         "ground_truth": "3.14", "depth": "Delta-1", "budget": 35, "baseline": 200},

        # Circuits & Electromagnetism
        {"prompt": "In a series circuit with three resistors R1 = 12 ohms, R2 = 18 ohms, and R3 = 30 ohms connected across a 120V supply, find the voltage drop across R2 in volts.",
         "ground_truth": "36", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "Find the equivalent resistance in ohms of three resistors 10 ohms, 20 ohms, and 30 ohms connected in parallel (round to two decimal places).",
         "ground_truth": "5.45", "depth": "Delta-1", "budget": 40, "baseline": 230},
        {"prompt": "A parallel plate capacitor with capacitance 50 uF is charged to 200V. Find the energy stored in the capacitor in joules.",
         "ground_truth": "1", "depth": "Delta-0", "budget": 25, "baseline": 180},
        {"prompt": "An alternating voltage source V(t) = 170 sin(377 t) volts is connected across a 50-ohm resistor. What is the RMS power dissipated in watts (round to one decimal place)?",
         "ground_truth": "289.0", "depth": "Delta-1", "budget": 35, "baseline": 220},
        {"prompt": "A magnetic flux through a coil of 50 turns increases uniformly from 0.02 Wb to 0.14 Wb in 0.3 seconds. Find the magnitude of induced EMF in volts.",
         "ground_truth": "20", "depth": "Delta-1", "budget": 30, "baseline": 190},
        {"prompt": "An LC oscillator has L = 2 mH and C = 8 uF. What is the resonant frequency f in Hz (round to nearest integer)?",
         "ground_truth": "1258", "depth": "Delta-1", "budget": 45, "baseline": 240},
        {"prompt": "What is the magnetic field B in teslas inside a solenoid with 500 turns/m carrying a current of 4 A (take mu_0 = 4*pi*1e-7; round to three decimal places)?",
         "ground_truth": "0.003", "depth": "Delta-0", "budget": 25, "baseline": 190},
        {"prompt": "A transformer has 400 primary turns and 50 secondary turns. If the input primary voltage is 240V AC, find the output secondary voltage in volts.",
         "ground_truth": "30", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "Two parallel conductors 0.1 m apart carry currents of 15 A and 20 A in the same direction. What is the force per unit length between them in N/m (round to five decimal places)?",
         "ground_truth": "0.00060", "depth": "Delta-1", "budget": 40, "baseline": 220},
        {"prompt": "Find the drift velocity of electrons in a copper wire of cross-section 2 mm^2 carrying 10 A current if electron density is 8.5e28 m^-3 and e = 1.6e-19 C in mm/s (round to three decimal places).",
         "ground_truth": "0.368", "depth": "Delta-2", "budget": 50, "baseline": 280},

        # Thermodynamics, Optics & Modern Physics
        {"prompt": "An ideal gas expands isothermally at 300 K from 2 L to 8 L. If n = 2 moles, calculate the work done by the gas in joules (take R = 8.314 J/mol K; round to integer).",
         "ground_truth": "6915", "depth": "Delta-2", "budget": 50, "baseline": 260},
        {"prompt": "A Carnot heat engine operates between temperatures of 500 K and 300 K. What is its maximum theoretical efficiency as a percentage?",
         "ground_truth": "40", "depth": "Delta-0", "budget": 20, "baseline": 150},
        {"prompt": "How much heat in kilojoules is required to raise the temperature of 2 kg of water from 20 deg C to 80 deg C (specific heat c = 4.184 kJ/kg K)?",
         "ground_truth": "502.08", "depth": "Delta-1", "budget": 35, "baseline": 200},
        {"prompt": "Light traveling in air (n=1) enters diamond (n=2.42). What is the critical angle for total internal reflection inside diamond in degrees (round to one decimal place)?",
         "ground_truth": "24.4", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "A convex lens has focal length f = 15 cm. An object is placed 30 cm in front of the lens. What is the image distance v in cm?",
         "ground_truth": "30", "depth": "Delta-1", "budget": 30, "baseline": 190},
        {"prompt": "Find the de Broglie wavelength in meters of an electron moving at 2e6 m/s (take m_e = 9.109e-31 kg, h = 6.626e-34 J s; express in scientific notation: A.BCe-10; round to two decimal places: e.g. 3.64e-10).",
         "ground_truth": "3.64e-10", "depth": "Delta-1", "budget": 45, "baseline": 250},
        {"prompt": "What is the relativistic Lorentz gamma factor for a particle moving at 0.8c?",
         "ground_truth": "1.67", "depth": "Delta-1", "budget": 30, "baseline": 190},
        {"prompt": "A radioactive sample has a half-life of 12 days. What fraction of the original sample remains after 48 days (express as fraction, e.g. 1/16)?",
         "ground_truth": "1/16", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "Calculate the energy in MeV released in complete annihilation of an electron-positron pair at rest (each rest mass = 0.511 MeV).",
         "ground_truth": "1.022", "depth": "Delta-0", "budget": 15, "baseline": 140},
        {"prompt": "A photon has wavelength 500 nm. Calculate its energy in electron-volts (take h*c = 1240 eV nm; round to two decimal places).",
         "ground_truth": "2.48", "depth": "Delta-0", "budget": 20, "baseline": 160},
    ]
    for q in physics_questions:
        q["domain"] = "Physics & Engineering"
    items.extend(physics_questions)

    # =========================================================================
    # 2. Computer Science & Discrete Mathematics (30 questions)
    # =========================================================================
    cs_questions = [
        # Complexity & Recurrences
        {"prompt": "Apply the Master Theorem to find the Big-Theta time complexity of T(n) = 4T(n/2) + n^2. State the exponent of n in the Theta bound (e.g. for n^2 log n, write 'n^2 log n').",
         "ground_truth": "n^2 log n", "depth": "Delta-2", "budget": 45, "baseline": 250},
        {"prompt": "What is the maximum number of nodes in a full binary tree of height 5 (where root is at height 0)?",
         "ground_truth": "63", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "How many comparisons does binary search make in the worst case for a sorted array of 1,000,000 elements?",
         "ground_truth": "20", "depth": "Delta-0", "budget": 20, "baseline": 170},
        {"prompt": "In an undirected connected planar graph with V = 12 vertices and F = 8 faces, how many edges E are there by Euler's planar formula?",
         "ground_truth": "18", "depth": "Delta-1", "budget": 30, "baseline": 190},
        {"prompt": "A graph has vertices with degrees: 2, 3, 3, 4, 4, 4, 6. How many edges does the graph have?",
         "ground_truth": "13", "depth": "Delta-1", "budget": 30, "baseline": 180},
        {"prompt": "How many non-isomorphic spanning trees does the complete graph K_4 possess according to Cayley's formula n^(n-2)?",
         "ground_truth": "16", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "What is the chromatic number of the complete bipartite graph K_{4, 7}?",
         "ground_truth": "2", "depth": "Delta-0", "budget": 15, "baseline": 140},
        {"prompt": "Convert the hexadecimal number 0x3FA to decimal.",
         "ground_truth": "1018", "depth": "Delta-1", "budget": 30, "baseline": 180},
        {"prompt": "What is the 8-bit two's complement binary representation of the decimal integer -43?",
         "ground_truth": "11010101", "depth": "Delta-1", "budget": 35, "baseline": 200},
        {"prompt": "Evaluate the bitwise operation in decimal: (45 & 27) | (14 ^ 3).",
         "ground_truth": "29", "depth": "Delta-1", "budget": 35, "baseline": 210},

        # Formal Logic & Boolean Algebra
        {"prompt": "Simplify the Boolean expression: A*B + A*B' + A'*B into its minimal SOP form.",
         "ground_truth": "A + B", "depth": "Delta-1", "budget": 30, "baseline": 190},
        {"prompt": "How many truth assignments satisfy the Boolean formula (A or B) and (not A or C) across variables A, B, C?",
         "ground_truth": "5", "depth": "Delta-2", "budget": 50, "baseline": 280},
        {"prompt": "Determine if the formula ((P -> Q) and (Q -> R)) -> (P -> R) is a tautology (answer: True or False).",
         "ground_truth": "True", "depth": "Delta-0", "budget": 20, "baseline": 170},
        {"prompt": "How many distinct binary boolean functions of 3 variables exist?",
         "ground_truth": "256", "depth": "Delta-0", "budget": 20, "baseline": 150},
        {"prompt": "What is the dual of the Boolean expression: (A + B) * (C + 0)?",
         "ground_truth": "A * B + C * 1", "depth": "Delta-1", "budget": 30, "baseline": 180},

        # Algorithms & Data Structures
        {"prompt": "What is the minimum number of keys in a B-tree of order m = 5 and height h = 2 (where root is height 0)?",
         "ground_truth": "17", "depth": "Delta-2", "budget": 50, "baseline": 260},
        {"prompt": "What is the length of the Longest Increasing Subsequence in the array [10, 22, 9, 33, 21, 50, 41, 60, 80]?",
         "ground_truth": "6", "depth": "Delta-2", "budget": 45, "baseline": 240},
        {"prompt": "In an AVL tree, after inserting keys 10, 20, 30 in that order into an initially empty tree, what key is at the root?",
         "ground_truth": "20", "depth": "Delta-0", "budget": 25, "baseline": 170},
        {"prompt": "How many inversions are there in the permutation [4, 1, 3, 2]?",
         "ground_truth": "4", "depth": "Delta-1", "budget": 30, "baseline": 180},
        {"prompt": "In a min-heap with 10 elements stored in a 0-indexed array, what is the array index of the right child of the element at index 3?",
         "ground_truth": "8", "depth": "Delta-0", "budget": 20, "baseline": 150},
        {"prompt": "What is the number of strongly connected components in a directed cycle graph of 7 vertices?",
         "ground_truth": "1", "depth": "Delta-0", "budget": 15, "baseline": 140},
        {"prompt": "How many topological orderings exist for a DAG consisting of 4 independent vertices with no edges?",
         "ground_truth": "24", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "What is the maximum flow from S to T in a network where S connects to A (cap 10) and B (cap 10), and A and B each connect to T (caps 8 and 7 respectively)?",
         "ground_truth": "15", "depth": "Delta-1", "budget": 35, "baseline": 200},
        {"prompt": "In Dijkstra's algorithm implemented with a binary min-heap for a graph with V vertices and E edges, what is the time complexity Big-O (express as O((V+E) log V))?",
         "ground_truth": "O((V+E) log V)", "depth": "Delta-0", "budget": 20, "baseline": 170},
        {"prompt": "What is the edit distance (Levenshtein) between 'kitten' and 'sitting'?",
         "ground_truth": "3", "depth": "Delta-2", "budget": 50, "baseline": 260},

        # Formal Grammars & Automata
        {"prompt": "How many states are in the minimal DFA that recognizes strings over {0,1} ending in '01'?",
         "ground_truth": "3", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "Is the language L = {a^n b^n c^n | n >= 1} context-free (answer: True or False)?",
         "ground_truth": "False", "depth": "Delta-0", "budget": 15, "baseline": 150},
        {"prompt": "What is the pumping lemma constant p property: if |w| >= p, then w = xyz such that |y| > 0 and |xy| <= p. For any i >= 0, xy^i z is in L (True or False)?",
         "ground_truth": "True", "depth": "Delta-0", "budget": 15, "baseline": 140},
        {"prompt": "How many equivalence classes does the Myhill-Nerode relation have for language L = {w in {0,1}* | |w| is a multiple of 3}?",
         "ground_truth": "3", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "What is the complement of a recursively enumerable language that is not recursive? Is it recursively enumerable (True or False)?",
         "ground_truth": "False", "depth": "Delta-0", "budget": 15, "baseline": 150},
    ]
    for q in cs_questions:
        q["domain"] = "Computer Science & Discrete Systems"
    items.extend(cs_questions)

    # =========================================================================
    # 3. Chemistry & Quantitative Biology (25 questions)
    # =========================================================================
    chem_questions = [
        # Stoichiometry & Gas Laws
        {"prompt": "Calculate the molar mass of Calcium Nitrate, Ca(NO3)2, in g/mol (use Ca=40.08, N=14.01, O=16.00; round to two decimal places).",
         "ground_truth": "164.10", "depth": "Delta-1", "budget": 35, "baseline": 200},
        {"prompt": "Balance the reaction: a C3H8 + b O2 -> c CO2 + d H2O. What is the sum of integer coefficients a + b + c + d in simplest terms?",
         "ground_truth": "13", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "How many moles of gas occupy 44.8 L at standard temperature and pressure (STP)?",
         "ground_truth": "2", "depth": "Delta-0", "budget": 20, "baseline": 150},
        {"prompt": "If 4 moles of H2 react completely with 1 mole of O2 to form H2O, which reactant is the limiting reagent (answer: H2 or O2)?",
         "ground_truth": "O2", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "Calculate the percent by mass of Nitrogen in Ammonium Nitrate, NH4NO3 (use N=14.01, H=1.01, O=16.00; round to one decimal place).",
         "ground_truth": "35.0", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "A sample of gas has volume 3.0 L at 2.0 atm. If compressed isothermally to 0.75 L, what is the new pressure in atm?",
         "ground_truth": "8.0", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "Calculate the density in g/L of Oxygen gas (O2, molar mass 32 g/mol) at STP (round to two decimal places).",
         "ground_truth": "1.43", "depth": "Delta-1", "budget": 30, "baseline": 180},
        {"prompt": "How many grams of NaCl are required to prepare 500 mL of a 0.25 M solution (molar mass NaCl = 58.44 g/mol; round to two decimal places)?",
         "ground_truth": "7.31", "depth": "Delta-1", "budget": 35, "baseline": 200},

        # Solutions & Equilibria
        {"prompt": "What is the pH of a 0.005 M solution of Ba(OH)2 assuming complete dissociation in water at 25 deg C (round to integer)?",
         "ground_truth": "12", "depth": "Delta-1", "budget": 35, "baseline": 220},
        {"prompt": "Find the [H+] ion concentration in mol/L for a solution with pH = 3.70 (express in scientific notation: A.BCe-4; round to two decimal places).",
         "ground_truth": "2.00e-4", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "A buffer solution contains 0.1 M acetic acid and 0.1 M sodium acetate. If Ka of acetic acid is 1.8e-5, what is the pH of the buffer (round to two decimal places)?",
         "ground_truth": "4.74", "depth": "Delta-1", "budget": 35, "baseline": 220},
        {"prompt": "What volume in mL of 12 M HCl is needed to prepare 600 mL of 0.5 M HCl?",
         "ground_truth": "25", "depth": "Delta-1", "budget": 25, "baseline": 170},
        {"prompt": "For the equilibrium N2 + 3H2 <=> 2NH3, if at equilibrium [N2]=0.5 M, [H2]=0.2 M, [NH3]=0.4 M, calculate Kc (round to one decimal place).",
         "ground_truth": "40.0", "depth": "Delta-2", "budget": 45, "baseline": 240},
        {"prompt": "What is the oxidation state of Chromium in the dichromate ion Cr2O7^(2-)?",
         "ground_truth": "+6", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "In an electrochemical cell with standard potentials E0(Zn2+/Zn) = -0.76 V and E0(Cu2+/Cu) = +0.34 V, what is the standard cell potential E0_cell in volts?",
         "ground_truth": "1.10", "depth": "Delta-0", "budget": 20, "baseline": 160},

        # Quantitative Biology & Genetics
        {"prompt": "In a population in Hardy-Weinberg equilibrium, the frequency of homozygous recessive genotype (q^2) is 0.09. What is the frequency of heterozygotes (2pq)?",
         "ground_truth": "0.42", "depth": "Delta-1", "budget": 30, "baseline": 190},
        {"prompt": "If a double-stranded DNA molecule contains 28% Adenine, what is the percentage of Cytosine according to Chargaff's rules?",
         "ground_truth": "22", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "In a dihybrid cross AaBb x AaBb with independent assortment, what fraction of progeny will be homozygous dominant for both genes (AABB)?",
         "ground_truth": "1/16", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "How many ATP molecules are net produced per glucose molecule during aerobic cellular respiration in typical eukaryotic cells (traditional value)?",
         "ground_truth": "36", "depth": "Delta-0", "budget": 15, "baseline": 140},
        {"prompt": "If a bacterial culture starts with 1,000 cells and doubles every 20 minutes, how many bacterial cells will be present after 2 hours?",
         "ground_truth": "64000", "depth": "Delta-1", "budget": 30, "baseline": 190},
        {"prompt": "How many base triplets (codons) specify amino acids out of the 64 possible mRNA codons in the standard genetic code?",
         "ground_truth": "61", "depth": "Delta-0", "budget": 15, "baseline": 140},
        {"prompt": "What is the maximum number of alleles for a single gene that a normal diploid individual can carry?",
         "ground_truth": "2", "depth": "Delta-0", "budget": 15, "baseline": 130},
        {"prompt": "Calculate the Michaelis-Menten initial rate V0 if Vmax = 100 umol/s, Km = 2 mM, and substrate [S] = 8 mM in umol/s.",
         "ground_truth": "80", "depth": "Delta-1", "budget": 30, "baseline": 190},
        {"prompt": "What is the respiratory quotient (RQ) for the complete oxidation of glucose (C6H12O6 + 6O2 -> 6CO2 + 6H2O)?",
         "ground_truth": "1.0", "depth": "Delta-0", "budget": 15, "baseline": 140},
        {"prompt": "If a father has blood type AB and mother has blood type O, what percentage of their children can have blood type AB?",
         "ground_truth": "0", "depth": "Delta-0", "budget": 20, "baseline": 160},
    ]
    for q in chem_questions:
        q["domain"] = "Chemistry & Quantitative Biology"
    items.extend(chem_questions)

    # =========================================================================
    # 4. Probability, Statistics & Financial Math (25 questions)
    # =========================================================================
    prob_questions = [
        {"prompt": "A disease has a prevalence of 1%. A diagnostic test has 95% sensitivity and 90% specificity. If a random person tests positive, what is the probability they actually have the disease (express as percentage rounded to one decimal place)?",
         "ground_truth": "8.8", "depth": "Delta-2", "budget": 60, "baseline": 290},
        {"prompt": "Two fair 6-sided dice are rolled. What is the probability that the sum is equal to 8 (express as simplest fraction)?",
         "ground_truth": "5/36", "depth": "Delta-1", "budget": 25, "baseline": 170},
        {"prompt": "What is the expected value of a random variable X with outcomes 2, 4, 8 with probabilities 0.5, 0.3, 0.2 respectively?",
         "ground_truth": "3.8", "depth": "Delta-0", "budget": 25, "baseline": 170},
        {"prompt": "If variance Var(X) = 9, what is the standard deviation of 4X + 7?",
         "ground_truth": "12", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "A coin is biased with P(Heads) = 0.6. If flipped 5 times, what is the probability of getting exactly 3 heads (round to four decimal places)?",
         "ground_truth": "0.3456", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "A Poisson random variable has mean lambda = 4. What is the variance of this distribution?",
         "ground_truth": "4", "depth": "Delta-0", "budget": 15, "baseline": 140},
        {"prompt": "In how many ways can 5 distinct books be arranged on a shelf such that 2 specific books are always adjacent?",
         "ground_truth": "48", "depth": "Delta-1", "budget": 30, "baseline": 190},
        {"prompt": "A committee of 3 people is chosen from 5 men and 4 women. What is the probability that the committee consists of 2 men and 1 woman (express as fraction)?",
         "ground_truth": "10/21", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "Calculate the sample standard deviation of data set: [3, 7, 7, 19] (round to two decimal places).",
         "ground_truth": "6.93", "depth": "Delta-2", "budget": 45, "baseline": 240},
        {"prompt": "If events A and B are independent with P(A) = 0.4 and P(B) = 0.5, what is P(A union B)?",
         "ground_truth": "0.7", "depth": "Delta-0", "budget": 25, "baseline": 170},

        # Financial Math & Compounding
        {"prompt": "An investment of $10,000 earns 6% annual interest compounded quarterly. What is the total accumulated value after 3 years in dollars (round to two decimal places)?",
         "ground_truth": "11956.18", "depth": "Delta-2", "budget": 50, "baseline": 250},
        {"prompt": "What is the effective annual rate (EAR) as a percentage for an investment with nominal rate 8% compounded monthly (round to two decimal places)?",
         "ground_truth": "8.30", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "How many years will it take for an investment to double at 7.2% annual interest compounded annually according to the Rule of 72?",
         "ground_truth": "10", "depth": "Delta-0", "budget": 15, "baseline": 140},
        {"prompt": "Find the present value in dollars of a cash flow of $5,000 received 4 years from now at a discount rate of 5% compounded annually (round to two decimal places).",
         "ground_truth": "4113.51", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "A bond with face value $1,000 pays a 5% annual coupon and has 1 year to maturity. If market yield is 6%, what is the bond price in dollars (round to two decimal places)?",
         "ground_truth": "990.57", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "What is the Sharpe ratio of a portfolio with expected return 14%, risk-free rate 4%, and portfolio standard deviation 20%?",
         "ground_truth": "0.5", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "If a stock has beta = 1.2, risk-free rate = 3%, and market risk premium = 5%, calculate expected return via CAPM as a percentage.",
         "ground_truth": "9.0", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "A perpetual annuity pays $600 at the end of each year. At an interest rate of 4%, what is its present value in dollars?",
         "ground_truth": "15000", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "What is the future value of an ordinary annuity of $1,000 paid annually for 3 years at 10% interest rate in dollars?",
         "ground_truth": "3310", "depth": "Delta-1", "budget": 35, "baseline": 200},
        {"prompt": "An asset is bought for $50,000 with residual value $5,000 after 5 years. Calculate annual straight-line depreciation expense in dollars.",
         "ground_truth": "9000", "depth": "Delta-0", "budget": 20, "baseline": 150},
        {"prompt": "If correlation coefficient between two assets is r = -0.5, what is the covariance if standard deviations are 10% and 20% (in decimal)?",
         "ground_truth": "-0.01", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "What is the median of the data set: [12, 5, 22, 30, 7, 9, 18]?",
         "ground_truth": "12", "depth": "Delta-0", "budget": 20, "baseline": 150},
        {"prompt": "In a standard normal distribution Z, what is P(Z < 0)?",
         "ground_truth": "0.5", "depth": "Delta-0", "budget": 15, "baseline": 130},
        {"prompt": "How many degrees of freedom are in a chi-square test of independence for a 3x4 contingency table?",
         "ground_truth": "6", "depth": "Delta-0", "budget": 15, "baseline": 140},
        {"prompt": "If the probability of failure on any trial is 0.2, what is the expected number of trials until the first success in a geometric distribution?",
         "ground_truth": "1.25", "depth": "Delta-0", "budget": 20, "baseline": 160},
    ]
    for q in prob_questions:
        q["domain"] = "Probability, Stats & Finance"
    items.extend(prob_questions)

    # =========================================================================
    # 5. Pure & Competition Mathematics (40 questions)
    # =========================================================================
    math_questions = [
        {"prompt": "If x + 1/x = 6, find the exact value of x^4 + 1/x^4.",
         "ground_truth": "1154", "depth": "Delta-1", "budget": 35, "baseline": 220},
        {"prompt": "Find the highest power of 5 that divides 250!.",
         "ground_truth": "62", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "Find the remainder when 3^100 is divided by 7.",
         "ground_truth": "4", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "A right triangle has inradius r = 4 and hypotenuse c = 25. Find the perimeter of the triangle.",
         "ground_truth": "58", "depth": "Delta-2", "budget": 45, "baseline": 250},
        {"prompt": "Find the number of positive integer solutions (x, y) to 1/x + 1/y = 1/6.",
         "ground_truth": "9", "depth": "Delta-2", "budget": 45, "baseline": 260},
        {"prompt": "Find the smallest positive integer n such that n = 2 mod 3, n = 3 mod 5, and n = 2 mod 7.",
         "ground_truth": "23", "depth": "Delta-2", "budget": 50, "baseline": 270},
        {"prompt": "How many derangements are there of a set of 5 elements (D_5)?",
         "ground_truth": "44", "depth": "Delta-2", "budget": 45, "baseline": 240},
        {"prompt": "If a and b are roots of x^2 - 7x + 3 = 0, find the value of a^3 + b^3.",
         "ground_truth": "280", "depth": "Delta-1", "budget": 35, "baseline": 220},
        {"prompt": "Find the sum of all distinct prime factors of 2026.",
         "ground_truth": "1015", "depth": "Delta-1", "budget": 30, "baseline": 190},
        {"prompt": "What is the remainder when 2^2024 is divided by 13?",
         "ground_truth": "3", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "Find the inradius of a triangle with side lengths 13, 14, 15.",
         "ground_truth": "4", "depth": "Delta-2", "budget": 45, "baseline": 250},
        {"prompt": "How many trailing zeros does the number 1000! have in base 10?",
         "ground_truth": "249", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "Find the smallest positive integer x such that 17x = 1 mod 43.",
         "ground_truth": "38", "depth": "Delta-2", "budget": 50, "baseline": 260},
        {"prompt": "If x, y are real numbers satisfying x^2 + y^2 = 25, find the maximum possible value of 3x + 4y.",
         "ground_truth": "25", "depth": "Delta-1", "budget": 30, "baseline": 190},
        {"prompt": "Find the 5th Catalan number C_5 = (1/6) * (10 choose 5).",
         "ground_truth": "42", "depth": "Delta-1", "budget": 30, "baseline": 190},
        {"prompt": "What is the value of Euler's totient function phi(360)?",
         "ground_truth": "96", "depth": "Delta-1", "budget": 35, "baseline": 200},
        {"prompt": "Find the area of a cyclic quadrilateral with side lengths 2, 3, 4, 5 (use Brahmagupta's formula; round to two decimal places).",
         "ground_truth": "10.95", "depth": "Delta-2", "budget": 45, "baseline": 250},
        {"prompt": "Solve for integer x: 5x = 9 mod 11. Give the smallest non-negative integer solution.",
         "ground_truth": "4", "depth": "Delta-1", "budget": 30, "baseline": 180},
        {"prompt": "If polynomial P(x) = x^3 - 6x^2 + 11x - 6, find the sum of squares of its roots.",
         "ground_truth": "14", "depth": "Delta-1", "budget": 35, "baseline": 200},
        {"prompt": "Find the number of positive divisors of 7200.",
         "ground_truth": "54", "depth": "Delta-1", "budget": 30, "baseline": 190},
        {"prompt": "Find the highest power of 2 that divides 100!.",
         "ground_truth": "97", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "If x - 1/x = 4, find x^2 + 1/x^2.",
         "ground_truth": "18", "depth": "Delta-1", "budget": 25, "baseline": 170},
        {"prompt": "Find the remainder when 7^77 is divided by 10.",
         "ground_truth": "3", "depth": "Delta-1", "budget": 30, "baseline": 180},
        {"prompt": "What is the sum of integers from 1 to 100 that are divisible by neither 2 nor 5?",
         "ground_truth": "2000", "depth": "Delta-2", "budget": 50, "baseline": 270},
        {"prompt": "Find the smallest positive integer n such that 2^n > n^10.",
         "ground_truth": "60", "depth": "Delta-3", "budget": 60, "baseline": 320},
        {"prompt": "Compute the sum of the infinite geometric series: 12 + 4 + 4/3 + 4/9 + ...",
         "ground_truth": "18", "depth": "Delta-0", "budget": 20, "baseline": 160},
        {"prompt": "What is the coefficient of x^3 in the expansion of (2x - 3)^5?",
         "ground_truth": "720", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "Find the sum of all angles in a convex nonagon (9-sided polygon) in degrees.",
         "ground_truth": "1260", "depth": "Delta-0", "budget": 20, "baseline": 150},
        {"prompt": "If log_2(x) + log_2(x - 2) = 3, find the positive real solution x.",
         "ground_truth": "4", "depth": "Delta-1", "budget": 30, "baseline": 190},
        {"prompt": "How many integers between 1 and 500 are divisible by 6 or 8?",
         "ground_truth": "125", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "Find the determinant of the 2x2 matrix: [[7, 4], [3, 2]].",
         "ground_truth": "2", "depth": "Delta-0", "budget": 15, "baseline": 140},
        {"prompt": "What is the trace of the 3x3 identity matrix?",
         "ground_truth": "3", "depth": "Delta-0", "budget": 10, "baseline": 120},
        {"prompt": "Find the length of the diagonal of a rectangular cuboid with dimensions 3, 4, 12.",
         "ground_truth": "13", "depth": "Delta-1", "budget": 25, "baseline": 170},
        {"prompt": "Find the unit digit of 17^2025.",
         "ground_truth": "7", "depth": "Delta-1", "budget": 25, "baseline": 170},
        {"prompt": "Evaluate: 10! / (7! * 3!).",
         "ground_truth": "120", "depth": "Delta-0", "budget": 20, "baseline": 150},
        {"prompt": "Find the arithmetic mean of all prime numbers between 10 and 30.",
         "ground_truth": "18.67", "depth": "Delta-1", "budget": 35, "baseline": 210},
        {"prompt": "Find the area of an equilateral triangle with side length 8 (round to two decimal places).",
         "ground_truth": "27.71", "depth": "Delta-1", "budget": 30, "baseline": 190},
        {"prompt": "Find the value of k such that x - 2 divides x^3 - 3x^2 + kx - 8 without remainder.",
         "ground_truth": "6", "depth": "Delta-1", "budget": 30, "baseline": 190},
        {"prompt": "If sin(theta) = 3/5 with theta in quadrant I, find cos(2*theta) as a fraction.",
         "ground_truth": "7/25", "depth": "Delta-1", "budget": 30, "baseline": 180},
        {"prompt": "What is the maximum number of regions into which 5 lines can divide the Euclidean plane?",
         "ground_truth": "16", "depth": "Delta-1", "budget": 25, "baseline": 170},
    ]
    for q in math_questions:
        q["domain"] = "Pure & Competition Mathematics"
    items.extend(math_questions)

    # =========================================================================
    # 6. Scientific & Multi-Hop Factual Retrieval (25 questions)
    # =========================================================================
    factual_questions = [
        {"prompt": "What is the atomic number of Iron?", "ground_truth": "26", "depth": "Delta-0", "budget": 10, "baseline": 120},
        {"prompt": "What is the chemical formula for Ozone?", "ground_truth": "O3", "depth": "Delta-0", "budget": 10, "baseline": 120},
        {"prompt": "Who formulated the three laws of planetary motion?", "ground_truth": "Johannes Kepler", "depth": "Delta-0", "budget": 15, "baseline": 140},
        {"prompt": "What is the SI unit of electric capacitance?", "ground_truth": "Farad", "depth": "Delta-0", "budget": 10, "baseline": 120},
        {"prompt": "What planet in our solar system has the highest surface temperature?", "ground_truth": "Venus", "depth": "Delta-0", "budget": 10, "baseline": 130},
        {"prompt": "Which amino acid is encoded by the mRNA start codon AUG?", "ground_truth": "Methionine", "depth": "Delta-0", "budget": 10, "baseline": 130},
        {"prompt": "What is the chemical symbol for Tungsten?", "ground_truth": "W", "depth": "Delta-0", "budget": 10, "baseline": 120},
        {"prompt": "What is the approximate speed of sound in air at 20 deg C in m/s (to nearest integer)?", "ground_truth": "343", "depth": "Delta-0", "budget": 10, "baseline": 130},
        {"prompt": "What organelle is known as the site of cellular respiration and ATP synthesis?", "ground_truth": "Mitochondria", "depth": "Delta-0", "budget": 10, "baseline": 130},
        {"prompt": "Which subatomic particle has zero electric charge and a mass slightly greater than a proton?", "ground_truth": "Neutron", "depth": "Delta-0", "budget": 10, "baseline": 130},
        {"prompt": "What is the primary greenhouse gas released by human agricultural livestock?", "ground_truth": "Methane", "depth": "Delta-0", "budget": 10, "baseline": 130},
        {"prompt": "What is the name of the nearest major spiral galaxy to the Milky Way?", "ground_truth": "Andromeda", "depth": "Delta-0", "budget": 10, "baseline": 130},
        {"prompt": "In computer architecture, what does RISC stand for?", "ground_truth": "Reduced Instruction Set Computer", "depth": "Delta-0", "budget": 15, "baseline": 140},
        {"prompt": "Which element has the highest electronegativity on the Pauling scale?", "ground_truth": "Fluorine", "depth": "Delta-0", "budget": 10, "baseline": 120},
        {"prompt": "What is the approximate value of Avogadro's number (e.g. 6.022e23)?", "ground_truth": "6.022e23", "depth": "Delta-0", "budget": 10, "baseline": 130},
        {"prompt": "What type of chemical bond involves the sharing of electron pairs between atoms?", "ground_truth": "Covalent", "depth": "Delta-0", "budget": 10, "baseline": 120},
        {"prompt": "Who proved Fermat's Last Theorem in 1994?", "ground_truth": "Andrew Wiles", "depth": "Delta-0", "budget": 15, "baseline": 140},
        {"prompt": "What is the hardest known natural mineral on the Mohs hardness scale?", "ground_truth": "Diamond", "depth": "Delta-0", "budget": 10, "baseline": 120},
        {"prompt": "Which fundamental physical force is mediated by the gluon?", "ground_truth": "Strong force", "depth": "Delta-0", "budget": 10, "baseline": 130},
        {"prompt": "What is the standard unit of magnetic flux density?", "ground_truth": "Tesla", "depth": "Delta-0", "budget": 10, "baseline": 120},
        {"prompt": "Which vitamin is synthesized in human skin upon exposure to ultraviolet B radiation?", "ground_truth": "Vitamin D", "depth": "Delta-0", "budget": 10, "baseline": 130},
        {"prompt": "What is the boiling point of liquid Nitrogen at 1 atm in Celsius (nearest integer)?", "ground_truth": "-196", "depth": "Delta-0", "budget": 10, "baseline": 130},
        {"prompt": "In networking, what does the acronym DNS stand for?", "ground_truth": "Domain Name System", "depth": "Delta-0", "budget": 15, "baseline": 140},
        {"prompt": "What is the principal quantum number n of the outermost valence electrons in neutral Sodium?", "ground_truth": "3", "depth": "Delta-0", "budget": 10, "baseline": 130},
        {"prompt": "Which mountain peak has the highest elevation above sea level on Earth?", "ground_truth": "Mount Everest", "depth": "Delta-0", "budget": 10, "baseline": 120},
    ]
    for q in factual_questions:
        q["domain"] = "Scientific & Multi-Hop Factual Retrieval"
    items.extend(factual_questions)

    # =========================================================================
    # 7. Strategic Intent & Pragmatics (25 questions)
    # =========================================================================
    dialogue_questions = [
        {"prompt": "A client is furious because their cloud service went down for 45 minutes during peak hours. How do you respond to acknowledge fault without admitting legal liability while retaining their business?",
         "ground_truth": "acknowledge fault without legal liability", "depth": "Delta-1", "budget": 75, "baseline": 220},
        {"prompt": "How do you tell a senior colleague that their proposed architectural refactor has a critical race condition without making them defensive?",
         "ground_truth": "respectful technical correction", "depth": "Delta-1", "budget": 75, "baseline": 210},
        {"prompt": "An executive demands that the security team bypass MFA compliance for their account to save time. How do you decline firmly while preserving professional rapport?",
         "ground_truth": "firm policy boundary with polite rapport", "depth": "Delta-1", "budget": 70, "baseline": 200},
        {"prompt": "A team member is dominating meetings and interrupting junior engineers. How do you address this in your 1-on-1 constructively?",
         "ground_truth": "constructive 1-on-1 feedback", "depth": "Delta-1", "budget": 75, "baseline": 220},
        {"prompt": "During a salary negotiation, the recruiter makes an initial offer 15% below your target. How do you counter-offer professionally without being confrontational?",
         "ground_truth": "anchored professional counter-offer", "depth": "Delta-1", "budget": 75, "baseline": 210},
        {"prompt": "A cross-functional department has missed two consecutive data handoffs, blocking your team's launch. How do you email their director to escalate constructively?",
         "ground_truth": "constructive executive escalation", "depth": "Delta-1", "budget": 80, "baseline": 230},
        {"prompt": "How do you explain to a non-technical board member why technical debt must be paid down before adding new features?",
         "ground_truth": "business metaphor for technical debt", "depth": "Delta-1", "budget": 80, "baseline": 240},
        {"prompt": "A customer requests a full refund past the 30-day return window citing personal emergency. How do you craft an empathetic response offering store credit as an exception?",
         "ground_truth": "empathetic policy exception with store credit", "depth": "Delta-1", "budget": 75, "baseline": 220},
        {"prompt": "You discovered that your manager made an error in the quarterly budget presentation scheduled in 30 minutes. How do you alert them discreetly?",
         "ground_truth": "discreet urgent error notification", "depth": "Delta-1", "budget": 70, "baseline": 200},
        {"prompt": "An open-source contributor submitted a large PR that fundamentally misaligns with the project's roadmap. How do you decline the PR with gratitude for their time?",
         "ground_truth": "grateful roadmap-aligned PR decline", "depth": "Delta-1", "budget": 75, "baseline": 220},
        {"prompt": "How do you deliver news of an internal project cancellation to the engineering team that spent 6 months building it?",
         "ground_truth": "transparent empathetic project pivot", "depth": "Delta-1", "budget": 85, "baseline": 250},
        {"prompt": "A stakeholder asks for an unrealistic 2-week deadline for a 2-month feature. How do you negotiate a phased MVP scope without saying a flat 'no'?",
         "ground_truth": "phased MVP scope negotiation", "depth": "Delta-1", "budget": 80, "baseline": 230},
        {"prompt": "How do you ask a customer for an honest testimonial without making them feel pressured or obligated?",
         "ground_truth": "low-pressure testimonial request", "depth": "Delta-1", "budget": 65, "baseline": 190},
        {"prompt": "Two peers on your team have an ongoing personal conflict disrupting sprint reviews. How do you mediate as a tech lead?",
         "ground_truth": "impartial work-focused mediation", "depth": "Delta-1", "budget": 80, "baseline": 230},
        {"prompt": "How do you decline an invitation to join an advisory board due to bandwidth constraints while keeping future relationship doors open?",
         "ground_truth": "warm gracious capacity decline", "depth": "Delta-1", "budget": 65, "baseline": 190},
        {"prompt": "A client is micromanaging daily pull requests and messaging developers directly. How do you establish communication boundaries diplomatically?",
         "ground_truth": "diplomatic channel boundary setting", "depth": "Delta-1", "budget": 75, "baseline": 220},
        {"prompt": "How do you announce a critical security patch deployment requiring a 10-minute scheduled maintenance window tonight to users?",
         "ground_truth": "clear transparent maintenance alert", "depth": "Delta-1", "budget": 70, "baseline": 200},
        {"prompt": "A direct report expresses severe burnout and exhaustion in a 1-on-1. How do you respond with immediate psychological safety and workload triage?",
         "ground_truth": "immediate workload triage and psychological safety", "depth": "Delta-1", "budget": 80, "baseline": 240},
        {"prompt": "How do you write a LinkedIn connection message to a prominent AI researcher whose work inspired your project without sounding like spam?",
         "ground_truth": "specific respectful intellectual outreach", "depth": "Delta-1", "budget": 70, "baseline": 200},
        {"prompt": "A partner company accidentally shared confidential partner metrics in a public Slack channel. How do you handle containment and notification?",
         "ground_truth": "calm urgent containment notification", "depth": "Delta-1", "budget": 75, "baseline": 220},
        {"prompt": "How do you request an urgent letter of recommendation from a busy professor with only 5 days before the fellowship deadline?",
         "ground_truth": "pre-drafted respectful urgent recommendation request", "depth": "Delta-1", "budget": 75, "baseline": 210},
        {"prompt": "An enterprise prospect asks for a custom feature your roadmap does not support. How do you steer them toward existing APIs as a superior solution?",
         "ground_truth": "consultative API solution steering", "depth": "Delta-1", "budget": 80, "baseline": 230},
        {"prompt": "How do you address a client rumor that your startup is struggling financially with transparency and confidence?",
         "ground_truth": "fact-based confident transparency", "depth": "Delta-1", "budget": 80, "baseline": 240},
        {"prompt": "A team member consistently takes credit for collaborative team work during all-hands demos. How do you address this with them directly?",
         "ground_truth": "direct constructive credit attribution conversation", "depth": "Delta-1", "budget": 75, "baseline": 220},
        {"prompt": "How do you announce a necessary 20% price increase to existing legacy subscribers while maximizing retention?",
         "ground_truth": "value-justified grandfathered pricing announcement", "depth": "Delta-1", "budget": 85, "baseline": 250},
    ]
    for q in dialogue_questions:
        q["domain"] = "Strategic Intent & Pragmatics"
    items.extend(dialogue_questions)

    # Randomize order while preserving deterministic seed
    random.seed(42)
    random.shuffle(items)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for idx, item in enumerate(items, start=1):
            item["eval_id"] = f"CROSS-200-{idx:03d}"
            item["category"] = item["domain"]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Generated {len(items)} cross-domain benchmark questions -> {OUTPUT_FILE}")
    print("\nDomain Distribution:")
    counts = {}
    depths = {}
    for item in items:
        counts[item["domain"]] = counts.get(item["domain"], 0) + 1
        depths[item["depth"]] = depths.get(item["depth"], 0) + 1

    for domain, cnt in sorted(counts.items()):
        print(f"  - {domain:45s}: {cnt:2d} questions")

    print("\nCognitive Depth Distribution (Gradient Register Spectrum):")
    for d, cnt in sorted(depths.items()):
        print(f"  - {d:25s}: {cnt:2d} questions")


if __name__ == "__main__":
    generate_benchmark()
