# EasyChair Submission — QCE26 Poster Proposals & 2-page Papers (Phase 2)

Copy-paste these into the EasyChair "New Submission for QCE26" form. Plain text only (no HTML/LaTeX).

---

## Author 1
- **First name:** Mohammad
- **Last name:** Zoraiz
- **Email:** mz248@duke.edu
- **Country/region:** United States
- **Affiliation:** Duke University, Pratt School of Engineering
- **corresponding author:** ✓ (check)
- **presenter:** ✓ (check)

> Note: If your name splits differently (e.g., "Zoraiz" is not your family name), adjust First/Last accordingly. ORCID is optional in the form but recommended on the paper — add yours if you have one.

---

## Title
Hardware-Aware Quantum Bayesian Learner Compression: Linking Human Belief Updating to NISQ Circuit Complexity

---

## Abstract  (237 words — within the 200–250 requirement)
This work connects quantum models of cognition with the engineering constraints of near-term quantum hardware. Bayesian theories treat human learning as belief updating over structured hypotheses, and quantum cognition represents belief states in Hilbert space with evidence acting through quantum channels. We instantiate a small quantum Bayesian learner: a belief state on a two-qubit register, a prior, and a stimulus-dependent evidence channel that updates the belief on a synthetic similarity-structured categorization task inspired by the simplicity principle of Pothos and Chater, with the posterior category read out by measurement. To tie this cognitive picture to real devices, we make the learner a hardware-efficient variational circuit and add a hardware-aware penalty equal to the number of two-qubit gates that survive transpilation to IBM's FakeManila coupling map, then compress the circuit by greedy structured pruning of individual Heisenberg interaction terms. The central question is where the capacity boundary lies: how much entangling structure can be removed before the learner can no longer represent the task. Across three difficulty levels and five seeds we trace accuracy-complexity frontiers and find that structured compression is essentially free, and on the hardest task beneficial, while removing all entanglers collapses the learner to chance. Learned masks beat random masks at every matched budget. Finally, on a physical IBM device the full circuit collapses to near chance while the compressed circuit retains accuracy: on real NISQ hardware, hardware-aware compression is not merely economical but necessary.

---

## Keywords  (one per line; >= 3 required)
Quantum cognition
Bayesian learning
Variational quantum circuits
Hardware-efficient ansatz
NISQ devices
Circuit compression
Quantum machine learning
Quantum channels

---

## Other Information and Files
- **Poster Proposal Phase 1 or Phase 2:** select **POS2: Poster Proposal Phase 2 (4 pages max)**
- **IEEE Conference Authorship & AI Policies:** check **"We acknowledge that our contribution conforms to the IEEE Conference & AI Policies"** (review the linked author-policy PDF and AI-policy video first).
- **2-page Poster Proposal (file upload):** `QBL_PosterProposal_Phase2.pdf`  (page 1 = Sections 1–4, page 2 = the poster)
- **2-page Extended Poster Abstract Paper (file upload):** `QBL_ExtendedAbstract.pdf`

> Reminder: submitting commits you to register and attend in person (Toronto, Sep 2026) if accepted.
