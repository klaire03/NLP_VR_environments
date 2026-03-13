# VR Workrooms — Affective NLP Analysis
**Meta Reality Labs · UX Research · Affective NLP Pipeline**

A multi-source affective NLP pipeline that takes raw user feedback from three channels and maps how users emotionally experience six proposed VR meeting environments — delivering structured, emotion-linked design recommendations for the 3D arts team.

---

## Overview

VR meeting spaces are not neutral containers. Their materials, lighting, and spatial scale actively shape how users feel and perform. This project asks: **which environments feel right, for which meeting occasions, and which specific design elements are driving those emotional responses?**

| Source | File | N | Participant type | Weight |
|---|---|---|---|---|
| Surveys | `survey_responses.csv` | 108 | Customer | 0.40 |
| Interviews | `interview_segments.csv` | 20 | Customer | 0.45 |
| Sticky notes | `sticky_notes.csv` | 60 | Internal colleague | 0.15 |

> Source weights reflect research quality judgment, not volume. Interview segments are richest in emotional language and carry the highest weight. Sticky notes are very short and informal — lowest signal per document.

---

## The Six Environments

| Environment | Emotional register | Primary NRC emotion |
|---|---|---|
| Executive Boardroom | Dark wood, leather, authoritative proportions | Trust |
| Nature Space | Natural light, greenery, open scale | Joy |
| Clean Studio | Minimal, neutral, distraction-free | Trust |
| Residential Warm | Soft lighting, warm tones, domestic materials | Joy |
| Futuristic Tech | Cool blues, geometric precision, high-tech | Joy |
| City Loft | Industrial materials, urban scale, brick and metal | Anticipation |

---

## NLP Pipeline

Eight techniques run in sequence, each answering a specific analytical question.

```
 1.  Install & import          — subprocess pip; spaCy download; VADER, transformers, sklearn, wordcloud
 2.  Load CSVs                 — survey_responses.csv (108) · interview_segments.csv (20) · sticky_notes.csv (60)
 3.  Source-aware cleaning     — surveys: whitespace · interviews: filler removal · stickies: lowercase + abbrev + punct
 4.  Build unified corpus      — standardised columns; source weights 0.40 / 0.45 / 0.15; participant_type flag
 5.  VADER sentiment           — 17-term VR domain extension; compound score; label; weighted mean per environment
 6.  Visualise sentiment       — weighted bar chart; source channel grouped bar; violin distribution
 7.  NRC emotion scoring       — self-contained lexicon dict (~160 words); 8-emotion proportion scores; word clouds
 8.  Occasion mapping          — zero-shot NLI (bart-large-mnli fallback chain); 8 occasions; multi-label threshold 0.15
 9.  Occasion heatmap + flow   — normalised occasion × environment heatmap; stacked bar flow chart
10.  Design attribute extract  — spaCy en_core_web_sm; material/colour/lighting/spatial vocab; NRC adjective co-occurrence
11.  Topic modelling           — TF-IDF 300 features ngram 1–2; NMF 8 components; topic heatmap
12.  Emotion profiles          — weighted NRC per environment; overlaid radar + individual radars + heatmap
13.  Tech vs non-tech split    — customer corpus only; industry_class filter; Futuristic Tech gap quantification
14.  Natural light signal      — keyword search; sentiment delta light vs non-light docs; cross-source agreement check
15.  Triangulation & confidence— source pivot; agreement() function; confidence = f(variance, doc_count)
16.  Synthesis table           — design attr + primary emotion + valence + top occasion + confidence → synthesis_table.csv
```

---

## Techniques & Design Decisions

### VADER + VR domain lexicon extension
VADER handles informal text and short phrases well — a good fit for interview transcripts and sticky notes. Extended with 17 domain-specific terms before any scoring:

```python
analyser.lexicon.update({
    'immersive': 2.5,  'grounded': 1.8,   'presence': 2.0,
    'restorative': 2.2, 'purposeful': 2.0, 'cinematic': 2.0,
    'claustrophobic': -2.5, 'alienating': -2.8, 'disconnected': -3.0,
    'oppressive': -2.5, 'sterile': -1.5,  'floating': -1.5,
    # ...
})
```

Without this extension, high-signal VR vocabulary words like *floating*, *presence*, and *grounding* score neutrally or are unrecognised entirely.

### NRC Emotion Lexicon (self-contained)
Maps words to 8 emotion dimensions: joy, trust, fear, surprise, sadness, disgust, anger, anticipation. Embedded as a Python dict (~160 words curated for VR spatial vocabulary) rather than using the `nrclex` library — more robust, no download required, and tuned to the specific vocabulary of this dataset.

> ⚠ No negation handling — "no joy" scores positive on joy. Treat scores as comparative across environments, not absolute measurements.

### Zero-Shot NLI Occasion Mapping
Classifies each survey's occasion text into 8 meeting types with no labelled examples. Framed as: *does this text entail the space is suitable for [occasion]?* Uses a cascading model fallback — first available model wins:

1. `facebook/bart-large-mnli`
2. `typeform/distilbert-base-uncased-mnli`
3. `cross-encoder/nli-deberta-v3-small`
4. `valhalla/distilbart-mnli-12-3`

A rule-based fallback keeps all downstream cells running if no model loads — making the notebook fully portable without internet access or HuggingFace credentials.

### spaCy Design Attribute Extraction
`en_core_web_sm` parses each sentence. Tokens matching a curated design vocabulary (materials, colours, lighting, spatial elements) are extracted, and adjectives in the same sentence that appear in the NRC lexicon are linked as emotion words. Produces a `(design element → emotion → environment)` triple — the core of the final synthesis deliverable.

### TF-IDF + NMF Topic Modelling
NMF chosen over LDA because NMF produces sparser, more interpretable topics on short-document corpora. LDA's Dirichlet prior spreads signal too evenly when documents are only 1–3 sentences long.

```python
vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1,2),
                             stop_words='english', min_df=2)
nmf = NMF(n_components=8, random_state=42, max_iter=400)
```

---

## Key Findings

**1. Occasion fit > absolute sentiment**
The Executive Boardroom scores negatively in weighted aggregate sentiment, yet dominates `client_pitch` (0.41) and `performance_review` (0.30) occasion categories. The right question is not *do users like it?* but *does it match what they need to do?*

**2. Natural light is the highest-leverage design lever**
The only design attribute with full cross-source agreement — surveys, interviews, and sticky notes all flag it positively, across every environment where it appears. Every other finding has at least one source exception.

**3. The tech vs non-tech gap is invisible in aggregate**
Technology-sector customers respond significantly more positively to Futuristic Tech than non-tech customers. The aggregate score sits near neutral — masking a substantial split that only appears with an `industry_class` filter.

**4. Residential Warm: wellbeing yes, productivity no**
Highest comfort and joy NRC scores. Lowest focus and professionalism ratings. Dominates `team_social` and `onboarding` occasions, near-zero on output-driven sessions. Should be positioned explicitly as a pastoral and social environment.

**5. Confidence scores are themselves a finding**
The synthesis table assigns a confidence score per environment row combining cross-source variance (60%) and document count (40%). It is a prioritised action list — confidence tells the 3D arts team how much weight to give each row.

---

## Visualisations

| Figure | Description |
|---|---|
| Fig 1 | Weighted mean sentiment per environment |
| Fig 2 | Sentiment by environment and source channel |
| Fig 3 | Sentiment distribution — violin plots |
| Fig 4 | Vocabulary fingerprints — word clouds per environment |
| Fig 5 | Occasion fit heatmap (normalised per environment) |
| Fig 6 | Meeting occasion → environment flow (stacked bar) |
| Fig 7 | Topic distribution per environment (NMF, normalised) |
| Fig 8 | Emotion intensity heatmap (NRC 8 dimensions) |
| Fig 9 | NRC emotion profiles — overlaid radar chart |
| Fig 10 | Final synthesis table for 3D arts team |

---

## Final Deliverable

`synthesis_table.csv` — links design attributes, primary emotions, valence, top occasion, and confidence score in a single structured table for the 3D arts team.

| Column | Description |
|---|---|
| `Environment` | One of the six VR environments |
| `Design Attribute` | Extracted design element (e.g. *light*, *dark*, *warm*) |
| `Category` | lighting / colour / material / spatial |
| `Primary Emotion` | Dominant NRC emotion for that environment |
| `Valence` | Strongly positive / Positive / Negative / Mixed |
| `Top Occasion` | Highest-scoring meeting occasion |
| `Attribute Mentions` | Raw mention count across all sources |
| `Confidence` | Cross-source reliability score (0–1) |

---

## Tech Stack

```
Python 3          pandas            numpy
vaderSentiment    spaCy             scikit-learn
transformers      torch             wordcloud
matplotlib        seaborn           Google Colab
```

---

## Caveats

- **Synthetic data** — the three CSV files are synthetic, generated to match the structure and vocabulary distribution of real VR Workrooms feedback. The pipeline architecture generalises directly to real data.
- **Short text limitations** — survey responses average 2–3 sentences. NRC emotion scoring is noisier on very short text. Confidence scores partially compensate.
- **Source imbalance** — 108 surveys vs 20 interviews vs 60 sticky notes. Source weights compensate for quality differences but not coverage gaps.
- **No negation awareness** in NRC scoring. Use emotion scores comparatively across environments.

---

## Portfolio

Full write-up with visualisations: [vr_workrooms_portfolio.html](vr_workrooms_portfolio.html)
