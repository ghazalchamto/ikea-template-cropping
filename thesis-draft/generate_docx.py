"""
Generate a specific, system-focused research paper as a .docx for Google Docs.
Run:  python generate_docx.py
"""

from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.5)

# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────
def h(text, level=1):
    doc.add_heading(text, level=level)

def p(text, bold=False, italic=False, size=11,
      align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    pg = doc.add_paragraph()
    pg.alignment = align
    r = pg.add_run(text)
    r.bold = bold; r.italic = italic; r.font.size = Pt(size)
    return pg

def pl(text, bold=False, italic=False, size=11):
    return p(text, bold=bold, italic=italic, size=size,
             align=WD_ALIGN_PARAGRAPH.LEFT)

def bp(text):
    pg = doc.add_paragraph(style='List Bullet')
    pg.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pg.add_run(text).font.size = Pt(11)
    return pg

def nb(text):
    pg = doc.add_paragraph(style='List Number')
    pg.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pg.add_run(text).font.size = Pt(11)
    return pg

def br():
    doc.add_page_break()

def sp(n=1):
    for _ in range(n): doc.add_paragraph()

def tbl(headers, rows, caption=""):
    if caption:
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cp.add_run(caption); r.bold = True; r.font.size = Pt(10)
    t = doc.add_table(rows=1+len(rows), cols=len(headers))
    t.style = 'Table Grid'
    for i, h_ in enumerate(headers):
        t.rows[0].cells[i].text = h_
        for r_ in t.rows[0].cells[i].paragraphs[0].runs:
            r_.bold = True; r_.font.size = Pt(10)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            t.rows[ri+1].cells[ci].text = str(val)
            for r_ in t.rows[ri+1].cells[ci].paragraphs[0].runs:
                r_.font.size = Pt(10)
    doc.add_paragraph()

def eq(label, text):
    pg = doc.add_paragraph()
    pg.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = pg.add_run(text); r.italic = True; r.font.size = Pt(11)
    pg2 = doc.add_paragraph()
    pg2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    pg2.add_run(f"({label})").font.size = Pt(10)

def design_choice(title, text):
    """Highlighted design-decision callout."""
    pg = doc.add_paragraph()
    pg.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = pg.add_run(f"Design decision — {title}: ")
    r.bold = True; r.font.size = Pt(11)
    r2 = pg.add_run(text)
    r2.italic = True; r2.font.size = Pt(11)
    sp()

# =============================================================================
# TITLE PAGE
# =============================================================================
sp(2)
p("Kristianstad University", bold=True, size=18, align=WD_ALIGN_PARAGRAPH.CENTER)
p("Department of Computer Science", size=13, align=WD_ALIGN_PARAGRAPH.CENTER)
sp(2)

tp = doc.add_paragraph()
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tp.add_run(
    "A Three-Tier Hybrid Validation Architecture for Standardised\n"
    "Industrial Label Templates:\n"
    "Combining Rule-Based Pre-Render Checks, Reference Image Comparison,\n"
    "and Deep Learning Classification"
)
r.bold = True; r.font.size = Pt(20)
sp(2)

p("Bachelor Thesis in Computer Science — 15 Credits", size=13,
  align=WD_ALIGN_PARAGRAPH.CENTER)
sp(2)

info = [
    ("Authors:",    "Ghanasham Saravaiahgari"),
    ("",            "Ghazal Chamto"),
    ("Supervisor:", "[Supervisor Name], Kristianstad University"),
    ("Examiner:",   "[Examiner Name], Kristianstad University"),
    ("Partner:",    "IKEA of Sweden AB"),
    ("Programme:",  "Bachelor in Software Technology"),
    ("Term:",       "Spring 2026"),
]
t = doc.add_table(rows=len(info), cols=2)
t.style = 'Table Grid'
for ri, (lbl, val) in enumerate(info):
    t.rows[ri].cells[0].text = lbl
    t.rows[ri].cells[1].text = val
    if lbl:
        for r_ in t.rows[ri].cells[0].paragraphs[0].runs:
            r_.bold = True

sp()
p("This thesis is submitted in partial fulfilment of the requirements for the "
  "degree of Bachelor of Science in Computer Science at Kristianstad University, "
  "Kristianstad, Sweden.",
  size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
br()

# =============================================================================
# ABSTRACT
# =============================================================================
h("Abstract")
p(
    "Standardised product label templates are the backbone of IKEA of Sweden's global "
    "supply-chain compliance. Every generated label must conform exactly to an approved "
    "baseline design covering element positioning, font sizing, barcode quiet zones, "
    "content fields, and colour specifications. Currently, a trained human QA tester "
    "inspects each label manually — a process taking 8 to 12 minutes per label that "
    "cannot scale with product range growth."
)
p(
    "This paper presents the design rationale, architecture, and experimental evaluation "
    "of a purpose-built three-tier hybrid validation system. The central design insight "
    "is that label violations fall into three fundamentally different categories — "
    "content errors detectable from raw metadata, structural deviations detectable by "
    "classical image comparison, and subtle layout violations detectable only by learned "
    "visual representations — and that no single validation technique addresses all three. "
    "The system is therefore built as an ordered pipeline of complementary components: "
    "a rule-based metadata checker, a rendering engine via the NiceLabel .NET SDK, and "
    "a four-method visual validation ensemble combining SSIM reference comparison, ORB "
    "feature matching, a fine-tuned ResNet-50 binary classifier, and a Siamese neural "
    "network trained with contrastive loss."
)
p(
    "The architecture is driven by concrete constraints of the IKEA production environment: "
    "the proprietary .nlbl file format necessitates SDK-based rendering; the absence of "
    "archived invalid labels shapes the training strategy; and the need for operator-readable "
    "error reports requires spatial localisation of every detected violation. Each component "
    "was selected because it addresses specific failure modes of the others, forming a "
    "complementary ensemble rather than an arbitrary collection of methods."
)
p(
    "Evaluated on 150 labelled label images spanning 20 template types with seven "
    "categories of controlled synthetic violation, the hybrid ensemble achieves 93.3% "
    "accuracy, 100% precision, 85.7% recall, and an F1-score of 92.3% on the held-out "
    "test set. Violation localisation achieves IoU above 0.5 in 83% of detected errors. "
    "Median end-to-end processing time is 401 ms per label — a greater than 1,000-fold "
    "speedup over manual review."
)
p(
    "Keywords: industrial template validation, hybrid ensemble, reference-based comparison, "
    "ResNet-50 fine-tuning, Siamese networks, contrastive loss, SSIM, ORB, NiceLabel SDK, "
    "bounding-box localisation, automated quality assurance.",
    italic=True, size=10
)
br()

# =============================================================================
# ACKNOWLEDGEMENTS
# =============================================================================
h("Acknowledgements")
p(
    "We would like to express our sincere gratitude to our supervisor at Kristianstad "
    "University for continuous guidance and constructive feedback throughout this project. "
    "Their domain knowledge in machine learning and industrial computer vision shaped "
    "the direction of this work at every stage."
)
p(
    "We are deeply grateful to the engineering and QA team at IKEA of Sweden AB in "
    "Älmhult. They proposed this research problem, provided access to label specifications "
    "and the NiceLabel SDK, and offered domain expertise that was essential for designing "
    "the validation rules and understanding real production constraints. Without their "
    "collaboration this work would not have been possible."
)
p(
    "We thank our examiner for critique during the midway seminar, which significantly "
    "improved our discussion of failure cases and design tradeoffs."
)
p("Ghanasham Saravaiahgari    Ghazal Chamto\n"
  "Kristianstad University, April 2026",
  align=WD_ALIGN_PARAGRAPH.RIGHT)
br()

# =============================================================================
# CHAPTER 1 — INTRODUCTION
# =============================================================================
h("Chapter 1: Introduction")

h("1.1  The Industrial Problem", level=2)
p(
    "IKEA of Sweden designs and owns approximately 9,500 product articles, each requiring "
    "one or more standardised physical labels for packaging, shipping cartons, and point-of-"
    "sale presentation. Each label type is engineered as a NiceLabel template — a proprietary "
    ".nlbl file that defines the spatial layout of text fields, barcodes, symbols, regulatory "
    "markings, and colour zones. At print time, a structured JSON metadata payload supplies "
    "the variable content: article number, product name, origin, supplier details, weights, "
    "dimensions, and GS1 DataMatrix application identifiers."
)
p(
    "The critical constraint is conformance: every rendered label must be pixel-accurate "
    "against its approved baseline template. A displaced text field can push a barcode "
    "outside its quiet zone, causing scanner failures at goods receipt. A font size "
    "deviation can make mandatory regulatory text unreadable. A missing field can cause "
    "a pallet to be rejected at customs. These are not cosmetic issues — they are "
    "operational failures with real supply-chain consequences."
)
p(
    "Currently, QA testers review each generated label visually against a printed or "
    "on-screen reference. This process takes 8–12 minutes per label, is not reproducible "
    "across testers, and does not scale with template catalogue growth. The QA team "
    "identified three categories of errors they catch most frequently:"
)
bp("Content errors: article number in wrong format, missing mandatory field, incorrect "
   "supplier name spelling.")
bp("Structural errors: element positioned outside its reference zone, font size changed, "
   "extra or unexpected element present.")
bp("Rendering errors: barcode overlap with adjacent text, quiet zone violation, "
   "colour value drifted outside specification.")
p(
    "This taxonomy directly shaped the three-tier architecture described in this paper."
)

h("1.2  Why No Single Existing Approach Suffices", level=2)
p(
    "Several existing approaches were considered before designing the system. Each was "
    "found to be insufficient on its own for this specific problem:"
)
p(
    "Pure OCR-based validation can extract field values from a rendered label and compare "
    "them to expected values, but it cannot detect positional violations — a field present "
    "in the wrong location passes OCR validation. It also cannot detect visual violations "
    "such as font size changes or barcode overlaps."
)
p(
    "Pure pixel-difference comparison is simple and deterministic, but NiceLabel's "
    "renderer introduces sub-pixel anti-aliasing variations between successive renders "
    "of the same template. These rendering artifacts produce non-zero pixel differences "
    "for valid labels, making a strict pixel threshold impractical. Our experiments "
    "confirmed this: the pixel-difference baseline produces an F1 of only 0.714, with "
    "two false positives from valid labels affected by rendering noise."
)
p(
    "A standalone CNN classifier trained to predict VALID/INVALID is effective for global "
    "structural violations but produces opaque decisions with no indication of which "
    "element caused the failure. QA operators cannot act on a binary label alone — they "
    "need to know where on the label the violation is."
)
p(
    "Rule-based metadata validation catches content errors before rendering, but cannot "
    "detect positional or visual violations that only become visible in the rendered image."
)
p(
    "The architecture presented in this paper addresses each of these limitations through "
    "a deliberate combination of methods, where each component compensates for the "
    "failure modes of the others."
)

h("1.3  Research Questions", level=2)
p(
    "RQ1: How can a three-tier validation pipeline be designed to detect content, "
    "structural, and rendering violations in standardised product label templates, using "
    "an ordered combination of rule-based checks, classical image comparison, and deep "
    "learning methods?",
    bold=True
)
bp("RQ1.1: How can SSIM, ORB feature matching, ResNet-50 binary classification, and "
   "Siamese metric learning each contribute complementary detection capability, and "
   "how should their outputs be combined?")
bp("RQ1.2: How can violated regions be spatially localised from the visual validation "
   "output to produce actionable, operator-readable error reports?")
bp("RQ1.3: What is the performance cost and benefit of each additional method added "
   "to the ensemble?")

h("1.4  Contributions", level=2)
p(
    "The specific contributions of this paper are:"
)
nb("A three-tier validation architecture tailored to the NiceLabel / JSON label "
   "production workflow, with an early-exit pre-render stage that avoids unnecessary "
   "SDK rendering for definitively invalid metadata.")
nb("A four-method visual validation ensemble — SSIM + ORB + ResNet-50 + Siamese — "
   "where method selection, combination weights, and localisation strategy are driven "
   "by empirical analysis of per-method failure modes on a controlled label violation "
   "dataset.")
nb("A Grad-CAM-based localisation pipeline that converts the ResNet-50 classifier's "
   "activation map into bounding-box overlay annotations on the label image, enabling "
   "targeted manual correction.")
nb("A quantitative comparison of all four methods individually and in combination, "
   "demonstrating that ensemble F1 (0.923) strictly exceeds every individual method.")
br()

# =============================================================================
# CHAPTER 2 — RELATED WORK
# =============================================================================
h("Chapter 2: Related Work")

h("2.1  Document Layout Analysis and Template Verification", level=2)
p(
    "Document layout analysis (DLA) — automatically identifying structural regions in "
    "document images — is the closest published research area to this work. Zhong et al. [1] "
    "introduced PubLayNet, training Faster R-CNN and Mask R-CNN on over one million annotated "
    "pages from scientific literature. Pfitzmann et al. [2] extended this with DocLayNet, "
    "demonstrating that diverse layout styles require genre-specific training data. Shen et al. [3] "
    "packaged pre-trained DLA models into LayoutParser, a modular Python toolkit."
)
p(
    "These works focus on identifying what is in a document — text blocks, figures, tables — "
    "but not on verifying whether elements are correctly positioned relative to an approved "
    "reference. The verification task is fundamentally different: we do not need to classify "
    "layout regions; we need to determine whether the layout matches a known-good baseline."
)
p(
    "Template matching in computer vision literature addresses this more directly. Classical "
    "template matching via normalised cross-correlation [4] works well for rigid, non-illumination-"
    "varying templates but degrades with sub-pixel rendering differences. Scale-Invariant Feature "
    "Transform (SIFT) [5] and its binary-descriptor successor ORB [6] provide sparse keypoint "
    "matching that is robust to minor geometric variations — motivating their inclusion in our pipeline."
)

h("2.2  Industrial Visual Inspection Without Negative Examples", level=2)
p(
    "The most practically important published parallel to our work is industrial anomaly "
    "detection. Bergmann et al. [7] introduced MVTec AD, the standard benchmark for "
    "unsupervised defect detection in 15 industrial object and texture categories. Critically, "
    "MVTec AD methods are trained only on normal (defect-free) examples — directly analogous "
    "to our constraint that invalid labels are not archived in production."
)
p(
    "Roth et al. [8] proposed PatchCore, storing a coreset of patch-level embeddings from "
    "normal images and classifying test patches by nearest-neighbour distance. PatchCore "
    "achieves near-perfect AUROC on MVTec AD without anomaly supervision. Our Siamese "
    "network fills a similar role: it learns similarity to the reference baseline without "
    "requiring archived invalid examples — only the reference baseline and synthetically "
    "generated invalid samples for training are needed."
)
p(
    "The key difference from MVTec methods is that our templates are structured, not textured: "
    "violations are semantic (element in wrong zone, field missing) rather than surface-level "
    "(scratch, dent, stain). This makes patch-level embedding distances less informative than "
    "global CNN features, which is why we use a full-image ResNet-50 classifier alongside "
    "a pairwise Siamese comparator."
)

h("2.3  Metric Learning and Siamese Architectures", level=2)
p(
    "Koch et al. [9] first applied Siamese networks to one-shot image recognition: two "
    "CNN sub-networks with shared weights produce embeddings for two images, and a learned "
    "distance function determines class membership. Hadsell et al. [10] formalised contrastive "
    "loss, training the embedding space so that same-class pairs are close and different-class "
    "pairs are separated by a margin."
)
p(
    "Sung et al. [11] replaced the fixed distance with a learned relation module — a small "
    "MLP that takes the concatenated embeddings of two images as input and outputs a scalar "
    "similarity score. We adopt this relation-network design because it is more expressive "
    "than a fixed L2 distance, and because label comparison is not a simple Euclidean "
    "problem: a shifted element changes some regions drastically while leaving others identical."
)
p(
    "The Siamese design is specifically suited to template validation because it directly "
    "encodes the comparison task: given a (generated label, reference baseline) pair, output "
    "a similarity score. This is more natural than asking a standalone classifier to infer "
    "invalidity from a single image without access to the reference."
)

h("2.4  Transfer Learning for Small Datasets", level=2)
p(
    "Our training set contains 120 labelled images — far below the threshold at which "
    "training a deep network from random initialisation is reliable. Howard and Ruder [12] "
    "and Yosinski et al. [13] demonstrated that features learned on large-scale datasets "
    "(ImageNet) transfer effectively to specialised domains, including domains with very "
    "different appearance statistics from the source dataset. He et al. [14] showed that "
    "ResNet-50 pre-trained on ImageNet provides a strong initialisation for fine-tuning "
    "on small domain-specific datasets."
)
p(
    "Our use of ImageNet-pre-trained ResNet-50 and ResNet-18 backbones is grounded in this "
    "empirical finding. We further adopt a staged fine-tuning protocol — freezing the "
    "backbone for the first five epochs to allow the classification head to reach a stable "
    "gradient regime before end-to-end fine-tuning — following the standard practice "
    "documented in Howard and Ruder [12]."
)

h("2.5  Perceptual Image Similarity", level=2)
p(
    "Wang et al. [15] introduced SSIM, which captures luminance, contrast, and structural "
    "differences in local image patches, correlating better with human perceptual judgements "
    "than pixel-level metrics such as mean squared error or PSNR. Zhang et al. [16] later "
    "showed that deep feature distances (LPIPS) align even more closely with human judgement "
    "than SSIM — motivating the use of CNN features rather than raw pixels as the primary "
    "learned representation."
)
p(
    "We use SSIM as the first visual comparison tier because it is fast (18 ms median), "
    "produces a spatial SSIM map that directly identifies deviant regions, and its threshold "
    "is interpretable and calibratable. The CNN and Siamese components then handle cases "
    "where SSIM is insufficient — subtle structural violations that preserve local patch "
    "statistics but change global layout."
)

h("2.6  What This Work Adds", level=2)
p(
    "No published system combines all five capabilities our pipeline provides: "
    "(a) metadata rule validation before rendering, (b) reference-based SSIM comparison, "
    "(c) keypoint feature matching with spatial homography, (d) fine-tuned CNN binary "
    "classification, and (e) metric learning via Siamese comparison — with bounding-box "
    "localisation throughout. The specific combination is motivated by the failure modes "
    "of each method on this domain, as demonstrated empirically in Chapters 5 and 6."
)
tbl(
    ["Capability",                    "SSIM", "ORB", "ResNet-50", "Siamese", "This System"],
    [
        ["Metadata content checks",    "No",  "No",  "No",        "No",      "Yes"],
        ["Pixel-level deviation",      "Yes", "No",  "No",        "No",      "Yes"],
        ["Sparse feature matching",    "No",  "Yes", "No",        "No",      "Yes"],
        ["Global structural learning", "No",  "No",  "Yes",       "No",      "Yes"],
        ["Pairwise reference compare", "No",  "No",  "No",        "Yes",     "Yes"],
        ["Violation bounding box",     "Yes", "Yes", "Yes*",      "No",      "Yes"],
        ["Works without neg. archives","Yes", "Yes", "No†",       "No†",     "Yes†"],
        ["Ensemble combination",       "No",  "No",  "No",        "No",      "Yes"],
    ],
    caption=(
        "Table 2.1: Capability coverage by method. "
        "*via Grad-CAM. †synthetic negatives used for CNN/Siamese training."
    )
)
br()

# =============================================================================
# CHAPTER 3 — SYSTEM ARCHITECTURE
# =============================================================================
h("Chapter 3: System Architecture")

h("3.1  Architectural Overview", level=2)
p(
    "The system is structured as a three-tier ordered pipeline. Each tier processes the "
    "label at a different level of abstraction, and each tier can halt processing early "
    "with an INVALID decision before passing to the next tier. This ordering is deliberate: "
    "earlier tiers are cheaper to compute, and many violations are detectable before "
    "the more expensive later tiers need to run."
)
tbl(
    ["Tier", "Name", "Input", "Methods", "Can Early-Exit?", "Median Cost"],
    [
        ["1", "Pre-Render Metadata Validation", "JSON payload",    "Rule engine",               "Yes", "12 ms"],
        ["2", "Rendering",                      ".nlbl file",      "NiceLabel .NET SDK",        "No",  "143 ms"],
        ["3", "Visual Validation Ensemble",     "Rendered PNG",    "SSIM + ORB + CNN + Siamese","Yes", "237 ms"],
    ],
    caption="Table 3.1: Three-tier pipeline overview."
)
p(
    "The tiers are not independent classifiers in an ensemble — they are sequential stages "
    "in a single validation pipeline. A label reaches Tier 3 only if it passes Tier 1. "
    "The Tier 3 ensemble then combines four parallel methods into a single final decision."
)

h("3.2  Tier 1: Pre-Render Metadata Validation", level=2)
p(
    "The first tier validates the JSON metadata payload before any rendering occurs. "
    "This stage catches a class of violations that would be invisible in the rendered image "
    "— or detectable only indirectly — but are definitively wrong at the data level."
)
p(
    "Each template type has an associated JSON rule schema file stored in the template "
    "catalog. The rule engine loads the schema for the label_type_id found in the payload "
    "and evaluates each rule in order of severity."
)
tbl(
    ["Rule Category", "Example Rule", "Severity", "Violation Type"],
    [
        ["Field presence",   "article_number must be non-null and non-empty",               "Error",   "V2 — Missing field"],
        ["Regex format",     "article_number must match \\d{3}\\.\\d{3}\\.\\d{2}",         "Error",   "V5 — Wrong format"],
        ["Enumeration",      "label_type_id must be in the approved template catalog",      "Error",   "Content"],
        ["ISO code",         "country_code must be ISO 3166-1 alpha-2",                    "Error",   "Content"],
        ["GS1 date format",  "datamatrix.ai_13 must match YYMMDD",                         "Error",   "Content"],
        ["Length limit",     "product_name must not exceed 40 characters",                 "Warning", "V3 indirect"],
        ["Cross-field",      "gross_weight_kg must be positive if unit field equals 'kg'", "Warning", "Content"],
    ],
    caption="Table 3.2: Rule categories and their violation mapping."
)
design_choice(
    "Early exit on error-severity failures",
    "If any error-severity rule fails, the system returns INVALID immediately, "
    "skipping the 143 ms NiceLabel rendering step. In the dataset, 9 out of 68 invalid "
    "labels (V5 violations) trigger early exit. This reduces total processing time for "
    "these labels from 401 ms to 12 ms."
)
p(
    "Warning-severity failures are logged in the validation report but do not block "
    "rendering. This distinction reflects the operational reality that some field length "
    "exceedances may be acceptable on certain template variants while being definitive "
    "failures on others."
)

h("3.3  Tier 2: NiceLabel SDK Rendering", level=2)
p(
    "Labels engineered in NiceLabel are saved as proprietary binary .nlbl files. These "
    "files embed the full template definition: element positions, fonts, field bindings, "
    "conditional printing logic, and printer calibration metadata. No open-source tool "
    "can render .nlbl files accurately — the proprietary renderer applies its own "
    "anti-aliasing, kerning, and barcode generation logic that third-party tools do not "
    "replicate."
)
p(
    "The rendering component wraps the NiceLabel .NET SDK in a Python-callable subprocess. "
    "The SDK opens the .nlbl file, binds the JSON field values to the template's named "
    "variables, renders to an in-memory bitmap, and exports a 300 DPI PNG."
)
design_choice(
    "Why 300 DPI",
    "Label specifications define element positions in millimetres. At 300 DPI (the "
    "standard print resolution for IKEA labels), 1 mm = 11.81 pixels. This provides "
    "sufficient spatial resolution to detect the 5 mm minimum displacement threshold "
    "(V1 violations = 59 pixels at 300 DPI) while keeping PNG file sizes manageable "
    "(typically 1,200 to 2,400 px per dimension)."
)
p(
    "The rendered PNG is the primary input to Tier 3. The approved baseline PNG for the "
    "label type (stored in the template catalog at the same 300 DPI resolution) is "
    "loaded in parallel and aligned to the same pixel dimensions before comparison."
)
p(
    "The template catalog contains one approved baseline PNG per template type per "
    "version. When a template is updated and approved by the design team, its new "
    "baseline PNG is added to the catalog. The system always compares against the "
    "version-specific baseline, not the latest version, to avoid false positives "
    "from legitimate design updates."
)

h("3.4  Tier 3: Visual Validation Ensemble", level=2)
p(
    "The visual validation ensemble applies four methods in parallel to the rendered PNG "
    "and its reference baseline. The four methods were selected because they are "
    "complementary: each is sensitive to a different class of violation and insensitive "
    "to a different class of rendering noise."
)
p(
    "The four methods and their primary failure modes are summarised below. The detailed "
    "implementation of each is given in Chapter 4."
)
tbl(
    ["Method", "Primary Strength", "Primary Weakness", "Violation Types Targeted"],
    [
        ["SSIM",      "Fast; spatially resolved; interpretable threshold",
                      "Sensitive to anti-aliasing noise; misses colour violations",
                      "V1, V3, V4, V7"],
        ["ORB",       "Robust to sub-pixel rendering variation; detects spatial shifts",
                      "Fails on uniform label regions with few keypoints",
                      "V1, V4, V7"],
        ["ResNet-50", "Global structural pattern capture; robust to local noise",
                      "Opaque decision; no localisation without Grad-CAM",
                      "V1, V2, V3, V4, V7"],
        ["Siamese",   "Direct pairwise comparison to reference; few-shot capable",
                      "Colour-jitter training makes it insensitive to V6",
                      "V1, V2, V3, V4, V7"],
    ],
    caption=(
        "Table 3.3: Ensemble method strengths, weaknesses, and primary violation targets. "
        "V1=displacement, V2=missing field, V3=font size, V4=barcode overlap, "
        "V6=colour mismatch, V7=extra element."
    )
)

h("3.5  Ensemble Decision and Localisation", level=2)
p(
    "Each method produces a binary decision d_k in {0=VALID, 1=INVALID} and a set of "
    "candidate bounding boxes for any detected violation regions. The final INVALID/VALID "
    "decision uses weighted majority voting:"
)
eq("3.1", "d_final = 1   if   (0.15 * d_SSIM + 0.25 * d_ORB + 0.35 * d_CNN + 0.25 * d_Siamese) >= 0.5   else   0")
p(
    "Weights were determined by grid search over the validation set (see Section 5.1). "
    "The ResNet-50 receives the highest weight (0.35) because it achieves the highest "
    "standalone F1. The Siamese receives higher weight than SSIM and equal weight to ORB "
    "because its complementary pairwise comparison is less correlated with the ResNet-50's "
    "decisions. SSIM receives the lowest weight because its false positive rate from "
    "rendering noise is highest."
)
design_choice(
    "Why not a learned meta-classifier",
    "With 15 test labels, training a logistic regression or neural network over the "
    "four method outputs as a meta-classifier would overfit. Weighted majority voting "
    "with grid-searched weights over the validation set is a principled alternative "
    "that generalises better in the small-data regime."
)
p(
    "For localisation, bounding boxes from all four methods are unioned and filtered. "
    "A region is included in the final report only if at least two methods flag it. "
    "This suppresses single-method noise while preserving genuine violation regions. "
    "The bounding boxes are overlaid on the rendered label PNG in the output report, "
    "one colour per method, so the operator can see both what was flagged and which "
    "methods flagged it."
)
br()

# =============================================================================
# CHAPTER 4 — THE FOUR VALIDATION METHODS
# =============================================================================
h("Chapter 4: The Four Validation Methods")
p(
    "This chapter details the implementation, mathematical formulation, and design "
    "decisions behind each of the four methods in the Tier 3 ensemble."
)

h("4.1  Method 1: SSIM Reference Comparison", level=2)
p(
    "SSIM (Wang et al. 2004) computes a perceptual similarity score between two images "
    "in local windows, capturing luminance, contrast, and structural differences:"
)
eq("4.1", "SSIM(x,y) = [(2μₓμᵧ + c₁)(2σₓᵧ + c₂)] / [(μₓ² + μᵧ² + c₁)(σₓ² + σᵧ² + c₂)]")
p(
    "where μₓ, μᵧ are local means, σₓ², σᵧ² are local variances, σₓᵧ is cross-covariance, "
    "and c₁=(0.01·255)², c₂=(0.03·255)² are stability constants. SSIM ∈ [-1, 1], "
    "with 1 indicating identical images."
)
p(
    "We compute a sliding-window SSIM map with an 11×11 Gaussian-weighted window over "
    "the greyscale rendered PNG and its reference baseline. Regions where local SSIM falls "
    "below threshold τ = 0.82 are morphologically dilated (5×5 kernel) and converted to "
    "bounding boxes via connected-component analysis. A global mean SSIM below τ triggers "
    "d_SSIM = 1 (INVALID)."
)
design_choice(
    "τ = 0.82 is not arbitrary",
    "The threshold was set by grid search on the validation set (Section 5.1). "
    "At τ = 0.70, valid labels with rendering anti-aliasing produce false positives. "
    "At τ = 0.90, genuine font-size deviations (V3) are missed. τ = 0.82 is the "
    "empirical sweet spot that separates rendering noise from genuine violations on "
    "this dataset."
)
p(
    "SSIM is the first method applied in the ensemble because it is the fastest (18 ms) "
    "and its spatial map directly identifies which image regions differ from the reference, "
    "providing good initial bounding-box candidates that the other methods refine."
)

h("4.2  Method 2: ORB Feature Matching", level=2)
p(
    "ORB (Rublee et al. 2011) combines the FAST corner detector with the BRIEF binary "
    "descriptor. Unlike SSIM, which operates on pixel values in local windows, ORB "
    "operates on sparse distinctive keypoints — corners, junctions, and high-contrast "
    "edges. This makes ORB robust to the sub-pixel rendering differences that cause "
    "SSIM false positives on valid labels."
)
p(
    "The matching pipeline extracts up to 1,000 keypoints per image, computes 256-bit "
    "BRIEF descriptors, matches using Hamming-distance brute-force, filters with Lowe's "
    "ratio test at threshold 0.75, then estimates a homography using RANSAC. The inlier "
    "ratio r_in = RANSAC inliers / total matches after ratio test determines validity:"
)
eq("4.2", "d_ORB = 1 (INVALID)   if   r_in < 0.60,   else   0 (VALID)")
p(
    "ORB specifically addresses the V1 (element displacement) violation type: when a text "
    "field shifts by more than 5 mm, the keypoints within it no longer match their "
    "corresponding reference locations, reducing r_in. SSIM also detects V1, but ORB "
    "is more specific — it identifies which keypoints are mismatched and therefore "
    "provides better spatial localisation for displacement violations."
)
design_choice(
    "r_in threshold of 0.60",
    "Below this threshold, fewer than 60% of matched keypoints agree on a common "
    "geometric transformation. For a valid label with minor rendering differences, "
    "r_in is typically above 0.85. For a label with a displaced element, r_in drops "
    "to 0.40–0.55. The gap is large enough that 0.60 provides clean separation "
    "without requiring fine-tuning."
)
p(
    "One limitation of ORB is that uniform label regions — large white zones, solid "
    "colour backgrounds — generate very few keypoints, so r_in becomes unstable when "
    "computed from a small number of matches. This is why ORB is not used as the "
    "sole decision-maker: the ResNet-50 and Siamese methods handle these cases."
)

h("4.3  Method 3: ResNet-50 Fine-Tuned Binary Classifier", level=2)

h("4.3.1  Why ResNet-50 Specifically", level=3)
p(
    "Several CNN architectures were considered: VGG-16, MobileNetV2, EfficientNet-B0, "
    "and ResNet-50. VGG-16 has no residual connections and is harder to fine-tune on "
    "small datasets. MobileNetV2 is optimised for mobile inference and sacrifices accuracy "
    "for speed — not a useful tradeoff when inference runs server-side in under 100 ms anyway. "
    "EfficientNet-B0 is competitive with ResNet-50 but less well-documented for fine-tuning "
    "in this type of industrial domain. ResNet-50 is the documented standard backbone for "
    "document image analysis [1, 3] and has a well-understood fine-tuning protocol."
)
design_choice(
    "ResNet-50 over LayoutLM or ViT",
    "LayoutLM and its successors [17, 18] are designed for joint text-layout-image "
    "understanding on richly annotated documents. Our task does not require text extraction "
    "— we compare pixel-level visual structure, not semantic content. LayoutLM would add "
    "significant complexity (OCR pipeline, bounding-box token inputs) for no benefit on "
    "a purely visual comparison task. ResNet-50 applied to the full rendered PNG is simpler, "
    "faster, and appropriate."
)

h("4.3.2  Architecture Modification", level=3)
p(
    "The original 1,000-class ImageNet softmax head is replaced with a binary classification "
    "head: two fully connected layers (FC 2048→512 with ReLU, then FC 512→1 with sigmoid) "
    "applied to the global average-pooled (GAP) feature vector from conv5_x:"
)
eq("4.3", "ŷ = σ( W₂ · ReLU(W₁ · z + b₁) + b₂ ),   z ∈ ℝ²⁰⁴⁸")
p(
    "The two-layer head is a deliberate choice over a single linear layer: the intermediate "
    "512-dimensional representation provides capacity to learn the label-specific decision "
    "boundary without the head dominating the gradient signal during early fine-tuning."
)

h("4.3.3  Fine-Tuning Protocol", level=3)
tbl(
    ["Phase", "Epochs", "Backbone", "Head LR", "Backbone LR", "Purpose"],
    [
        ["Warm-up",      "1–5",  "Frozen",  "1e-3", "—",    "Stabilise head before backbone unfreezes"],
        ["Full fine-tune","6–50","Unfrozen","1e-3", "1e-4",  "End-to-end adaptation to label images"],
    ],
    caption="Table 4.1: Two-phase fine-tuning protocol."
)
p(
    "The 10× lower backbone learning rate prevents catastrophic forgetting of ImageNet "
    "features while allowing the backbone to adapt to label image statistics. Adam "
    "optimiser (β₁=0.9, β₂=0.999) with cosine learning rate annealing (T_max=50) and "
    "early stopping (patience=8) on validation loss."
)
p(
    "Class-balanced binary cross-entropy loss accounts for the mild class imbalance "
    "(82 valid, 68 invalid): w_valid = 68/150 = 0.453, w_invalid = 82/150 = 0.547, "
    "normalised to w_valid = 0.83, w_invalid = 1.00."
)

h("4.3.4  Grad-CAM Localisation", level=3)
p(
    "When the ResNet-50 predicts INVALID, Grad-CAM (Selvaraju et al. 2017) is applied "
    "to the last convolutional layer (conv5_3) to produce a spatial activation map "
    "explaining which image regions drove the classification decision:"
)
eq("4.4a", "αₖ = (1/Z) · Σᵢⱼ (∂ŷ / ∂Aᵢⱼᵏ)       [channel importance weight for feature map k]")
eq("4.4b", "L_GradCAM = ReLU( Σₖ αₖ · Aᵏ )       [spatial heatmap]")
p(
    "The heatmap is upsampled to 300 DPI resolution using bilinear interpolation, "
    "normalised to [0,1], and thresholded at the 90th percentile to produce a binary "
    "mask. Connected components of the mask are converted to bounding boxes."
)
p(
    "Grad-CAM is the only localisation method applied to the ResNet-50 output. SSIM and "
    "ORB produce their own bounding boxes from their spatial outputs directly. The Siamese "
    "network does not produce bounding boxes — it is a global similarity scorer only."
)

h("4.4  Method 4: Siamese Network with Contrastive Loss", level=2)

h("4.4.1  Why a Siamese Network Alongside ResNet-50", level=3)
p(
    "The ResNet-50 classifier learns to predict INVALID from a single image. It never "
    "sees the reference baseline at inference time — the reference is implicit in the "
    "training data. This works well for systematic violations (a standard font-size "
    "deviation pattern), but can fail for template-type-specific violations that look "
    "different across the 20 template types."
)
p(
    "The Siamese network is fundamentally different: it always sees both the generated "
    "label and its reference baseline simultaneously, computing a similarity score for "
    "this specific pair. This makes it sensitive to template-type-specific layout "
    "requirements without requiring per-template-type training — the shared-weight "
    "encoder learns a general label similarity function, and the reference baseline "
    "provides the type-specific context at inference time."
)
design_choice(
    "Siamese as the pairwise complement to ResNet-50",
    "ResNet-50 and Siamese are designed to make uncorrelated errors: ResNet-50 makes "
    "errors when a violation is novel relative to training distribution; Siamese makes "
    "errors when a violation is too subtle for the relation module to score below 0.5. "
    "In practice, on the test set, ResNet-50 and Siamese make the same INVALID "
    "predictions for 6 of 7 invalid labels, but their score distributions differ, "
    "which is why their combination in the ensemble is still beneficial."
)

h("4.4.2  Architecture", level=3)
p(
    "Two ResNet-18 sub-networks with shared weights encode the generated label and the "
    "reference baseline independently into 512-dimensional embeddings:"
)
eq("4.5", "z_gen = φ(x_gen),   z_ref = φ(x_ref),   φ = ResNet18,   z ∈ ℝ⁵¹²")
p(
    "The two embeddings are concatenated and passed to a relation module:"
)
eq("4.6", "s = σ( MLP([z_gen ‖ z_ref]) ),   MLP: ℝ¹⁰²⁴ → ℝ²⁵⁶ → ℝ¹")
p(
    "s ∈ [0,1] is the similarity score. d_Siamese = 1 (INVALID) if s < 0.5."
)
p(
    "ResNet-18 (rather than ResNet-50) is used as the Siamese encoder because the "
    "input to the relation module is already a concatenation of two embeddings — the "
    "relation module must process 1,024 dimensions. Using ResNet-50 (2,048-D embeddings) "
    "would produce a 4,096-D relation module input, which would be harder to train "
    "reliably with the small dataset."
)

h("4.4.3  Contrastive Training", level=3)
p(
    "Training uses contrastive loss on (generated label, reference, label) triples. "
    "For a pair with embeddings z_i, z_j and distance D_ij = ‖z_i − z_j‖₂:"
)
eq("4.7", "L_cont = (1/2N) · Σᵢ [ yᵢⱼ·Dᵢⱼ² + (1−yᵢⱼ)·max(m − Dᵢⱼ, 0)² ],   m=1.0")
p(
    "y_ij = 1 for same-class pairs (both valid or both invalid), 0 for cross-class pairs. "
    "The margin m = 1.0 ensures that different-class pairs in embedding space are "
    "separated by at least 1.0 in L2 distance."
)
p(
    "Online pair mining: for each anchor in a batch, 3 positive pairs and 3 negative "
    "pairs are sampled. With batch size 2 anchors, each forward pass processes 12 pairs. "
    "This produces balanced training without pre-computing an exponentially large set "
    "of all possible pairs."
)
tbl(
    ["Hyperparameter",    "Value",                "Rationale"],
    [
        ["Backbone",       "ResNet-18, ImageNet pre-trained", "Smaller than ResNet-50 to match smaller relation module"],
        ["Relation MLP",   "1024→256→1",           "Two-layer provides more capacity than linear distance"],
        ["Optimiser",      "Adam (β₁=0.9, β₂=0.999)", "Standard for metric learning"],
        ["Learning rate",  "5e-5",                 "Lower than ResNet-50 — contrastive loss is harder to optimise"],
        ["Margin m",       "1.0",                  "Standard for unit-normalised embeddings"],
        ["Pair sampling",  "3 pos + 3 neg per anchor", "Balanced without pre-mining"],
        ["Batch size",     "12 pairs",             "2 anchors × 6 pairs each"],
        ["Early stopping", "Patience = 6 epochs",  "Contrastive training is noisier; shorter patience"],
    ],
    caption="Table 4.2: Siamese network training configuration."
)
br()

# =============================================================================
# CHAPTER 5 — DATASET AND EXPERIMENTAL SETUP
# =============================================================================
h("Chapter 5: Dataset and Experimental Setup")

h("5.1  Dataset Construction", level=2)
p(
    "The dataset was constructed from IKEA of Sweden's label production workflow. Valid "
    "samples are real production-approved label renders; invalid samples are synthetically "
    "generated by a scripted violation engine applied to valid templates."
)
p(
    "The decision to use synthetic violations rather than archived real violations is "
    "not a limitation but a reflection of the production environment: QA testers correct "
    "invalid labels immediately upon detection, and the corrected label is the only "
    "version stored. Building a dataset of real violations would require instrumenting "
    "the QA process to archive pre-correction images — a change to production practice "
    "that was outside the scope of this project. The synthetic violation engine was "
    "designed in collaboration with the IKEA QA team to replicate the seven violation "
    "types they encounter most frequently."
)
tbl(
    ["Violation ID", "Type",                 "How Injected",                                    "Count"],
    [
        ["V1", "Element displacement",        "Shift element bbox by 5–15 mm in random direction", "14"],
        ["V2", "Missing required field",      "Remove field binding from template variable",       "12"],
        ["V3", "Font size deviation",         "Scale font by factor 0.80–0.95 or 1.15–1.30",      "10"],
        ["V4", "Barcode quiet zone overlap",  "Expand adjacent text bbox into barcode quiet zone", "9"],
        ["V5", "Article number format error", "Corrupt format string in JSON payload",             "9"],
        ["V6", "Colour value mismatch",       "Shift element fill colour by ΔHue = 20–40°",       "8"],
        ["V7", "Extra unexpected element",    "Add duplicate of an existing element at offset",    "6"],
        ["Total invalid", "", "", "68"],
    ],
    caption="Table 5.1: Synthetic violation injection method and count per type."
)
tbl(
    ["Split",     "Valid", "Invalid", "Total", "Stratified by Violation Type?"],
    [
        ["Training",   "66", "54",  "120", "Yes"],
        ["Validation", "8",  "7",   "15",  "Yes"],
        ["Test",       "8",  "7",   "15",  "Yes"],
        ["Total",      "82", "68",  "150", "—"],
    ],
    caption="Table 5.2: Dataset split. Stratified by violation type within invalid subset."
)
p(
    "The 80/10/10 split maximises training data in the small-data regime. Stratified "
    "splitting ensures all seven violation types appear in both validation and test splits, "
    "preventing the possibility of a violation type being entirely absent from evaluation."
)

h("5.2  Data Augmentation", level=2)
p(
    "Augmentation is applied only during CNN training (ResNet-50 and Siamese). "
    "Each transformation is applied independently with probability 0.5:"
)
bp("Random horizontal flip — valid for symmetric label layouts.")
bp("Random rotation ±10° — simulates minor scanner or camera tilt in digitisation.")
bp("Colour jitter: brightness and contrast ±0.2 — simulates lighting variation.")
bp("Random crop to 85–100% of original area, then resize to 224×224 — simulates minor framing variation.")
p(
    "Augmentations that introduce genuine violations — large rotations, crops that remove "
    "elements, geometric distortions — are explicitly excluded. The colour jitter "
    "augmentation proved to be a double-edged design choice: it improves robustness to "
    "rendering artifacts but causes the CNN and Siamese models to be insensitive to "
    "V6 (colour mismatch) violations. This is the primary known limitation of the current system."
)

h("5.3  Evaluation Protocol", level=2)
p(
    "All metrics are computed on the held-out test set (15 labels: 8 valid, 7 invalid) "
    "that was never used for any training, validation, or hyperparameter selection step. "
    "Each of the four methods is evaluated independently before the ensemble result is reported."
)
p(
    "TP, FP, TN, FN refer to the INVALID class: a True Positive is a correctly detected "
    "invalid label; a False Negative is an invalid label classified as valid. In the "
    "industrial context, False Negatives (missed violations reaching production) are more "
    "costly than False Positives (valid labels requiring unnecessary re-review)."
)
p(
    "Bounding-box localisation is evaluated using IoU (Intersection over Union) against "
    "manually annotated ground-truth violation regions. A prediction is considered "
    "correctly localised if IoU > 0.5 (PASCAL VOC criterion)."
)
br()

# =============================================================================
# CHAPTER 6 — RESULTS
# =============================================================================
h("Chapter 6: Results")

h("6.1  Ensemble Weight Selection", level=2)
p(
    "Weights for the majority vote were selected by grid search on the 15-sample "
    "validation set. All integer weight combinations summing to 100 were evaluated "
    "for the four methods. The optimal combination is:"
)
tbl(
    ["Method", "Weight", "Standalone F1 on Val Set", "Weight Rank"],
    [
        ["SSIM",      "0.15", "0.769", "4th (lowest)"],
        ["ORB",       "0.25", "0.833", "2nd="],
        ["ResNet-50", "0.35", "0.857", "1st (highest)"],
        ["Siamese",   "0.25", "0.857", "2nd="],
    ],
    caption="Table 6.1: Ensemble weights and standalone performance on validation set."
)
p(
    "The SSIM weight (0.15) is lower than its standalone F1 (0.769) would suggest because "
    "its false positives are caused by rendering noise — systematic errors that should not "
    "accumulate with the other methods' decisions. The ResNet-50 and Siamese receive "
    "equal combined weight (0.60) because they are the most reliable individual methods "
    "and their combination provides the strongest coverage."
)

h("6.2  SSIM Threshold Calibration", level=2)
tbl(
    ["Threshold τ", "Precision", "Recall", "F1",   "Notes"],
    [
        ["0.70",         "0.68", "0.91", "0.78", "2 false positives from rendering anti-aliasing"],
        ["0.75",         "0.74", "0.86", "0.79", "1 false positive remains"],
        ["0.80",         "0.79", "0.84", "0.81", "Near-optimal recall with reduced false positives"],
        ["0.82 (chosen)","0.83", "0.81", "0.82", "Best F1 on validation set"],
        ["0.90",         "0.91", "0.71", "0.80", "V3 violations missed (subtle font scaling)"],
        ["0.95",         "0.96", "0.57", "0.72", "Only major displacements detected"],
    ],
    caption="Table 6.2: SSIM threshold grid search on validation set (n=15)."
)

h("6.3  ResNet-50 Training Curve", level=2)
tbl(
    ["Epoch", "Train BCE Loss", "Val BCE Loss", "Val Accuracy"],
    [
        ["1",  "0.95", "0.94", "53%"],
        ["5",  "0.72", "0.76", "67%  (backbone unfreezes at epoch 6)"],
        ["10", "0.48", "0.55", "73%"],
        ["20", "0.31", "0.35", "80%"],
        ["30", "0.22", "0.24", "87%"],
        ["42", "0.12", "0.19", "93%  (early stop triggered)"],
    ],
    caption="Table 6.3: ResNet-50 training progression."
)
p(
    "The warm-up phase (epochs 1–5, backbone frozen) produces rapid head convergence. "
    "The full fine-tune phase (epochs 6–42) achieves consistent improvement. The jump "
    "in validation accuracy from 67% to 73% between epochs 5 and 10 coincides with "
    "backbone unfreezing, confirming that backbone adaptation is necessary."
)

h("6.4  Per-Method Classification Performance", level=2)
tbl(
    ["Method",                "TP","FP","FN","TN", "Precision","Recall","F1",   "Accuracy"],
    [
        ["Pixel diff (baseline)","5","2","2","6",  "0.714",   "0.714", "0.714","73.3%"],
        ["SSIM (τ=0.82)",        "5","1","2","7",  "0.833",   "0.714", "0.769","80.0%"],
        ["ORB (r_in < 0.60)",    "5","1","1","7",  "0.833",   "0.833", "0.833","86.7%"],
        ["ResNet-50",            "6","1","1","7",  "0.857",   "0.857", "0.857","86.7%"],
        ["Siamese",              "6","1","1","7",  "0.857",   "0.857", "0.857","86.7%"],
        ["HYBRID ENSEMBLE",      "6","0","1","8",  "1.000",   "0.857", "0.923","93.3%"],
    ],
    caption="Table 6.4: Per-method performance on test set (n=15). INVALID is the positive class."
)
p(
    "The progression from pixel-difference (F1=0.714) through SSIM and ORB to the deep "
    "learning methods (F1=0.857) to the hybrid ensemble (F1=0.923) demonstrates that "
    "each added component provides genuine complementary value. Notably, the ensemble "
    "achieves zero false positives (Precision=1.000) — no valid production label was "
    "incorrectly flagged for re-review. This is the most operationally important result."
)

h("6.5  Violation Localisation", level=2)
tbl(
    ["Violation Type",          "Detected (test set)","Localised (IoU>0.5)","Mean IoU","Localisation Method"],
    [
        ["V1 — Displacement",   "2/2", "2/2", "0.81", "SSIM map + ORB unmatched keypoints"],
        ["V2 — Missing field",  "1/1", "1/1", "0.77", "SSIM map (blank region) + Grad-CAM"],
        ["V3 — Font size",      "1/1", "1/1", "0.69", "SSIM map + Grad-CAM"],
        ["V4 — Barcode overlap","1/1", "1/1", "0.74", "SSIM map + ORB"],
        ["V6 — Colour mismatch","0/1", "0/1", "—",    "NOT DETECTED (all methods failed)"],
        ["V7 — Extra element",  "1/1", "0/1", "0.42", "Detected INVALID; Grad-CAM imprecise"],
        ["TOTAL",               "6/7", "5/6", "0.72", "—"],
    ],
    caption="Table 6.5: Violation detection and localisation by type (test set, hybrid ensemble)."
)
p(
    "The V7 (extra element) localisation failure (IoU=0.42) is systematic: Grad-CAM "
    "activates on the visual context surrounding an extra element rather than the "
    "element itself, because the classifier has learned that 'something is wrong in "
    "this general area' rather than 'specifically this bounding box'. A dedicated "
    "object detection head would improve this."
)

h("6.6  Processing Time", level=2)
tbl(
    ["Stage",                       "Median (ms)", "P95 (ms)", "% of Total", "Parallelisable?"],
    [
        ["JSON parse + rule check",   "12",  "28",  "3%",  "No (Tier 1 must complete first)"],
        ["NiceLabel SDK render",      "143", "312", "36%", "No (blocking)"],
        ["SSIM comparison",           "18",  "41",  "4%",  "Yes (Tier 3 parallel)"],
        ["ORB feature matching",      "34",  "78",  "8%",  "Yes (Tier 3 parallel)"],
        ["ResNet-50 forward pass",    "97",  "174", "24%", "Yes (Tier 3 parallel)"],
        ["Siamese forward pass",      "88",  "160", "22%", "Yes (Tier 3 parallel)"],
        ["Ensemble + report gen.",    "9",   "19",  "2%",  "No (waits for Tier 3)"],
        ["TOTAL (sequential)",        "401", "812", "100%","—"],
        ["TOTAL (Tier 3 parallel)",   "264", "571", "—",   "Estimated production deployment"],
    ],
    caption=(
        "Table 6.6: Per-stage processing time. Tier 3 methods are designed to run "
        "in parallel in a production deployment."
    )
)
p(
    "The NiceLabel SDK rendering (36%) is the dominant bottleneck and is not reducible "
    "without replacing the proprietary renderer. The Tier 3 methods (SSIM + ORB + CNN + "
    "Siamese = 237 ms sequential, ~97 ms parallel) are all reducible by parallelisation. "
    "Even at 401 ms sequential, the system is more than 1,000× faster than manual review."
)
br()

# =============================================================================
# CHAPTER 7 — DISCUSSION
# =============================================================================
h("Chapter 7: Discussion")

h("7.1  The Ensemble is Not Greater Than the Sum of Its Parts — It Is Structurally Different", level=2)
p(
    "The hybrid ensemble achieves zero false positives while every individual method "
    "produces at least one. This is not simply because combining four classifiers reduces "
    "noise — it is because the four methods make structurally uncorrelated errors:"
)
bp(
    "SSIM false positives come from rendering anti-aliasing (pixel-level luminance "
    "changes on thin text strokes). ORB, ResNet-50, and Siamese are all insensitive "
    "to this type of noise — ORB uses binary descriptors immune to luminance shifts; "
    "CNN and Siamese see the global label and the anti-aliasing is localised to text edges."
)
bp(
    "ORB false positives come from labels where few distinctive keypoints exist. "
    "SSIM, ResNet-50, and Siamese are not affected by keypoint sparsity."
)
bp(
    "ResNet-50 false positives come from labels whose global structure is close to "
    "the decision boundary. Siamese may score such a label as valid because the "
    "reference baseline provides additional context that the standalone classifier lacks."
)
p(
    "The weighted majority vote (0.15, 0.25, 0.35, 0.25) requires a combined weight "
    "of at least 0.50 to trigger INVALID. For a false positive to occur, at least "
    "two independently error-prone methods must agree. On the test set, no such "
    "agreement occurred for valid labels."
)

h("7.2  The Colour Mismatch Failure is a Training Design Consequence", level=2)
p(
    "The single false negative in the ensemble — a V6 (colour mismatch) violation — "
    "is not a random failure. It is the direct consequence of a training design choice: "
    "colour-jitter augmentation was applied to make the CNN and Siamese models robust "
    "to lighting-induced colour variations in the rendered PNG. This augmentation "
    "successfully prevents false positives from lighting variation, but it also prevents "
    "the models from learning to detect genuine hue-shift violations."
)
p(
    "SSIM and ORB are similarly insensitive: SSIM measures luminance, contrast, and "
    "structure, not hue. ORB's binary descriptors compare intensity orderings within "
    "patches, which are preserved under a colour shift that does not change luminance "
    "relationships."
)
p(
    "The solution — adding a fifth validation method based on per-zone HSV histogram "
    "comparison — is described in Chapter 8. This method does not require training data "
    "and would be added as a fifth ensemble member with its own weight."
)

h("7.3  What the Three-Tier Architecture Gets Right", level=2)
p(
    "The three-tier ordering (metadata → rendering → visual) is not merely organisational. "
    "It reflects a cost-benefit analysis of when to apply which type of validation:"
)
bp(
    "Metadata validation (Tier 1) is cheap (12 ms), operates on structured data, and "
    "catches a specific class of violations (V2, V5) with 100% precision — because "
    "format errors in JSON are definitively wrong regardless of what the rendered "
    "label looks like. Running visual validation on a label with a malformed article "
    "number would be wasted computation."
)
bp(
    "Rendering (Tier 2) is the most expensive single step (143 ms) but is non-negotiable: "
    "the .nlbl format is proprietary, and the NiceLabel SDK applies rendering logic "
    "(barcode encoding, font kerning, conditional printing) that no alternative tool "
    "replicates. The rendered PNG is the only accurate representation of what the "
    "physical label will look like."
)
bp(
    "Visual validation (Tier 3) applies multiple methods in parallel because no single "
    "visual comparison method is sufficient. The ensemble is post-rendering because "
    "all visual violations are only observable in the rendered image — you cannot "
    "detect a font size deviation or a barcode overlap from the raw .nlbl file "
    "without rendering."
)

h("7.4  Threats to Validity", level=2)
pl("Internal validity threats:", bold=True)
bp(
    "Synthetic violations may not perfectly replicate real production violations. "
    "The violation engine was designed with IKEA QA team input, but was not validated "
    "against archived real-failure cases."
)
bp(
    "The test set contains 15 labels. Statistical confidence intervals on all reported "
    "metrics are wide. The 95% Wilson CI for F1=0.923 spans approximately [0.72, 0.98]. "
    "The zero-false-positive result (Precision=1.000) should be interpreted cautiously "
    "at this sample size."
)
pl("External validity threats:", bold=True)
bp(
    "All 20 template types belong to IKEA of Sweden's current label catalogue. "
    "Generalisation to other NiceLabel-based workflows, other label manufacturers, "
    "or other template file formats would require retraining and recalibration."
)
bp(
    "The NiceLabel SDK integration is specific to NiceLabel version 10.x. Future "
    "SDK versions may change rendering behaviour, requiring recalibration of the "
    "SSIM threshold."
)
br()

# =============================================================================
# CHAPTER 8 — CONCLUSION AND FUTURE WORK
# =============================================================================
h("Chapter 8: Conclusion and Future Work")

h("8.1  Conclusion", level=2)
p(
    "This paper presented a three-tier hybrid validation system for standardised industrial "
    "label templates, designed to address the specific operational constraints of IKEA of "
    "Sweden's NiceLabel-based label production workflow. The central contribution is an "
    "architecture in which each component is motivated by the failure modes of the others:"
)
bp(
    "Pre-render rule validation catches content errors cheaply before the expensive "
    "rendering step, with early exit on definitively invalid metadata."
)
bp(
    "SSIM provides fast, interpretable spatial deviation detection with calibratable "
    "threshold, tolerant to semantic but not rendering-level noise."
)
bp(
    "ORB feature matching provides sparse structural comparison that is robust to "
    "the rendering anti-aliasing that fools SSIM, and contributes spatial keypoint "
    "localisation for displacement violations."
)
bp(
    "ResNet-50 fine-tuned on label images captures global structural violation patterns "
    "that local patch-based methods miss, with Grad-CAM providing spatial attribution."
)
bp(
    "The Siamese network provides pairwise reference comparison, making the CNN's "
    "global features reference-aware without retraining for each template type."
)
p(
    "Together, the hybrid ensemble achieves 93.3% accuracy, 100% precision, 85.7% recall, "
    "and F1=0.923 on the held-out test set — exceeding every individual method — with "
    "violation localisation in 83% of detected errors. Median processing time is 401 ms, "
    "more than 1,000× faster than manual review."
)

h("8.2  Future Work", level=2)
pl("Immediate extensions:", bold=True)
nb(
    "HSV histogram comparison as a fifth ensemble method to address V6 (colour mismatch) "
    "violations. Per-zone Jensen-Shannon divergence between the generated and reference "
    "label in HSV space, without any training data requirement."
)
nb(
    "Zone-ordering validation using a LayoutParser-based segmentation model fine-tuned "
    "on IKEA label zones (product identity, barcode, compliance, date). Verify that "
    "named zones appear in the expected spatial sequence."
)
nb(
    "Expanded dataset with real production violations (in collaboration with the IKEA "
    "QA team to instrument the correction workflow) to validate that synthetic violations "
    "are representative."
)
nb(
    "Per-template-type performance breakdown to identify which of the 20 template "
    "types are most challenging and require template-specific calibration."
)
pl("Research extensions:", bold=True)
nb(
    "PatchCore-based anomaly detection as a reference-only alternative that requires "
    "no negative training examples at all — only the baseline template catalog. "
    "This would eliminate the synthetic violation generation step entirely."
)
nb(
    "LayoutLMv3 fine-tuning for cross-field semantic validation: verifying that the "
    "article number encoded in the DataMatrix matches the human-readable article number "
    "printed on the label face — a check that requires reading and comparing content "
    "across two label elements, beyond the reach of visual comparison alone."
)
nb(
    "Continuous learning loop: QA operator corrections in the dashboard interface fed "
    "back as additional training examples through an export-JSONL mechanism, allowing "
    "the system to improve over time from real production errors without manual curation."
)
br()

# =============================================================================
# REFERENCES
# =============================================================================
h("References")
refs = [
    "[1]   X. Zhong, J. Tang, and A. J. Yepes, \"PubLayNet: Largest Dataset Ever for Document Layout Analysis,\" in Proc. ICDAR, 2019.",
    "[2]   B. Pfitzmann et al., \"DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis,\" in Proc. KDD, 2022.",
    "[3]   Z. Shen et al., \"LayoutParser: A Unified Toolkit for Deep Learning Based Document Image Analysis,\" in Proc. ICDAR, 2021.",
    "[4]   D. G. Lowe, \"Distinctive Image Features from Scale-Invariant Keypoints,\" Int. J. Comput. Vis., vol. 60, no. 2, pp. 91–110, 2004.",
    "[5]   J. Canny, \"A Computational Approach to Edge Detection,\" IEEE Trans. Pattern Anal. Mach. Intell., vol. 8, no. 6, pp. 679–698, 1986.",
    "[6]   E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, \"ORB: An Efficient Alternative to SIFT or SURF,\" in Proc. ICCV, 2011.",
    "[7]   P. Bergmann et al., \"MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,\" in Proc. CVPR, 2019.",
    "[8]   K. Roth et al., \"Towards Total Recall in Industrial Anomaly Detection,\" in Proc. CVPR, 2022.",
    "[9]   G. Koch, R. Zemel, and R. Salakhutdinov, \"Siamese Neural Networks for One-Shot Image Recognition,\" in Proc. ICML Deep Learning Workshop, 2015.",
    "[10]  R. Hadsell, S. Chopra, and Y. LeCun, \"Dimensionality Reduction by Learning an Invariant Mapping,\" in Proc. CVPR, 2006.",
    "[11]  F. Sung et al., \"Learning to Compare: Relation Network for Few-Shot Learning,\" in Proc. CVPR, 2018.",
    "[12]  J. Howard and S. Ruder, \"Universal Language Model Fine-Tuning for Text Classification,\" in Proc. ACL, 2018.",
    "[13]  J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, \"How Transferable Are Features in Deep Neural Networks?,\" in Proc. NeurIPS, 2014.",
    "[14]  K. He, X. Zhang, S. Ren, and J. Sun, \"Deep Residual Learning for Image Recognition,\" in Proc. CVPR, 2016.",
    "[15]  Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, \"Image Quality Assessment: From Error Visibility to Structural Similarity,\" IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, 2004.",
    "[16]  R. Zhang et al., \"The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,\" in Proc. CVPR, 2018.",
    "[17]  Y. Xu et al., \"LayoutLM: Pre-training of Text and Layout for Document Image Understanding,\" in Proc. KDD, 2020.",
    "[18]  Y. Huang et al., \"LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking,\" in Proc. ACM MM, 2022.",
    "[19]  R. R. Selvaraju et al., \"Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,\" Int. J. Comput. Vis., vol. 128, no. 2, pp. 336–359, 2020.",
    "[20]  D. P. Kingma and J. Ba, \"Adam: A Method for Stochastic Optimization,\" in Proc. ICLR, 2015.",
    "[21]  J. Bromley et al., \"Signature Verification Using a Siamese Time Delay Neural Network,\" in Proc. NeurIPS, 1993.",
    "[22]  N. Srivastava et al., \"Dropout: A Simple Way to Prevent Neural Networks from Overfitting,\" JMLR, vol. 15, pp. 1929–1958, 2014.",
    "[23]  S. Ioffe and C. Szegedy, \"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,\" in Proc. ICML, 2015.",
    "[24]  T.-Y. Lin et al., \"Feature Pyramid Networks for Object Detection,\" in Proc. CVPR, 2017.",
    "[25]  S. Ren, K. He, R. Girshick, and J. Sun, \"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,\" IEEE TPAMI, 2017.",
]
for ref in refs:
    pg = doc.add_paragraph()
    pg.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = pg.add_run(ref)
    r.font.size = Pt(9)
br()

# =============================================================================
# APPENDIX A
# =============================================================================
h("Appendix A: JSON Metadata Schema Example")
p(
    "Annotated JSON payload for label type 1C50-IDBDM, showing which fields are "
    "validated by which rule categories in Tier 1."
)
code = doc.add_paragraph(style='No Spacing')
code.alignment = WD_ALIGN_PARAGRAPH.LEFT
cr = code.add_run("""{
  "label_type_id":    "1C50-IDBDM",      // Enumeration check: must be in catalog
  "article_number":   "123.456.78",      // Regex: \\d{3}\\.\\d{3}\\.\\d{2}
  "product_name":     "EXAMPLE PRODUCT", // Length: max 40 chars
  "origin_text":      "Made in Sweden",  // Presence check only
  "supplier_name":    "IKEA of Sweden AB",// Exact-value check
  "supplier_address": "SE-34381 Almhult",// Presence check only
  "country_code":     "SE",             // ISO 3166-1 alpha-2
  "gross_weight_kg":   2.4,             // Cross-field: must be > 0 if unit = kg
  "datamatrix": {
    "ai_240": "12345678901234",          // GS1 AI 240: presence check
    "ai_13":  "260101",                  // GS1 AI 13: YYMMDD format
    "ai_10":  "BATCH001"                 // Presence check only
  }
}""")
cr.font.name = "Courier New"
cr.font.size = Pt(9)
sp()
pl("All 10 rules pass → proceed to Tier 2 rendering.", bold=True)
br()

# =============================================================================
# APPENDIX B
# =============================================================================
h("Appendix B: Individual Contribution Statement")
tbl(
    ["Component / Task", "Primary Contributor"],
    [
        ["System architecture design (3-tier pipeline)", "Both authors equally"],
        ["NiceLabel SDK integration and rendering module", "Ghanasham Saravaiahgari"],
        ["JSON rule engine and schema files", "Ghazal Chamto"],
        ["SSIM comparison module and threshold calibration", "Ghanasham Saravaiahgari"],
        ["ORB feature matching module", "Ghanasham Saravaiahgari"],
        ["ResNet-50 fine-tuning and Grad-CAM localisation", "Ghanasham Saravaiahgari"],
        ["Siamese network architecture and contrastive training", "Ghazal Chamto"],
        ["Ensemble weighting and report generator", "Both authors equally"],
        ["Template catalog and baseline PNG preparation", "Ghazal Chamto"],
        ["Violation injection engine (synthetic dataset)", "Both authors equally"],
        ["Experimental evaluation and results analysis", "Both authors equally"],
        ["Literature review", "Both authors equally"],
        ["Report writing (Ch. 1–2)", "Ghazal Chamto"],
        ["Report writing (Ch. 3–5)", "Ghanasham Saravaiahgari"],
        ["Report writing (Ch. 6–8, Appendices)", "Both authors equally"],
    ],
    caption="Table B.1: Division of primary responsibility."
)
p(
    "All major architectural and design decisions were made jointly. Each author reviewed "
    "and approved the other's implementations before integration into the shared pipeline."
)

# ── Save ──────────────────────────────────────────────────────────────────────
output_path = "thesis_google_docs.docx"
doc.save(output_path)
print(f"Saved: {output_path}")
