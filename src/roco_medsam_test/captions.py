"""Rule-based caption parsing (TRAINING TIME ONLY).

Extracts: imaging modality, a localizable anomaly phrase, and whether the
caption is usable at all. Uses word-boundary regexes (substring matching would
make "ct" match "lectin", "pet" match "petechiae", etc.).
"""
import re

# (name, regex). A caption mentioning two different modalities is rejected.
MODALITY_PATTERNS = [
    ("PET/CT", r"\bpet[\s/\-]?ct\b"),
    ("PET", r"\bpet\b|positron emission"),
    ("CT", r"\bct\b|\bcect\b|\bcta\b|computed tomograph"),
    ("MRI", r"\bmri\b|\bmr\b|magnetic resonance|\bflair\b|\bdwi\b|\bt1[- ]?weighted|\bt2[- ]?weighted"),
    ("ultrasound", r"ultraso|sonograph|echocardiogra|\bdoppler\b"),
    ("angiography", r"angiogra|\bdsa\b"),
    ("X-ray", r"x-?ray|radiograph|\bcxr\b|chest film|mammogra|orthopantomogra|panoramic"),
]
_MOD = [(n, re.compile(p)) for n, p in MODALITY_PATTERNS]

NON_RADIOLOGY = re.compile(
    r"histolog|microscop|h&e|hematoxylin|stain|photograph|specimen|gross |intraoperative|"
    r"clinical photo|endoscop|fundus|dermoscop|schematic|diagram|flow ?chart|\bgraph\b|"
    r"\bdrawing\b|illustration|colonoscop|laparoscop|arthroscop|\boct\b|\becg\b|\bekg\b"
)
MULTI_PANEL = re.compile(r"\(\s?[a-hA-H]\s?\)|(?<![\w])[a-hA-H]\)|\bpanels?\b|\brespectively\b")
MARKED = re.compile(r"\barrow|asterisk|circle|star\b|dotted|outlined|encircled")
# 'normal' deliberately NOT here: "normal liver with a hypodense lesion" still has a lesion
NEG = re.compile(r"\b(no|without|absence of|negative for|ruled out|free of|resolved|resolution of)\b")
NORMAL = re.compile(r"\b(normal|unremarkable)\b")

FINDING_STEMS = [
    r"nodules?", r"masses|mass", r"lesions?", r"tumou?rs?", r"opacit(?:y|ies)", r"consolidations?",
    r"(?:pleural )?effusions?", r"fractures?", r"cysts?", r"h(?:a)?emorrhages?", r"infarcts?",
    r"infarctions?", r"thromb(?:us|i|osis)", r"aneurysms?", r"stenos(?:is|es)", r"abscess(?:es)?",
    r"calcifications?", r"o?edema", r"pneumothora(?:x|ces)", r"metasta(?:sis|ses)", r"thickening",
    r"enlargement", r"h(?:a)?ematomas?", r"polyps?", r"ulcers?", r"dilat(?:ation|ion)",
    r"hyper(?:intense|dense|echoic|vascular)", r"hypo(?:intense|dense|echoic|vascular)",
    r"enhancement", r"filling defects?", r"cavit(?:y|ies)", r"cavitation", r"emphysema",
    r"fibrosis", r"atelectasis", r"cardiomegaly", r"hydronephrosis", r"calcul(?:us|i)", r"stones?",
    r"occlusion", r"hernia", r"herniation", r"diverticul(?:um|a)", r"osteolytic", r"sclerotic",
    r"ground[- ]glass", r"pneumonia", r"carcinoma", r"adenoma", r"neoplasm", r"h(?:a)?emangioma",
    r"osteophytes?", r"protrusion", r"stricture", r"foreign bod(?:y|ies)",
    r"defects?", r"deformit(?:y|ies)", r"dissection", r"lymphadenopathy",
    r"goit(?:er|re)", r"fistula", r"osteomyelitis", r"sarcoma", r"lymphoma", r"meningioma",
]
_TERM = re.compile(r"\b(" + "|".join(sorted(FINDING_STEMS, key=len, reverse=True)) + r")\b")
_TAIL = re.compile(
    r"(?:\s+(?:is|are|was|were)\s+(?:seen|noted|visible|present|located|identified|demonstrated))?"
    r"\s+(?:in|of|at|within|involving|near|adjacent to|around|along)\s+(?:the\s+)?(?:[a-z\-]+\s?){1,3}"
)
_LEAD_STOP = {
    "the", "a", "an", "of", "with", "and", "showing", "shows", "show", "demonstrates", "demonstrating",
    "revealing", "reveals", "revealed", "there", "is", "are", "was", "were", "in", "on", "at", "by",
    "to", "from", "image", "images", "scan", "axial", "coronal", "sagittal", "view", "ct", "mri",
    "x-ray", "xray", "radiograph", "also", "that", "which", "large", "multiple", "patient", "his",
    "her", "case", "year", "old", "arrow", "arrows", "arrowhead", "or", "for", "as", "seen", "note",
    "ultrasound", "sonography", "doppler", "angiography", "angiogram", "pet", "weighted", "t1", "t2",
    "flair", "dwi", "contrast", "enhanced", "post", "pre", "mr", "magnetic", "resonance", "computed",
    "tomography", "positron", "emission", "chest", "abdominal", "showed", "demonstrated",
}
_PREPS = {"in", "of", "at", "within", "involving", "near", "adjacent", "around", "along"}
_TAIL_STOP = {"and", "with", "which", "that", "is", "was", "are", "were", "showing", "shows", "or",
              "on", "by", "to", "from", "after", "before", "during", "for"}


_ARROW_COLORS = {
    "colored": r"red|yellow|green|blue|orange|pink|purple|cyan|magenta",
    "white": r"white",
    "black": r"black",
}


def arrow_hints(caption: str):
    """What the caption says about its arrows: dict(colors, max_n).
    colors: channels named right before 'arrow(head)', e.g. "red arrow" -> {"colored"}; None = unknown.
    max_n: how many arrows to keep (singular 'arrow' -> 2, plural or several colours -> 6)."""
    low = caption.lower() if isinstance(caption, str) else ""
    colors = {c for c, p in _ARROW_COLORS.items()
              if re.search(r"\b(?:" + p + r")\b[^.;,()]{0,25}\barrow", low)}
    plural = bool(re.search(r"\barrow(?:head)?s\b", low)) or len(re.findall(r"\barrow", low)) > 1
    return {"colors": colors or None, "max_n": 6 if plural or len(colors) > 1 else 2}


def detect_modality(caption: str):
    """Single modality or None. Captions that mention several modalities are ambiguous -> None."""
    low = caption.lower()
    found = {n for n, p in _MOD if p.search(low)}
    if not found:
        return None
    if "PET/CT" in found:
        found -= {"PET", "CT"}
    if "angiography" in found and len(found) == 2 and ({"CT", "MRI"} & found):
        found.discard("angiography")          # "CT angiography" is CT
    return next(iter(found)) if len(found) == 1 else None


def _clause_before(low: str, pos: int, n: int, delims: str) -> str:
    """Up to n chars before pos, cut at the last clause delimiter (so negation/lead words
    from the previous sentence do not leak into this finding)."""
    return re.split(delims, low[max(0, pos - n):pos])[-1]


def extract_finding(caption: str):
    """dict(phrase, term, n_terms) for the first non-negated finding, else None.
    n_terms = number of distinct non-negated finding terms (>1 means the caption is
    about several findings and one mask cannot represent it)."""
    low = re.sub(r"\s+", " ", caption.lower())
    terms, first = set(), None
    for m in _TERM.finditer(low):
        if NEG.search(_clause_before(low, m.start(), 40, r"[.;:()]")):
            continue
        # adjective such as "hyperintense" directly followed by a noun finding: let the noun win
        if re.match(r"hyper|hypo", m.group(0)) and _TERM.match(low[m.end():].lstrip()):
            continue
        terms.add(m.group(0))
        if first is not None:
            continue
        lead_ctx = _clause_before(low, m.start(), 40, r"[.;:,()]")
        lead = [w for w in re.findall(r"[a-z][a-z\-]*", lead_ctx)[-3:] if w not in _LEAD_STOP]
        tail_words = []
        tm = _TAIL.match(low[m.end():])
        if tm:
            words = re.findall(r"[a-z\-]+", tm.group(0))
            while words and words[0] not in _PREPS:      # drop the optional "is seen" prefix
                words.pop(0)
            tail_words.append(words[0])  # the preposition
            for w in words[1:]:
                if w in _TAIL_STOP:
                    break
                tail_words.append(w)
            if len(tail_words) == 1:
                tail_words = []
        first = (" ".join(lead + [m.group(0)] + tail_words).strip(), m.group(0))
    if first is None:
        return None
    return {"phrase": first[0], "term": first[1], "n_terms": len(terms)}


def classify(caption):
    """None if the caption is unusable, else dict(modality, kind, phrase, term, n_terms, marked).
    kind is 'finding' or 'normal'."""
    if not isinstance(caption, str) or len(caption) < 15:
        return None
    low = caption.lower()
    if NON_RADIOLOGY.search(low) or MULTI_PANEL.search(caption):
        return None
    modality = detect_modality(caption)
    if modality is None:
        return None
    marked = bool(MARKED.search(low))
    f = extract_finding(caption)
    if f is not None:
        return {"modality": modality, "kind": "finding", "phrase": f["phrase"],
                "term": f["term"], "n_terms": f["n_terms"], "marked": marked}
    if NORMAL.search(low):
        return {"modality": modality, "kind": "normal", "phrase": "", "term": "", "n_terms": 0,
                "marked": marked}
    return None
