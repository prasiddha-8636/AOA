import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "references", "pdfs")
os.makedirs(OUT, exist_ok=True)
NS = {"a": "http://www.w3.org/2005/Atom"}

REFS = [
    ("vaswani2017", "Attention is all you need"),
    ("touvron2023", "Llama: Open and efficient foundation language models"),
    (
        "devlin2019",
        "BERT: Pre-training of deep bidirectional transformers for language understanding",
    ),
    (
        "press2022",
        "Train short, test long: Attention with linear biases enables input length extrapolation",
    ),
    ("shaw2018", "Self-attention with relative position representations"),
    (
        "raffel2020",
        "Exploring the limits of transfer learning with a unified text-to-text transformer",
    ),
    (
        "dai2019",
        "Transformer-XL: Attentive language models beyond a fixed-length context",
    ),
    ("su2024roformer", "RoFormer: Enhanced Transformer with Rotary Position Embedding"),
    (
        "chen2023",
        "Extending context window of large language models via positional interpolation",
    ),
    (
        "peng2023yarn",
        "YaRN: Efficient Context Window Extension of Large Language Models",
    ),
    ("liu2024scaling", "Scaling Laws of RoPE-based Extrapolation"),
    ("sun2022lex", "A Length-Extrapolatable Transformer"),
    (
        "kazemnejad2023",
        "The Impact of Positional Encoding on Length Generalization in Transformers",
    ),
    (
        "ruoss2023",
        "Randomized Positional Encodings Boost Length Generalization of Transformers",
    ),
    (
        "chi2022",
        "KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation",
    ),
    (
        "zhao2024survey",
        "Length Extrapolation of Transformers: A Survey from the Perspective of Positional Encoding",
    ),
    ("veisi2025cable", "Context-aware Biases for Length Extrapolation"),
    ("shang2025longrope2", "LongRoPE2: Near-Lossless LLM Context Window Scaling"),
    (
        "he2024bipe",
        "Two Stones Hit One Bird: Bilevel Positional Encoding for Better Length Extrapolation",
    ),
]


def arxiv_id_for_title(title):
    q = urllib.parse.quote(f'ti:"{title}"')
    url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=3"
    req = urllib.request.Request(url, headers={"User-Agent": "ref-download/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    root = ET.fromstring(data)
    out = []
    for e in root.findall("a:entry", NS):
        aid = e.find("a:id", NS).text.strip()
        etitle = e.find("a:title", NS).text.strip().replace("\n", " ")
        out.append((aid, etitle))
    return out


def download(url, dest):
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ref-download/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if data[:5] == b"%PDF-":
        with open(dest, "wb") as f:
            f.write(data)
        return True
    return False


for key, title in REFS:
    dest = os.path.join(OUT, f"{key}.pdf")
    if os.path.exists(dest):
        print(f"{key}: already exists")
        continue
    try:
        results = arxiv_id_for_title(title)
        ok = False
        for aid, etitle in results:
            pdf_url = (
                aid.replace("http://arxiv.org/abs/", "https://arxiv.org/pdf/") + ".pdf"
            )
            if download(pdf_url, dest):
                print(f"{key}: OK from {aid} | {etitle[:70]}")
                ok = True
                break
        if not ok:
            print(f"{key}: FAIL no PDF from arxiv results: {results}")
    except Exception as ex:
        print(f"{key}: ERROR {ex}")
    time.sleep(3)
