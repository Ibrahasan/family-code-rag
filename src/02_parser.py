import re
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RAW_DATA_DIR / "aile_mecellesi.txt"

def load_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_into_sections(text):
    
    markers = [
        "İSTİFADƏ OLUNMUŞ MƏNBƏ SƏNƏDLƏRİNİN SİYAHISI",
        "Konstitusiya Məhkəməsinin Qərarları",
        "MƏCƏLLƏYƏ EDİLMİŞ DƏYİŞİKLİK VƏ ƏLAVƏLƏRİN SİYAHISI",
    ]
    positions = sorted((text.find(m), m) for m in markers if text.find(m) != -1)
    cuts = [0] + [p[0] for p in positions] + [len(text)]
    labels = ["ƏSAS_MƏTN"] + [p[1] for p in positions]
    return {label: text[cuts[i]:cuts[i + 1]].strip() for i, label in enumerate(labels)}


ARTICLE_REF = re.compile(r"(\d+(?:\.\d+)*(?:-\d+)?)-c[iüıu]\s+maddə")
CODE_MARKERS = ["Ailə Məcəlləsinin", "Konstitusiyasının"]
OUR_CODE = "Ailə Məcəlləsinin"


def find_related_articles(chunk_text):
    """
    Returns numbers belonging to the format "Article ... of the Family Code".

    Args:
        chunk_text (str): The text segment to analyze.

    Returns:
        list[str]: A sorted list of unique article numbers.
    """
    marker_hits = []
    for marker in CODE_MARKERS:
        for mm in re.finditer(re.escape(marker), chunk_text):
            marker_hits.append((mm.start(), marker))
    marker_hits.sort()

    result = []
    for am in ARTICLE_REF.finditer(chunk_text):
        preceding = [mh for mh in marker_hits if mh[0] < am.start()]
        owner = preceding[-1][1] if preceding else OUR_CODE
        if owner == OUR_CODE:
            result.append(am.group(1))
    return sorted(set(result))


def parse_articles(main_text):
    """
    Splits the main text into individual articles based on the 'Maddə N.' format.

    Args:
        main_text (str): The main text of the document to parse.

    Returns:
        list[dict]: A list of dictionaries containing article details and full text.
    """
    pattern = re.compile(r"Maddə\s+(\d+(?:-\d+)?)\.\s*(.+)")
    matches = list(pattern.finditer(main_text))

    articles = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(main_text)
        articles.append({
            "type": "article",
            "madde_no": m.group(1),
            "basliq": m.group(2).strip(),
            "full_text": main_text[start:end].strip()
        })
    return articles


def parse_kmq(kmq_text):
    """
    Splits Constitutional Court Decisions based on the 'KMQ N' format.

    Args:
        kmq_text (str): The text section containing court decisions to parse.

    Returns:
        list[dict]: A list of dictionaries containing decision details and related articles.
    """
    pattern = re.compile(r"KMQ(\d+)\s+(.+?)(?=KMQ\d+|$)", re.DOTALL)
    records = []
    for m in pattern.finditer(kmq_text):
        entry_text = m.group(2).strip()
        records.append({
            "type": "court_decision",
            "kmq_no": m.group(1),
            "related_articles": find_related_articles(entry_text),
            "full_text": entry_text
        })
    return records


def parse_amendments(amend_text):
    """
    Splits amendments based on the '[N]' format.

    Args:
        amend_text (str): The text section containing amendments to parse.

    Returns:
        list[dict]: A list of dictionaries containing amendment details and related articles.
    """
    pattern = re.compile(r"\[(\d+)\]\s+(.+?)(?=\[\d+\]|$)", re.DOTALL)
    records = []
    for m in pattern.finditer(amend_text):
        entry_text = m.group(2).strip()
        records.append({
            "type": "amendment",
            "amendment_no": m.group(1),
            "related_articles": find_related_articles(entry_text),
            "full_text": entry_text
        })
    return records


def parse_source_documents(source_text):
    """
    Splits the list of source documents into simple paragraphs.

    Args:
        source_text (str): The text section containing source documents.

    Returns:
        list[dict]: A list of dictionaries containing the full text of each source document.
    """
    items = [p.strip() for p in re.split(r"\n(?=\d+\.\s)", source_text) if p.strip()]
    return [{"type": "source_document", "full_text": it} for it in items]


def main():
    text = load_text(INPUT_FILE)
    sections = split_into_sections(text)
    
    articles = parse_articles(sections.get("ƏSAS_MƏTN", ""))
    kmq_records = parse_kmq(sections.get("Konstitusiya Məhkəməsinin Qərarları", ""))
    amendment_records = parse_amendments(
        sections.get("MƏCƏLLƏYƏ EDİLMİŞ DƏYİŞİKLİK VƏ ƏLAVƏLƏRİN SİYAHISI", "")
    )
    source_records = parse_source_documents(
        sections.get("İSTİFADƏ OLUNMUŞ MƏNBƏ SƏNƏDLƏRİNİN SİYAHISI", "")
    )

    with open(PROCESSED_DATA_DIR / "articles.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
        
    with open(PROCESSED_DATA_DIR / "court_decisions.json", "w", encoding="utf-8") as f:
        json.dump(kmq_records, f, ensure_ascii=False, indent=2)
        
    with open(PROCESSED_DATA_DIR / "amendments.json", "w", encoding="utf-8") as f:
        json.dump(amendment_records, f, ensure_ascii=False, indent=2)
        
    with open(PROCESSED_DATA_DIR / "source_documents.json", "w", encoding="utf-8") as f:
        json.dump(source_records, f, ensure_ascii=False, indent=2)

    print("=== SUMMARY ===")
    print(f"Articles:            {len(articles)}")
    print(f"Court Decisions:     {len(kmq_records)}")
    print(f"Amendments:          {len(amendment_records)}")
    print(f"Source Documents:    {len(source_records)}")
    print(f"\nResults successfully written to '{PROCESSED_DATA_DIR}' directory.")

    if len(articles) < 5:
        print("\n⚠️  WARNING: The number of parsed articles is unexpectedly low. "
              "Check if the 'Maddə N.' format differs in the actual document "
              "(e.g., uppercase letters, different spacing).")

if __name__ == "__main__":
    main()