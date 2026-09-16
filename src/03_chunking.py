import re
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_ARTICLES = BASE_DIR / "data" / "processed" / "articles.json"
INPUT_KMQ = BASE_DIR / "data" / "processed" / "court_decisions.json"
INPUT_AMENDMENTS = BASE_DIR / "data" / "processed" / "amendments.json"

OUTPUT_CHUNKS = BASE_DIR / "data" / "processed" / "chunks.json"
OUTPUT_PARENT_LOOKUP = BASE_DIR / "data" / "processed" / "article_full_texts.json"

WORD_THRESHOLD = 300

def word_count(text):
    """
    Calculates the total number of words in a given text.

    Args:
        text (str): The text string to be evaluated.

    Returns:
        int: The word count.
    """
    return len(text.split())


def split_into_first_level_bendler(full_text, madde_no):
    """
    Splits the full text strictly by first-level clause boundaries.
    
    For example, for article 118, it extracts 118.1, 118.2, etc. 
    Second-level sub-clauses (such as 118.1.1 or 118.3.2) are not split 
    and remain as part of their parent clause's text.

    Args:
        full_text (str): The complete text of the article.
        madde_no (str): The article number used as the base for splitting.

    Returns:
        list[dict] | None: A list of dictionaries containing clause numbers 
        and their corresponding texts, or None if no first-level clauses are found.
    """
    escaped_no = re.escape(madde_no)
    # Məntiq: "{madde_no}.{rəqəm}" formatından sonra yalnız bir boşluq
    # gəlirsə, bu birinci səviyyədir. Əgər ardınca daha bir rəqəm+nöqtə
    # gəlirsə (məs. 118.1.1), bu regex artıq uyğun gəlməyəcək, çünki
    # "\." dan sonra whitespace tələb edir, rəqəm yox.
    pattern = re.compile(rf'(?:^|\n)[ \t]*({escaped_no}\.\d+(?:-\d+)?)\.\s')
    matches = list(pattern.finditer(full_text))

    if not matches:
        return None

    segments = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        segments.append({
            "bend_no": m.group(1),
            "text": full_text[start:end].strip()
        })
    return segments


def build_chunks_from_articles(articles):
    """
    Processes articles to generate text chunks for embedding and builds a parent lookup dictionary.

    Articles shorter than the word threshold are kept intact. Longer articles are split 
    into first-level clauses. If a long article lacks the expected clause format, it is 
    kept whole with a warning note appended to its metadata.

    Args:
        articles (list[dict]): A list of parsed article dictionaries.

    Returns:
        tuple[list[dict], dict]: A tuple containing the list of generated chunks 
        and a dictionary mapping article numbers to their full texts.
    """
    chunks = []
    parent_lookup = {}

    for art in articles:
        madde_no = art["madde_no"]
        basliq = art["basliq"]
        full_text = art["full_text"]
        parent_lookup[madde_no] = full_text

        header = f"Maddə {madde_no}. {basliq}"
        wc = word_count(full_text)

        if wc <= WORD_THRESHOLD:
            chunks.append({
                "chunk_id": f"article_{madde_no}",
                "type": "article",
                "madde_no": madde_no,
                "bend_no": None,
                "basliq": basliq,
                "text": full_text,
                "parent_madde_no": madde_no
            })
            continue

        segments = split_into_first_level_bendler(full_text, madde_no)

        if not segments:
            # Uzundur, amma bənd formatı tapılmadı -- ehtiyat variant:
            # bütöv saxlanılır ki, məlumat itməsin. Bu halları sonra
            # əl ilə yoxlamaq lazımdır.
            chunks.append({
                "chunk_id": f"article_{madde_no}",
                "type": "article",
                "madde_no": madde_no,
                "bend_no": None,
                "basliq": basliq,
                "text": full_text,
                "parent_madde_no": madde_no,
                "note": "UZUN AMMA BÖLÜNMƏDİ -- əl ilə yoxla"
            })
            continue

        for seg in segments:
            chunk_text = f"{header}\n{seg['text']}"
            chunks.append({
                "chunk_id": f"article_{madde_no}_{seg['bend_no']}",
                "type": "article",
                "madde_no": madde_no,
                "bend_no": seg["bend_no"],
                "basliq": basliq,
                "text": chunk_text,
                "parent_madde_no": madde_no
            })

    return chunks, parent_lookup


def build_chunks_from_kmq(kmq_records):
    """
    Converts Constitutional Court decision records into embedding-ready chunks.

    Args:
        kmq_records (list[dict]): A list of parsed court decision records.

    Returns:
        list[dict]: A list of chunk dictionaries with appended headers and metadata.
    """

    chunks = []
    for rec in kmq_records:
        header = f"Konstitusiya Məhkəməsinin Qərarı (KMQ{rec['kmq_no']})"
        chunks.append({
            "chunk_id": f"kmq_{rec['kmq_no']}",
            "type": "court_decision",
            "kmq_no": rec["kmq_no"],
            "related_articles": rec.get("related_articles", []),
            "text": f"{header}\n{rec['full_text']}"
        })
    return chunks


def build_chunks_from_amendments(amendment_records):
    """
    Converts amendment records into embedding-ready chunks.

    Args:
        amendment_records (list[dict]): A list of parsed amendment records.

    Returns:
        list[dict]: A list of chunk dictionaries with appended headers and metadata.
    """

    chunks = []
    for rec in amendment_records:
        header = f"Dəyişiklik [{rec['amendment_no']}]"
        chunks.append({
            "chunk_id": f"amendment_{rec['amendment_no']}",
            "type": "amendment",
            "amendment_no": rec["amendment_no"],
            "related_articles": rec.get("related_articles", []),
            "text": f"{header}\n{rec['full_text']}"
        })
    return chunks


def print_length_report(articles):
    """
    Calculates and prints a statistical report on article word counts.
    This helps in determining the optimal word threshold for chunking.

    Args:
        articles (list[dict]): A list of parsed article dictionaries.

    Returns:
        None
    """
    counts = sorted(word_count(a["full_text"]) for a in articles)
    n = len(counts)

    def pct(p):
        idx = min(n - 1, int(n * p))
        return counts[idx]

    print("=== full_text word count analysis (for WORD_THRESHOLD selection) ===")
    print(f"Number of articles: {n}")
    print(f"Min: {counts[0]}   Max: {counts[-1]}")
    print(f"Median (50%): {pct(0.5)}   75%: {pct(0.75)}   90%: {pct(0.9)}   95%: {pct(0.95)}")
    for th in (150, 200, 250, 300, 400, 500):
        over = sum(1 for c in counts if c > th)
        print(f"  Articles longer than {th} words: {over} ({over / n * 100:.1f}%)")
    print()


def main():
    with open(INPUT_ARTICLES, encoding="utf-8") as f:
        articles = json.load(f)
    with open(INPUT_KMQ, encoding="utf-8") as f:
        kmq_records = json.load(f)
    with open(INPUT_AMENDMENTS, encoding="utf-8") as f:
        amendment_records = json.load(f)

    print_length_report(articles)

    article_chunks, parent_lookup = build_chunks_from_articles(articles)
    kmq_chunks = build_chunks_from_kmq(kmq_records)
    amendment_chunks = build_chunks_from_amendments(amendment_records)
    all_chunks = article_chunks + kmq_chunks + amendment_chunks

    with open(OUTPUT_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_PARENT_LOOKUP, "w", encoding="utf-8") as f:
        json.dump(parent_lookup, f, ensure_ascii=False, indent=2)

    split_count = sum(
        1 for a in articles if word_count(a["full_text"]) > WORD_THRESHOLD
    )
    unresolved = sum(1 for c in article_chunks if c.get("note"))

    print("=== SUMMARY ===")
    print(f"Chunks generated from articles: {len(article_chunks)}")
    print(f"Court decision (KMQ) chunks:    {len(kmq_chunks)}")
    print(f"Amendment chunks:               {len(amendment_chunks)}")
    print(f"TOTAL chunks:                   {len(all_chunks)}")
    print()
    print(f"{split_count} articles were split into clauses "
          f"(threshold: {WORD_THRESHOLD} words)")
    print(f"{len(articles) - split_count} articles remained intact")
    
    if unresolved:
        print(f"\n⚠️  WARNING: {unresolved} long articles could not be split "
              f"due to clause formatting (check the 'note' field in chunks.json, "
              f"manual review required).")

    print(f"\nResults saved to: {OUTPUT_CHUNKS}, {OUTPUT_PARENT_LOOKUP}")


if __name__ == "__main__":
    main()