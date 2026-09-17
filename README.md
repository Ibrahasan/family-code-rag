# Family Code RAG — Telegram Bot

A document-grounded question-answering Telegram bot built on the **Family Code of the Republic of Azerbaijan** (source: [e-qanun.az/framework/46946](https://e-qanun.az/framework/46946)).

The bot uses only information from the legal text and does not make up answers when the answer is not found in the context — it responds with "No information on this was found in the document."

---

## 1. Important Note: This Repository Is Delivered in a Ready-to-Run State

All pipeline stages (scraping → parsing → chunking → embedding) have **already been executed**, and the results are included in the repository:

- `data/raw/` — scraped raw text
- `data/processed/` — `articles.json`, `court_decisions.json`, `amendments.json`, `chunks.json`, `article_full_texts.json`
- `chroma_db/` — **pre-populated** vector database (including embeddings)

**Primary scenario for the reviewer:** clone the repository, add the API keys to `.env`, run `python src/bot.py` directly, and ask a question on Telegram. There is **no need** to rerun the pipeline from scratch (starting from scraping).

---

## 2. Architecture

```
Question → [Retrieval: clause-level search in ChromaDB]
     → [Small-to-Big: retrieved clauses are expanded to their full articles]
     → [Generation: Gemini LLM synthesizes an answer from the full article texts]
     → Answer
```

| Stage | File | Input → Output | For Reviewer? |
|---|---|---|---|
| 1a. Scraping | `src/01_scrape.py` | e-qanun.az → `data/raw/aile_mecellesi.txt` | Optional (result already exists) |
| 1b. Parsing | `src/02_parser.py` | raw text → 3 JSON files | Optional (result already exists) |
| 2. Chunking | `src/03_chunking.py` | 3 JSON files → `chunks.json` + `article_full_texts.json` | Optional (result already exists) |
| 3. Embedding | `src/04_build_vector_db.py` | `chunks.json` → `chroma_db/` | Optional (result already exists) |
| 3. Retrieval + Generation | `src/rag_engine.py` | question → answer (`answer_question()`) | — |
| **4. Deployment** | **`src/bot.py`** | **Telegram interface** | **✅ Main entry point** |

---

## 3. Project Structure

```
aile_mecellesi_rag/
│
├── data/
│   ├── raw/                    # Raw scraped text (aile_mecellesi.txt)
│   └── processed/              # articles.json, court_decisions.json, amendments.json,
│                                # chunks.json, article_full_texts.json  (ALL READY)
│
├── chroma_db/                  # ChromaDB vector database — INCLUDED in populated state
│
├── src/
│   ├── __init__.py
│   ├── 01_scrape.py            # Text extraction from e-qanun.az using Selenium
│   ├── 02_parser.py            # Structuring raw text into 4 JSON files
│   ├── 03_chunking.py          # Creating chunks.json and article_full_texts.json
│   ├── 04_build_vector_db.py   # Generating embeddings + writing to ChromaDB
│   ├── models.py                # Model names/configuration (EMBED_MODEL, GENERATION_MODEL, etc.)
│   ├── rag_engine.py           # Retrieval + Small-to-Big + Generation logic
│   └── bot.py                  # Telegram bot (aiogram)
│
├── .env.example
├── .gitignore                  
├── requirements.txt
└── README.md
```

---

## 4. Quick Start (Primary Scenario)

### Requirements
- Python 3.10+
- Google Gemini API key ([aistudio.google.com](https://aistudio.google.com))
- Telegram Bot Token (created via [@BotFather](https://t.me/BotFather))

> Microsoft Edge browser is **not required** — it is only needed if you want to rerun scraping (see Section 5). It is not part of this quick-start scenario.

### Steps

```bash
git clone https://github.com/Ibrahasan/family-code-rag.git
cd aile_mecellesi_rag

python -m venv venv
source venv/bin/activate   #Linux

pip install -r requirements.txt

cp .env.example .env
# Open `.env` and enter the GEMINI_API_KEY and TELEGRAM_BOT_TOKEN values

python src/bot.py
```

Once the bot is running, find the bot on Telegram, send `/start`, and then send your question. That's it — no additional steps are required because `chroma_db/` is already populated.

### `.env` Variables

| Variable | Required? | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Used for embedding and answer generation (`rag_engine.py` uses it at runtime) |
| `TELEGRAM_BOT_TOKEN` | Yes | Bot token obtained from @BotFather |

---

## 5. Reproducing the Full Pipeline from Scratch (Optional)

If you want to verify the code itself (not just the results), you can run all stages in sequence. All commands must be run from the **project root directory**.

**Additional requirement:** Microsoft Edge must be installed for `01_scrape.py` (Selenium + Edge WebDriver are used).

```bash
# Stage 1a — Scraping
python src/01_scrape.py

# Stage 1b — Parsing
python src/02_parser.py

# Stage 2 — Chunking
python src/03_chunking.py

# Stage 3 — Embedding generation and writing to ChromaDB
python src/04_build_vector_db.py

# Local test (in the terminal, without connecting to Telegram)
python src/rag_engine.py

# Stage 4 — Start the Telegram bot
python src/bot.py
```

> ⚠️ If you change `EMBED_DIM` in `models.py`, delete the existing `chroma_db/` directory and rerun `04_build_vector_db.py` — otherwise, you will get a dimension mismatch error between the old (different-sized) vectors and new queries.

---

## 6. Chunking Strategy

Instead of a **fixed-size window**, a **document-structure-based segmentation** approach was chosen because article/clause boundaries are already semantically meaningful units in legal text.

- **Segmentation unit:** article, and when necessary, down to the clause level.
- **Threshold:** `WORD_THRESHOLD = 300` words. This was empirically selected based on word-count statistics (median/75th/90th/95th percentiles) produced when running `03_chunking.py`; it is not arbitrary.
- **Rule:**
  - Articles shorter than 300 words → kept as a single chunk.
  - Articles longer than 300 words → split **ONLY at the FIRST LEVEL** clause boundaries (e.g. 118.1, 118.2, ... 118.24). Second-level subclauses (such as 118.1.1) are not separated from their parent clause — segmentation is not made excessively granular.
  - Long articles where no clause format is detected are kept intact with the marker `note: "UZUN AMMA BÖLÜNMƏDİ"` (to avoid information loss), for manual review.
- **Overlap:** not used — in structure-based segmentation, each clause is already a complete semantic unit, so artificial overlap does not add value.
- **Header injection:** the prefix "Article N. Title" is added to the beginning of every chunk so that the chunk remains meaningful even when retrieved out of context.
- **CC Court Decisions (Constitutional Court decisions) and Amendments** are stored as separate types, with `related_articles` metadata indicating which articles they relate to.

---

## 7. Embedding

- **Model:** `gemini-embedding-2-preview` (Google), 768-dimensional (`EMBED_DIM=768`, selected as a balance between speed and quality — can be increased up to 3072).
- **`task_type` distinction:** `RETRIEVAL_DOCUMENT` during indexing and `RETRIEVAL_QUERY` during querying — the approach recommended by Gemini, improving retrieval quality.
- **Batching is not used** — `gemini-embedding-2-preview` has a known bug: when a list is passed to `contents`, it returns only 1 embedding. Therefore, each chunk is embedded with a separate request.
- Embeddings are stored in the `family_code_az` collection in ChromaDB using cosine similarity.

---

## 8. Retrieval Strategy: Small-to-Big

Instead of a simple "retrieve and send" approach, the **Parent Document Retrieval (Small-to-Big)** strategy is used:

1. **Search is performed at the precise (small) level** — over clause-level chunks in ChromaDB, because smaller, focused texts produce more precise similarity results.
2. **Context is provided to the LLM at the broader (large) level** — the **full article text** corresponding to each retrieved clause is looked up from `article_full_texts.json` and sent to the LLM so the model can answer with more complete legal context.
3. **Dedup:** if multiple clauses from the same article are retrieved (e.g. 118.1 and 118.3), the full text of that article is sent only **once** — preventing unnecessary token usage.
4. **Fallback chain:** if `article_full_texts.json` is not found → the system automatically falls back to the previous (small-chunk-only) mode instead of crashing. If a specific article is not found in the lookup → the chunk's own text is used.
5. `k=5` (number of clauses retrieved initially) — intentionally kept low to control the token budget because results are expanded to full articles.

---

## 9. Answer Generation (LLM)

- **Model:** `gemini-3.5-flash-lite` — on the free tier (for the current account), with a 500 RPD / 15 RPM limit, which is sufficient for the project's testing and demo requirements.
- **`temperature=0.2`** — deliberately kept low to reduce the chance of the model going beyond the provided context and producing "creative" answers (hallucinations).
- **System prompt rules:**
  1. Use only the provided CONTEXT.
  2. If the answer is not in the context, do not make it up — say "No direct information on this was found in the document."
  3. The answer should be in Azerbaijani, short, and clear.
  4. When possible, mention which article the answer is based on.

---

## 10. Handling Azerbaijani Language Variations

Users may write questions using informal transliteration (`sh`→ş, `ch`→ç, `w`→v, etc.), as well as spelling mistakes.

**Chosen approach:** instead of implementing a separate normalization layer, the project relies on the **multilingual `gemini-embedding-2-preview` model** to already capture these variations at the semantic level. This decision was tested:

- Test question: *"Menim həyat yoldashımla yetkinlik yashına çatmayan ümumi uşagımız yoxdur və ikimiz də boşanmağa razıyıq..."* (qəsdən orfoqrafik səhvlərlə: "yoldashım", "yashına", "uşagımız", "bowanmaq", "müraciet", "nigaha").
- This question was sent both with correct spelling and in transliterated form; in both cases, **the same chunks were retrieved and the resulting answer did not differ in content**.
- Validation method: the `chunk_id` lists returned by `retrieve_context()` were compared between the two versions 
- This decision was made after testing it on a sufficient number of examples.


> Note: this is not a replacement for a fully formal normalization layer (case-folding, character-replacement dictionary), but it is a tested decision that is sufficient for the scale of this project.

---

## 11. Tested Example Questions

| Sual | Cavab |
|---|---|
| "Nikaha daxil olmaq üçün tibbi müayinə lazımdırmı?" | Azərbaycan Respublikasının Ailə Məcəlləsinə əsasən, nikaha daxil olmaq üçün tibbi müayinədən keçmək zəruridir. 

Bu hüquqi tələb aşağıdakı maddələrlə tənzimlənir:

* **Məcəllənin 11.1-ci maddəsinə əsasən**, nikahın bağlanması üçün nikaha daxil olan şəxslərin yazılı razılığı, nikah yaşına çatmaları və **13-cü maddəyə uyğun olaraq tibbi müayinədən keçmələri zəruridir**.
* **Məcəllənin 13.1-ci maddəsinə əsasən**, nikaha daxil olmaq istəyən şəxslər siyahısı müvafiq icra hakimiyyəti orqanı tərəfindən müəyyən edilən xəstəliklər üzrə tibbi müayinədən keçirlər. |
| "şikəstlik” sözünün “xəsarət” sözü ilə əvəz edilməsi hansı qanunla heyata keçirilib" | Təqdim olunan kontekstə əsasən, "şikəstlik" sözünün "xəsarət" sözü ilə əvəz edilməsi 8 iyul 2022-ci il tarixli 581-VIQD nömrəli Azərbaycan Respublikasının Qanunu ilə həyata keçirilmişdir (**Maddə 32.2.1-ə əsasən**). |
| Sənədə aid olmayan sual (məs. "Bakının əhalisi neçədir?") | "Təqdim olunan sənədlərdə bu suala birbaşa məlumat tapılmadı." |

---

## 12. Error Handling (Telegram Bot)

- **Empty/whitespace-only message:** checked before the RAG engine is called, and a warning is sent to the user.
- **Non-text messages** (image, voice, document): handled by a separate handler, informing the user that only text messages are accepted.
- **API/network errors:** caught with `try/except`; the user sees a generic "technical issue" message, while the actual error is written to the server log.
- **Non-blocking execution:** because `rag.answer_question()` is a synchronous (blocking) call, it is run in a separate thread using `asyncio.to_thread()` so that other bot users are not blocked at the same time.

---

## 13. License / Source

The text content was obtained from e-qanun.az, the official legal portal of the Republic of Azerbaijan, and is public legal information.
