import os
import json
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from google import genai
from google.genai import types

from models import EMBED_MODEL, EMBED_DIM, GENERATION_MODEL


class FamilyCodeRAGEngine:
    """
    RAG (Retrieval-Augmented Generation) Engine for the Azerbaijani Family Code.
    This class implements the "Small-to-Big" (Parent Document Retrieval) strategy:
    it retrieves highly granular clauses from ChromaDB for precision, but expands 
    them to their full parent articles before feeding them to the LLM for better context.
    """
    
#     SYSTEM_PROMPT = """Sən Azərbaycan Ailə Məcəlləsi üzrə sual-cavab köməkçisisən.
# Qaydalar:
# 1. YALNIZ aşağıda verilən KONTEKST bölməsindəki məlumatdan istifadə et.
# 2. Kontekstdə cavab yoxdursa, uydurma - "Sənəddə bu barədə birbaşa məlumat tapılmadı." de.
# 3. Cavabı Azərbaycan dilində, detallı,geniş və aydın ver.
# 4. Mümkünsə, hansı maddəyə əsaslandığını qeyd et (məs. "Maddə 13.2-yə görə...")."""

    SYSTEM_PROMPT = """Sən Azərbaycan Ailə Məcəlləsi üzrə ixtisaslaşmış, peşəkar və obyektiv Süni İntellekt hüquq məsləhətçisisən.
Sənin vəzifən istifadəçilərin suallarına YALNIZ sənə təqdim olunan KONTEKST əsasında dəqiq, əhatəli və anlaşıqlı cavab verməkdir.

QƏTİ QAYDALAR:
1. MÜTLƏQ İSTİNAD: Həmişə qanunun nə dediyini konkret maddə nömrəsini göstərərək əsaslandır (məsələn: **Maddə 13.2-yə əsasən...**). Əgər mənbə maddə deyilsə, ümumi qanun adını çək, amma "source_document", "chunk" kimi texniki adları QƏTİYYƏN cavabında istifadə etmə.
2. XƏYALİ MƏLUMAT YARATMA (NO HALLUCINATION): Yalnız və yalnız sənə verilən KONTEKST-dəki məlumatlardan istifadə et. Kontekstdən kənar heç bir hüquqi bilik və ya şəxsi fərziyyə uydurma.
3. BİLMƏDİYİNİ ETİRAF ET: Əgər sualın cavabı verilmiş kontekstdə yoxdursa, istifadəçini yanltmamaq üçün sadəcə bunu de: "Təqdim olunan sənədlərdə bu suala birbaşa məlumat tapılmadı."
4. PEŞƏKAR VƏ STRUKTURLU FORMAT: 
   - Cavabı Azərbaycan dilində, geniş, detallı və hüquqi dildə formalaşdır.
   - İstifadəçinin oxumasını asanlaşdırmaq üçün uzun mətnləri abzaslara böl.
   - Şərtlər və ya hallar sadalananda mütləq işarələnmiş siyahılardan (bullet points: -, *) istifadə et.
   - Əsas məqamları və hüquqi terminləri **qalın şriftlə** vurğula.
5. OBYEKTİVLİK: Sən qərar verən hakim və ya vəkil deyilsən. Şəxsi hüquqi məsləhət (məs. "belə etməyiniz məsləhətdir") vermə, yalnız qanunun tələblərini izah et."""

    def __init__(self) -> None:            
        """
        Initializes the RAG engine.
        Loads environment variables, establishes the GenAI client, connects to ChromaDB, 
        and loads the full article lookup dictionary into memory for Small-to-Big expansion.
        
        Raises:
            ValueError: If the GEMINI_API_KEY environment variable is missing.
        """
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY") 
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found! Please check your .env file.")
        
        self.client = genai.Client(api_key=api_key)
        
        base_dir = Path(__file__).resolve().parent.parent
        chroma_db_path = base_dir / "chroma_db"
        
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))
        self.collection = self.chroma_client.get_or_create_collection(
            name="family_code_az",
            metadata={"hnsw:space": "cosine"},
        )

        lookup_path = base_dir / "data" / "processed" /"article_full_texts.json"  
        try:
            with open(lookup_path, "r", encoding="utf-8") as f:
                self.article_lookup = json.load(f)
        except FileNotFoundError:
            print(f"WARNING: Full text lookup file not found at {lookup_path}.")
            print("The engine will fall back to using small chunk texts directly.")
            self.article_lookup = {}

    def retrieve_context(self, question: str, k: int = 4) -> list[dict]:
        """
        Retrieves the most relevant granular text chunks from ChromaDB.

        Args:
            question (str): The user's query to search for.
            k (int, optional): The number of top chunks to retrieve. Defaults to 4.

        Returns:
            list[dict]: A list of dictionaries containing chunk IDs, text, metadata, 
            and similarity distance metrics.
        """
        q_result = self.client.models.embed_content(
            model=EMBED_MODEL,
            contents=question,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY", 
                output_dimensionality=EMBED_DIM
            ),
        )
        
        results = self.collection.query(
            query_embeddings=[q_result.embeddings[0].values], 
            n_results=k
        )

        chunks = []
    
        if not results["ids"][0]:
            return chunks

        for cid, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunks.append({"chunk_id": cid, "text": doc, "metadata": meta, "distance": dist})
        
        return chunks

    def expand_to_full_articles(self, chunks: list[dict]) -> list[dict]:
        """
        Expands granular chunks to their full parent articles (Small-to-Big approach)
        and deduplicates them to prevent sending the same parent article multiple times.

        Args:
            chunks (list[dict]): The raw, granular chunks retrieved from ChromaDB.

        Returns:
            list[dict]: A list of expanded context dictionaries with 'title' and 'text'.
        """
        expanded_context = []
        seen_articles = set()

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            madde_no = meta.get("madde_no")

            if madde_no and madde_no in seen_articles:
                continue

            if madde_no:
                seen_articles.add(madde_no)
                full_text = self.article_lookup.get(str(madde_no))

                if full_text:
                    expanded_context.append({
                        "title": f"Maddə {madde_no}",
                        "text": full_text
                    })
                else:
                    expanded_context.append({
                        "title": f"Fragment ({chunk['chunk_id']})",
                        "text": chunk["text"]
                    })
            else:
                expanded_context.append({
                    "title": f"Sənəd ({chunk['chunk_id']})",
                    "text": chunk["text"]
                })

        return expanded_context

    def _build_prompt(self, question: str, expanded_chunks: list[dict]) -> str:
        """
        Constructs the final prompt for the LLM by combining the expanded full articles 
        and the user's question.

        Args:
            question (str): The user's original query.
            expanded_chunks (list[dict]): A list of deduplicated, full-article context dictionaries.

        Returns:
            str: The fully formatted prompt ready for LLM generation.
        """
        context_text = "\n\n".join(f"[{c['title']}]\n{c['text']}" for c in expanded_chunks)
        return f"KONTEKST:\n{context_text}\n\nSUAL: {question}\n\nCAVAB:"

    def generate_answer(self, question: str, expanded_chunks: list[dict]) -> str:
        """
        Generates an answer using the Gemini model based on the provided expanded context.

        Args:
            question (str): The user's query.
            expanded_chunks (list[dict]): A list of deduplicated, full-article context dictionaries.

        Returns:
            str: The generated response text from the LLM.
        """
        prompt = self._build_prompt(question, expanded_chunks)
        
        response = self.client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.SYSTEM_PROMPT,
                temperature=0.5, 
            ),
        )
        return response.text

    def answer_question(self, question: str, k: int = 5) -> str:
        """
        Executes the complete RAG pipeline utilizing the Small-to-Big retrieval strategy.

        Args:
            question (str): The user's input question.
            k (int, optional): The number of granular chunks to retrieve initially. 
                               Lowered to 4 to prevent exceeding token limits when expanded.

        Returns:
            str: The final generated answer in Azerbaijani, or a fallback message 
            if no relevant context is found in the database.
        """
        context_chunks = self.retrieve_context(question, k=k)
        
        if not context_chunks:
            return "Sənəddə bu barədə məlumat tapılmadı."
            
        expanded_chunks = self.expand_to_full_articles(context_chunks)
        print(expanded_chunks)

        return self.generate_answer(question, expanded_chunks)


def main() -> None:
    """
    Main execution block to test the RAG engine locally.
    """
    print("Initializing the engine...")
    rag = FamilyCodeRAGEngine()
    
    # test_question = "şikəstlik sözünün xəsarət sözü ilə əvəz edilməsi hansı qanunla heyata keçirilib"
    test_question = "İstifadə olunmuş sənədlər arasında 20 oktyabr 2006 tarixli hansı qanun var?"
    
    print(f"\nQuestion: {test_question}")
    print("System is thinking...\n")
    
    answer = rag.answer_question(test_question)
    print(f"Answer:\n{answer}")

if __name__ == "__main__":
    main()