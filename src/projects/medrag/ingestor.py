from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from llama_index.core.schema import Document

from src.core.base import DocumentIngestor

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class MedRAGIngestor(DocumentIngestor):
    bootstrap_documents = (
        {
            "title": "Type 2 diabetes first-line therapy overview",
            "text": (
                "Type 2 diabetes management usually begins with lifestyle modification, "
                "glycemic monitoring, and metformin when there are no contraindications. "
                "Guidelines commonly describe individualized escalation to GLP-1 receptor "
                "agonists, SGLT2 inhibitors, or insulin based on comorbidities, kidney "
                "function, cardiovascular risk, and glycemic control."
            ),
            "specialty": "endocrinology",
        },
        {
            "title": "Type 1 diabetes treatment overview",
            "text": (
                "Type 1 diabetes requires insulin replacement therapy rather than oral "
                "first-line agents. Guideline-based care typically includes basal-bolus "
                "or pump-based insulin delivery, glucose monitoring, hypoglycemia "
                "education, and individualized nutrition planning."
            ),
            "specialty": "endocrinology",
        },
        {
            "title": "Hypertension guideline overview",
            "text": (
                "Hypertension guidelines generally cover blood pressure thresholds, "
                "target ranges, home blood pressure monitoring, lifestyle changes, and "
                "medication classes such as ACE inhibitors, ARBs, calcium-channel "
                "blockers, and thiazide-type diuretics. Follow-up cadence depends on "
                "severity, symptoms, and treatment response."
            ),
            "specialty": "cardiology",
        },
        {
            "title": "Why medical guidance should include a disclaimer",
            "text": (
                "Medical guidance in retrieval-augmented systems should include a disclaimer "
                "that the information is for educational purposes only and is not a substitute "
                "for clinician judgment, diagnosis, or personalized treatment. The disclaimer "
                "helps users understand the limits of the retrieved material and encourages "
                "consultation with qualified healthcare professionals."
            ),
            "specialty": "general_medicine",
        },
    )
    pubmed_queries = (
        "type 2 diabetes treatment guideline",
        "hypertension management guideline",
    )
    pubmed_max_results = 5

    def load_and_parse(self):
        documents = []
        documents.extend(self._load_guideline_pdfs())
        documents.extend(self._load_pubmed_abstracts())
        if os.getenv("MEDRAG_INCLUDE_BOOTSTRAP", "true").lower() in {"1", "true", "yes"}:
            documents.extend(self._load_bootstrap_documents())
        return documents

    def enrich_metadata(self, docs):
        for doc in docs:
            metadata = dict(getattr(doc, "metadata", {}) or {})
            source_file = str(metadata.get("source_file", "")).lower()
            text_hint = " ".join(
                [
                    source_file,
                    str(metadata.get("title", "")),
                    str(metadata.get("query", "")),
                ]
            ).lower()

            if metadata.get("source") == "pubmed":
                metadata["source_org"] = "PubMed"
                metadata["evidence_type"] = "research_abstract"
            elif any(token in source_file for token in ("fda", "dailymed", "label")):
                metadata["source_org"] = "FDA"
                metadata["evidence_type"] = "drug_label"
            else:
                metadata["source_org"] = metadata.get("source_org", "WHO")
                metadata["evidence_type"] = metadata.get("evidence_type", "guideline")

            metadata["specialty"] = self._infer_specialty(text_hint)
            doc.metadata = metadata
        return docs

    def _load_guideline_pdfs(self):
        guideline_dir = Path(self.config.data_dir) / "guidelines"
        pdf_paths = sorted(guideline_dir.glob("*.pdf"))
        max_guideline_files = int(os.getenv("MAX_GUIDELINE_FILES", "3"))
        if max_guideline_files >= 0:
            pdf_paths = pdf_paths[:max_guideline_files]
        if not pdf_paths:
            return []

        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLAMA_CLOUD_API_KEY is required to parse guideline PDFs with LlamaParse."
            )

        from llama_cloud import LlamaCloud

        client = LlamaCloud(api_key=api_key)
        tier = os.getenv("LLAMA_PARSE_TIER", "cost_effective")

        documents = []
        for pdf_path in pdf_paths:
            # Raises on failure so a bad parse surfaces as an error instead of an empty index.
            uploaded = client.files.create(file=pdf_path, purpose="parse")
            result = client.parsing.parse(
                tier=tier,
                version="latest",
                file_id=uploaded.id,
                expand=["markdown"],
                timeout=1800,
            )
            pages = result.markdown.pages if result.markdown else []
            for page in pages:
                text = getattr(page, "markdown", None)
                if not getattr(page, "success", True) or not text or not text.strip():
                    continue
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "source_file": pdf_path.name,
                            "page": page.page_number,
                            "parser": "llamaparse_v2",
                            "source": "guideline_pdf",
                        },
                    )
                )
            if not any(d.metadata["source_file"] == pdf_path.name for d in documents):
                raise RuntimeError(f"LlamaParse returned no text for '{pdf_path.name}'.")
        return documents

    def _load_pubmed_abstracts(self):
        if os.getenv("PUBMED_ENABLED", "true").lower() not in {"1", "true", "yes"}:
            return []

        max_queries = int(os.getenv("PUBMED_QUERY_LIMIT", "1"))
        queries = self.pubmed_queries[:max_queries] if max_queries > 0 else ()
        max_results = int(os.getenv("PUBMED_MAX_RESULTS", str(self.pubmed_max_results)))

        documents = []
        for query in queries:
            documents.extend(_fetch_pubmed_documents(query, max_results))
        return documents

    def _load_bootstrap_documents(self):
        documents = []
        for item in self.bootstrap_documents:
            documents.append(
                Document(
                    text=item["text"],
                    metadata={
                        "source": "bootstrap",
                        "source_file": "bootstrap_seed",
                        "title": item["title"],
                        "parser": "bootstrap_seed",
                        "source_org": "Bootstrap",
                        "evidence_type": "guideline_summary",
                        "specialty": item["specialty"],
                    },
                )
            )
        return documents

    @staticmethod
    def _infer_specialty(text_hint: str) -> str:
        mapping = {
            "diabetes": "endocrinology",
            "glucose": "endocrinology",
            "metformin": "endocrinology",
            "insulin": "endocrinology",
            "hypertension": "cardiology",
            "blood pressure": "cardiology",
            "warfarin": "cardiology",
            "lisinopril": "cardiology",
            "asthma": "pulmonology",
            "copd": "pulmonology",
            "kidney": "nephrology",
        }
        for keyword, specialty in mapping.items():
            if keyword in text_hint:
                return specialty
        return "general_medicine"


def _fetch_pubmed_documents(query: str, max_results: int) -> list[Document]:
    """Search PubMed via NCBI E-utilities and return one Document per abstract."""
    common = {"tool": "medrag"}
    try:
        search = requests.get(
            f"{EUTILS_BASE}/esearch.fcgi",
            params={**common, "db": "pubmed", "term": query, "retmax": max_results,
                    "retmode": "json", "sort": "relevance"},
            timeout=30,
        )
        search.raise_for_status()
        pmids = search.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            logger.warning("PubMed returned no results for query %r.", query)
            return []

        fetch = requests.get(
            f"{EUTILS_BASE}/efetch.fcgi",
            params={**common, "db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
            timeout=60,
        )
        fetch.raise_for_status()
        root = ET.fromstring(fetch.content)
    except (requests.RequestException, ET.ParseError, ValueError):
        logger.exception("PubMed fetch failed for query %r; continuing without it.", query)
        return []

    documents: list[Document] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//MedlineCitation/PMID") or "").strip()
        title = "".join(article.find(".//ArticleTitle").itertext()).strip() \
            if article.find(".//ArticleTitle") is not None else ""
        abstract = " ".join(
            "".join(part.itertext()).strip() for part in article.findall(".//Abstract/AbstractText")
        ).strip()
        if not abstract:
            continue
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        documents.append(
            Document(
                text=f"{title}\n\n{abstract}" if title else abstract,
                doc_id=f"pubmed-{pmid}",
                metadata={
                    "source": "pubmed",
                    "query": query,
                    "parser": "pubmed_eutils",
                    "title": title,
                    "pmid": pmid,
                    "URL": url,
                    "journal": article.findtext(".//Journal/Title") or "",
                    "year": article.findtext(".//JournalIssue/PubDate/Year") or "",
                },
            )
        )
    return documents
