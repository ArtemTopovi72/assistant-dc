"""Benchmark task suite for the Deep Research scored evaluation harness (spec D).

50+ tasks across six categories. Each task is a topic plus light gold signals used
by the scorer (no full reference answers — those don't exist for live web research):

    must_terms     : substrings that a correct, complete answer should contain
                     (case-insensitive) — used for answer_completeness / correctness
    expect_primary : True if authoritative/primary sources are expected to exist
    expect_contra  : True if the topic is genuinely contested (contradiction coverage)

Categories: technology, science, history, current_events, long_horizon, contradiction.
"""

TASKS = [
    # ---------------- technology (10) ----------------
    {"topic": "Qdrant vs Weaviate vs Milvus for production RAG", "category": "technology",
     "must_terms": ["qdrant", "milvus", "weaviate", "vector"], "expect_primary": True, "expect_contra": True},
    {"topic": "BGE-M3 vs Cohere rerank for retrieval quality", "category": "technology",
     "must_terms": ["rerank", "bge", "cohere"], "expect_primary": True, "expect_contra": True},
    {"topic": "Rust vs Go for high-throughput network services", "category": "technology",
     "must_terms": ["rust", "go", "concurrency"], "expect_primary": False, "expect_contra": True},
    {"topic": "HNSW vs IVF-PQ approximate nearest neighbor tradeoffs", "category": "technology",
     "must_terms": ["hnsw", "recall", "latency"], "expect_primary": True, "expect_contra": False},
    {"topic": "WebAssembly use cases outside the browser in 2025", "category": "technology",
     "must_terms": ["webassembly", "wasm"], "expect_primary": False, "expect_contra": False},
    {"topic": "Postgres vs MySQL for OLTP workloads", "category": "technology",
     "must_terms": ["postgres", "mysql", "transaction"], "expect_primary": False, "expect_contra": True},
    {"topic": "Kubernetes vs Nomad for container orchestration", "category": "technology",
     "must_terms": ["kubernetes", "nomad"], "expect_primary": False, "expect_contra": True},
    {"topic": "QLoRA vs full fine-tuning for LLM adaptation", "category": "technology",
     "must_terms": ["qlora", "fine-tun"], "expect_primary": True, "expect_contra": True},
    {"topic": "gRPC vs REST for microservice communication", "category": "technology",
     "must_terms": ["grpc", "rest"], "expect_primary": False, "expect_contra": True},
    {"topic": "RISC-V adoption in data center processors", "category": "technology",
     "must_terms": ["risc-v"], "expect_primary": False, "expect_contra": False},

    # ---------------- science (10) ----------------
    {"topic": "CRISPR base editing vs prime editing accuracy", "category": "science",
     "must_terms": ["crispr", "prime editing", "base editing"], "expect_primary": True, "expect_contra": True},
    {"topic": "mRNA vaccine durability against respiratory viruses", "category": "science",
     "must_terms": ["mrna", "vaccine"], "expect_primary": True, "expect_contra": True},
    {"topic": "Room-temperature superconductivity claims status", "category": "science",
     "must_terms": ["superconduct"], "expect_primary": True, "expect_contra": True},
    {"topic": "JWST findings on early galaxy formation", "category": "science",
     "must_terms": ["jwst", "galaxy"], "expect_primary": True, "expect_contra": False},
    {"topic": "Gut microbiome influence on immune regulation", "category": "science",
     "must_terms": ["microbiome", "immune"], "expect_primary": True, "expect_contra": False},
    {"topic": "Tokamak vs stellarator for fusion confinement", "category": "science",
     "must_terms": ["tokamak", "stellarator", "fusion"], "expect_primary": True, "expect_contra": True},
    {"topic": "Amyloid hypothesis of Alzheimer's disease debate", "category": "science",
     "must_terms": ["amyloid", "alzheimer"], "expect_primary": True, "expect_contra": True},
    {"topic": "Carbon capture direct air capture energy cost", "category": "science",
     "must_terms": ["carbon capture", "direct air"], "expect_primary": True, "expect_contra": True},
    {"topic": "Quantum error correction surface code progress", "category": "science",
     "must_terms": ["quantum", "error correction"], "expect_primary": True, "expect_contra": False},
    {"topic": "Ocean acidification effects on coral calcification", "category": "science",
     "must_terms": ["acidification", "coral"], "expect_primary": True, "expect_contra": False},

    # ---------------- history (8) ----------------
    {"topic": "Causes of the fall of the Western Roman Empire", "category": "history",
     "must_terms": ["roman", "empire"], "expect_primary": False, "expect_contra": True},
    {"topic": "Economic impact of the Marshall Plan on postwar Europe", "category": "history",
     "must_terms": ["marshall plan", "europe"], "expect_primary": False, "expect_contra": True},
    {"topic": "Role of the printing press in the Reformation", "category": "history",
     "must_terms": ["printing press", "reformation"], "expect_primary": False, "expect_contra": False},
    {"topic": "Causes of the 1929 Wall Street crash", "category": "history",
     "must_terms": ["1929", "crash"], "expect_primary": False, "expect_contra": True},
    {"topic": "Industrial Revolution effects on urbanization", "category": "history",
     "must_terms": ["industrial revolution", "urban"], "expect_primary": False, "expect_contra": False},
    {"topic": "Bronze Age collapse leading theories", "category": "history",
     "must_terms": ["bronze age", "collapse"], "expect_primary": False, "expect_contra": True},
    {"topic": "Decline of the Silk Road trade networks", "category": "history",
     "must_terms": ["silk road"], "expect_primary": False, "expect_contra": False},
    {"topic": "Impact of the Columbian Exchange on global agriculture", "category": "history",
     "must_terms": ["columbian exchange"], "expect_primary": False, "expect_contra": False},

    # ---------------- current_events (8) ----------------
    {"topic": "EU AI Act compliance requirements for foundation models", "category": "current_events",
     "must_terms": ["ai act", "compliance"], "expect_primary": True, "expect_contra": True},
    {"topic": "Global semiconductor supply chain resilience 2025", "category": "current_events",
     "must_terms": ["semiconductor", "supply chain"], "expect_primary": False, "expect_contra": True},
    {"topic": "State of grid-scale battery storage deployment", "category": "current_events",
     "must_terms": ["battery", "grid"], "expect_primary": False, "expect_contra": False},
    {"topic": "Central bank digital currency pilots progress", "category": "current_events",
     "must_terms": ["cbdc", "digital currency"], "expect_primary": True, "expect_contra": True},
    {"topic": "Open-weight LLM licensing landscape 2025", "category": "current_events",
     "must_terms": ["license", "open"], "expect_primary": True, "expect_contra": True},
    {"topic": "Electric vehicle charging standard consolidation NACS vs CCS", "category": "current_events",
     "must_terms": ["nacs", "ccs", "charging"], "expect_primary": False, "expect_contra": True},
    {"topic": "Heat pump adoption barriers in cold climates", "category": "current_events",
     "must_terms": ["heat pump"], "expect_primary": False, "expect_contra": True},
    {"topic": "Undersea cable security and resilience concerns", "category": "current_events",
     "must_terms": ["cable", "undersea"], "expect_primary": False, "expect_contra": False},

    # ---------------- long_horizon (8) ----------------
    {"topic": "Long-term effects of remote work on urban commercial real estate", "category": "long_horizon",
     "must_terms": ["remote work", "real estate"], "expect_primary": False, "expect_contra": True},
    {"topic": "Demographic decline consequences for pension systems", "category": "long_horizon",
     "must_terms": ["pension", "demographic"], "expect_primary": False, "expect_contra": True},
    {"topic": "Antibiotic resistance trajectory and economic burden", "category": "long_horizon",
     "must_terms": ["antibiotic resistance"], "expect_primary": True, "expect_contra": False},
    {"topic": "AGI timeline forecasts and their disagreements", "category": "long_horizon",
     "must_terms": ["agi", "forecast"], "expect_primary": False, "expect_contra": True},
    {"topic": "Long-term water scarcity risk in major agricultural regions", "category": "long_horizon",
     "must_terms": ["water scarcity"], "expect_primary": False, "expect_contra": True},
    {"topic": "Compounding effects of soil degradation on food security", "category": "long_horizon",
     "must_terms": ["soil", "food security"], "expect_primary": True, "expect_contra": False},
    {"topic": "Space debris growth and Kessler syndrome risk", "category": "long_horizon",
     "must_terms": ["debris", "kessler"], "expect_primary": True, "expect_contra": True},
    {"topic": "Permafrost thaw feedback on climate projections", "category": "long_horizon",
     "must_terms": ["permafrost"], "expect_primary": True, "expect_contra": True},

    # ---------------- contradiction-heavy (8) ----------------
    {"topic": "Does intermittent fasting outperform calorie restriction", "category": "contradiction",
     "must_terms": ["fasting", "calorie"], "expect_primary": True, "expect_contra": True},
    {"topic": "Is nuclear power cost-competitive with renewables", "category": "contradiction",
     "must_terms": ["nuclear", "renewable", "cost"], "expect_primary": False, "expect_contra": True},
    {"topic": "Effectiveness of minimum wage increases on employment", "category": "contradiction",
     "must_terms": ["minimum wage", "employment"], "expect_primary": True, "expect_contra": True},
    {"topic": "Do violent video games increase aggression", "category": "contradiction",
     "must_terms": ["video game", "aggression"], "expect_primary": True, "expect_contra": True},
    {"topic": "Is dietary saturated fat a cardiovascular risk", "category": "contradiction",
     "must_terms": ["saturated fat", "cardiovascular"], "expect_primary": True, "expect_contra": True},
    {"topic": "Microservices vs monolith which is better architecture", "category": "contradiction",
     "must_terms": ["microservice", "monolith"], "expect_primary": False, "expect_contra": True},
    {"topic": "Does homework improve student achievement", "category": "contradiction",
     "must_terms": ["homework", "achievement"], "expect_primary": True, "expect_contra": True},
    {"topic": "Is remote work more productive than office work", "category": "contradiction",
     "must_terms": ["remote work", "productiv"], "expect_primary": False, "expect_contra": True},
]


def by_category():
    out = {}
    for t in TASKS:
        out.setdefault(t["category"], []).append(t)
    return out


if __name__ == "__main__":
    cats = by_category()
    print(f"{len(TASKS)} tasks across {len(cats)} categories:")
    for c, ts in cats.items():
        print(f"  {c}: {len(ts)}")
