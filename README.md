<div align="center">
  
  # Crisis Search Reranker

  ### Search results you can trust — when truth matters most

  [![Status](https://img.shields.io/badge/Status-In_Progress_(Day_1%2F14)-EA580C?style=for-the-badge)](https://github.com/Jaya242/crisis-search-reranker)
  [![Python](https://img.shields.io/badge/Python-3.12-EA580C?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
  [![HF Transformers](https://img.shields.io/badge/Transformers-EA580C?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/docs/transformers)
  [![License](https://img.shields.io/badge/License-MIT-EA580C?style=for-the-badge)](LICENSE)

  </div>
  
  ---                  

  ## The Problem
  
  When you search _"wildfire evacuation routes"_ during a real disaster,
   popular search engines may surface:

  - Articles from **3 years ago**
  - **Misinformation** that went viral on social media
  - Lower-quality content that ranks high through SEO games
  
  During a crisis, "popular" can be deadly. People need **recent, 
  credible, actionable information** — fast.

  ## The Solution

  A search reranker that takes existing search results and re-orders
  them based on four signals. When the query is detected as
  crisis-related, the weights automatically flip to prioritize freshness
   and credibility.

  | Signal | What it measures | Why it matters in a crisis |
  |---|---|---|
  | **Credibility** | Fake-news classifier (DistilBERT, fine-tuned onLIAR) | Filter out misinformation at the source |
  | **Freshness** | Time-decay scoring on publish date | A flood from 2019 isn't today's flood |
  | **Relevance** | Semantic similarity to query | Match user intent,not just keywords |
  | **Behavioral signal** | Pogo-stick simulation (dwell vs back-button)| Real users vote with their behavior |
  
  ## Architecture

  ```
  USER QUERY
      │
      ▼                
  ┌────────────────────────────────────┐
  │  1. Retrieval                       │  ← 80-article curated corpus
  └────────────────────────────────────┘
      │
      ▼
  ┌────────────────────────────────────┐
  │  2. Credibility Classifier          │  ← DistilBERT fine-tuned on
  LIAR
  └────────────────────────────────────┘
      │                
      ▼
  ┌────────────────────────────────────┐
  │  3. Freshness Scoring               │  ← Time-decay function
  └────────────────────────────────────┘
      │                
      ▼
  ┌────────────────────────────────────┐
  │  4. Crisis-Mode Detector            │  ← LLM call to flip weights
  └────────────────────────────────────┘
      │
      ▼
  ┌────────────────────────────────────┐
  │  5. Composite Ranker                │  ← Weighted score → sort
  └────────────────────────────────────┘
      │
      ▼
  ┌────────────────────────────────────┐
  │  6. Behavioral Feedback             │  ← Simulated user signals
  └────────────────────────────────────┘
      │
      ▼
  ┌────────────────────────────────────┐
  │  7. RAG Summary                     │  ← LLM reads top-3 results
  └────────────────────────────────────┘
      │
      ▼                
  RANKED RESULTS + LATEST SITUATION SUMMARY
  ```
  
  ## Status: Day 1 of 14 Shipped

  Building this in public over 14 days. Current progress:

  - [x] **Day 1** — Project setup, LIAR dataset exploration, 80-article
  corpus assembled
  - [ ] **Days 2-5** — DistilBERT credibility classifier (fine-tune +
  eval)
  - [ ] **Days 6-7** — Freshness scoring + composite ranker
  - [ ] **Day 8** — Crisis-mode detector (LLM-based)
  - [ ] **Days 9-10** — Pogo-stick behavioral feedback simulator
  - [ ] **Days 11-12** — RAG summary layer
  - [ ] **Days 13-14** — Gradio UI + HuggingFace Spaces deployment

  ## Dataset

  ### Credibility training: LIAR (Wang, UCSB 2017)
  - **12,836** short political statements fact-checked by
  [Politifact](https://www.politifact.com/)
  - 6 truthfulness labels (`pants-fire` → `true`), collapsed to binary
  (real/fake) for training
  - Source: [cs.ucsb.edu/~william/data/liar_dataset.zip](https://www.cs.
  ucsb.edu/~william/data/liar_dataset.zip)
  
  ### Retrieval corpus: 80 hand-curated articles
  - **40 real news** pulled from BBC, Guardian, NPR RSS feeds
  - **40 misinformation samples** from LIAR's pants-fire and false
  labels    
  - Balanced across: crisis vs non-crisis topics, recent vs older
  timestamps
  - Stored as `data/corpus.json` (gitignored, regeneratable via
  `notebooks/02_build_corpus.py`)
  
  ## Tech Stack

  | Layer | Technology |
  |---|---|
  | Credibility classifier | PyTorch + HuggingFace Transformers (DistilBERT) |
  | Data processing | pandas, feedparser, HuggingFace `datasets` |
  | Retrieval | sentence-transformers (semantic similarity) |
  | Crisis detection | Claude / OpenAI API |
  | RAG summary | Claude / OpenAI API |
  | UI | Gradio |
  | Deployment | HuggingFace Spaces |

  ## Project Structure
  
  ```
  crisis-search-reranker/
  ├── data/
  │   ├── liar/                   # LIAR TSV files (gitignored)
  │   └── corpus.json             # 80-article retrieval corpus
  (gitignored) 
  ├── notebooks/
  │   ├── 01_explore_liar.py      # LIAR dataset exploration
  │   └── 02_build_corpus.py      # Corpus assembly script
  ├── src/  
  │   ├── classifier.py           # DistilBERT credibility classifier
  │   ├── ranker.py               # Composite ranker
  │   ├── crisis_detector.py      # LLM-based crisis mode detector
  │   └── feedback_sim.py         # Pogo-stick behavioral feedback
  ├── app.py                      # Gradio interface (TBD)
  ├── requirements.txt
  ├── LICENSE
  └── README.md
  ```

  ## Setup
                       
  ```bash
  # Clone
  git clone https://github.com/Jaya242/crisis-search-reranker.git
  cd crisis-search-reranker

  # Virtual environment
  python3 -m venv venv
  source venv/bin/activate
  
  # Install dependencies
  pip install -r requirements.txt

  # Download LIAR dataset
  mkdir -p data        
  curl -L https://www.cs.ucsb.edu/~william/data/liar_dataset.zip -o
  data/liar_dataset.zip
  unzip data/liar_dataset.zip -d data/liar

  # Build the 80-article corpus 
  python notebooks/02_build_corpus.py
  ```

  ## Future Roadmap (Post-MVP)
  
  The MVP demonstrates the core insight on a fixed corpus. Production
  extensions:

  - **Live search integration** — replace fixed corpus with NewsAPI /
  web search
  - **Multi-language support** — extend beyond English
  - **React frontend** — production UI instead of Gradio
  - **Elasticsearch backend** — scale retrieval to millions of documents
  - **Crowdsourced credibility signals** — incorporate real user
  feedback
  - **Multi-agent reranking** — LangGraph pipeline with specialized
  agents

  ## Inspiration       

  This project was inspired by the observation that during real crises
  (COVID, wildfires, geopolitical conflicts), search engines have
  struggled to consistently surface trustworthy, current information.
  While platforms have added crisis-response panels for major events,
  the underlying ranking algorithms don't fundamentally re-prioritize
  for emergency contexts.
  
  ## About             

  **Jaya Arora** — Mechanical engineering student at MNNIT Allahabad,
  self-teaching AI/ML.

  From CAD to CNNs — building deployable AI systems as a 40-day
  portfolio sprint.   

  - Other projects: [Traffic Analytics](https://github.com/Jaya242/traffic_detector) · [Emotion Classifier](https://github.com/Jaya242/emotion_classifier)
  - [@Jaya242](https://github.com/Jaya242)
  
  ## License

  MIT — see [LICENSE](LICENSE).

