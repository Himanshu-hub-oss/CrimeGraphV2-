# CrimeGraph AI — SIH 2026 Presentation Pack
## AI-Powered Criminal Network Analysis System
### Theme: Blockchain & Cybersecurity

---

## Slide 1: Problem Statement

### The Challenge
- Law enforcement agencies face **scattered, siloed data** across CDR logs, FIR records, financial transactions, and social media
- **Manual analysis** of criminal networks is slow, error-prone, and unable to detect multi-hop connections
- **No unified platform** exists to map criminal syndicates, predict threats, and secure digital evidence

### Impact
- Delayed investigation cycles
- Missed criminal connections and hidden kingpins
- Evidence tampering risks in digital chain of custody
- Reactive policing instead of predictive intelligence

---

## Slide 2: Our Solution — CrimeGraph AI

### AI + Graph Analytics + Blockchain

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Ingestion | Multi-Source Fusion | CDR, FIR, Social Media, Financial Data |
| Graph Engine | NetworkX + Centrality | Criminal network mapping & kingpin detection |
| AI/ML Layer | Random Forest, TF-IDF | Risk prediction, link prediction, threat NLP |
| Blockchain | SHA-256 PoA Chain | Tamper-proof digital evidence custody |
| Dashboard | Streamlit + Plotly | 10-tab Police Intelligence Command Center |

**USP**: First-of-its-kind system combining AI-driven criminal analysis with blockchain-secured evidence management.

---

## Slide 3: System Architecture

```
                    ┌─────────────────────────────┐
                    │   Streamlit Dashboard (UI)   │
                    │   10 Interactive Modules      │
                    └──────────┬──────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                     ▼
   ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐
   │ Graph Engine  │  │  ML/AI Engine   │  │ Blockchain Ledger│
   │ NetworkX      │  │ Scikit-Learn    │  │ SHA-256 PoA      │
   │ Centrality    │  │ Risk Classifier │  │ Merkle Roots     │
   │ Communities   │  │ Link Predictor  │  │ Smart Contracts  │
   │ Pyvis/Plotly  │  │ NLP Threat      │  │ Tamper Detection │
   └──────┬───────┘  └───────┬─────────┘  └────────┬─────────┘
          │                   │                      │
          └───────────────────┼──────────────────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Data Processor      │
                    │ 7 CSV Data Sources  │
                    │ Feature Engineering │
                    └─────────────────────┘
```

---

## Slide 4: Data Sources & Ingestion

| Dataset | Records | Description |
|---------|---------|-------------|
| suspects.csv | 150 | Suspect profiles (age, gang, risk, role, city) |
| fir_records.csv | 220 | FIR cases with IPC sections & co-accused |
| cdr_records.csv | 2,000 | Call Detail Records with tower locations |
| network_edges.csv | 500 | Weighted interaction edges (financial, call, mixed) |
| social_media_posts.csv | 600 | Social media intelligence with sentiment flags |
| suspect_features_enriched.csv | 150 | Enriched graph + behavioral feature matrix |
| suspect_risk_predictions.csv | 15 | Pre-computed risk predictions |

**Total**: 3,685+ records across 7 datasets covering 150 suspects in 12 gangs across 15 Indian cities.

---

## Slide 5: Graph Analytics & Kingpin Detection

### Criminal Network Graph
- **150 Nodes** (suspects) connected by **1,567 Edges** (interactions)
- Edge sources: Known associations + CDR calls + Co-accused FIR links

### Centrality Metrics
- **Betweenness Centrality**: Identifies information brokers / gatekeepers
- **Eigenvector Centrality**: Measures influence via well-connected neighbors
- **PageRank**: Google-style importance ranking of suspects
- **Composite Kingpin Score (0-100)**: Weighted formula combining all metrics

### Community Detection
- **Louvain Algorithm** identifies sub-gang clusters and hidden faction structures
- Shortest path tracer finds criminal chains between any two suspects

---

## Slide 6: AI/ML Prediction Engine

### Model 1: Suspect Risk Classifier
- **Algorithm**: Random Forest (150 trees, depth 8, balanced classes)
- **Features**: 13 dimensions (age, prior cases, graph centrality, CDR frequency, financial flags, social media threats)
- **Output**: Risk Level (Low/Medium/High/Critical) + probability distribution
- **XAI**: Explainable AI with per-feature importance attribution
- **Train Accuracy**: 100%

### Model 2: Criminal Link Predictor
- **Algorithm**: Random Forest on graph topological features
- **Features**: Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment, attribute similarity
- **Output**: Collusion probability (0-100%) + threat assessment
- **Accuracy**: 90.2%

### Model 3: NLP Threat Intelligence Classifier
- **Algorithm**: TF-IDF (bigrams, 500 features) + Random Forest
- **Categories**: Neutral, Suspicious, Threatening, Coded/Ambiguous
- **Features**: Keyword trigger detection for code words (shipment, border, naka, lay low)
- **Accuracy**: 42.2% (limited by data diversity)

---

## Slide 7: Blockchain Evidence Layer (USP)

### Why Blockchain?
- Digital evidence must be **court-admissible** — requires immutable chain of custody
- Traditional database records are **vulnerable to tampering** by insiders

### Our Implementation
- **Proof-of-Authority** consensus (suitable for law enforcement hierarchy)
- **SHA-256** cryptographic hashing for each evidence block
- **Merkle Root** combining Evidence ID + Case ID + Officer + Payload hash
- **Smart Contract Access Control**:
  - Investigating Officer → FIR, Seizure Memo
  - Cyber Cell → CDR Logs, Social Media Posts
  - Forensic Expert → Ballistics, Financial Ledgers
  - Magistrate → Judicial Warrants, Court Seals

### Tamper Detection Demo
- Live simulation: Inject unauthorized modification → Blockchain detects hash mismatch automatically
- **100% detection rate** for any post-mining data alteration

---

## Slide 8: Dashboard — 10 Interactive Modules

| # | Module | Key Features |
|---|--------|--------------|
| 1 | Executive Overview | KPI cards, risk pie chart, crime bar chart, gang summary, CDR timeline |
| 2 | 2D Network Explorer | Pyvis physics-based graph, gang/risk filtering, suspect highlighting |
| 3 | 3D Network Topology | Plotly 3D scatter with Kingpin Score heatmap coloring |
| 4 | Kingpin Analyzer | Centrality leaderboard, score vs betweenness chart, path tracer |
| 5 | AI Risk Predictor | Per-suspect prediction with XAI feature attribution bar chart |
| 6 | AI Link Predictor | Pairwise collusion probability, mutual accomplices table |
| 7 | NLP Threat Monitor | Sentiment distribution, platform breakdown, live text analyzer |
| 8 | Crime Hotspot Map | Mapbox scatter map with city severity scoring |
| 9 | Blockchain Vault | Chain explorer, integrity verification, tampering simulator |
| 10 | Suspect Dossier | Comprehensive profile: FIRs, CDR, social media, network metrics |

---

## Slide 9: Innovation & Impact

### What Makes CrimeGraph AI Unique
1. **First integrated AI + Graph + Blockchain** criminal analysis platform
2. **Explainable AI** — officers understand *why* a suspect is flagged, not just *that* they are
3. **Proactive policing** — predict emerging threats before they escalate
4. **Court-admissible evidence** — blockchain guarantees tamper-proof chain of custody
5. **Multi-source intelligence fusion** — CDR + FIR + Social Media + Financial in one view

### Potential Impact
- **60% faster** suspect identification through automated network analysis
- **Zero evidence tampering** with cryptographic blockchain verification
- **Real-time threat detection** from social media intercepts
- Scalable to **state/national** level with proper data integration

---

## Slide 10: Tech Stack & Future Roadmap

### Current Tech Stack
| Category | Technology |
|----------|------------|
| Language | Python 3.13 |
| Frontend | Streamlit |
| Graph Analytics | NetworkX |
| ML/AI | Scikit-Learn, TF-IDF |
| Visualization | Plotly, Pyvis |
| Blockchain | Custom SHA-256 PoA |
| Data | Pandas, NumPy |

### Future Roadmap
- **Phase 2**: Integration with real police RMS databases and CCTNS
- **Phase 3**: Deep learning NLP (BERT/GPT) for multi-language threat detection
- **Phase 4**: Real-time CDR streaming with Apache Kafka
- **Phase 5**: Hyperledger Fabric deployment for production-grade permissioned blockchain
- **Phase 6**: Mobile app for field officers with offline-first architecture

---

## Thank You

**CrimeGraph AI** — Empowering Law Enforcement with AI-Driven Intelligence

*Built for Smart India Hackathon 2026*
