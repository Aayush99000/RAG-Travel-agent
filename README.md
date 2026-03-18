# **TripCraft 🧳✈️**

> An NLP-powered travel planner that converts your preferences — destination, budget, mood, dates, and transport — into a personalized day-by-day itinerary. Built with RAG (Groq + Llama 3 + ChromaDB), semantic mood-to-activity mapping, and real venue data from Yelp API. Supports multi-turn conversational refinement.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## **📌 Overview**

TripCraft takes the hassle out of travel planning. Instead of spending hours researching places, manually budgeting, and building day schedules — just describe your trip in natural language and TripCraft generates a complete, grounded itinerary tailored to you.

**Example Input:**

> _"I'm flying from San Francisco to New York for 5 days with a $2,000 budget. I prefer public transport, and I'm in the mood for art, local food, and some chill walks."_

**Example Output:**
A full 5-day itinerary with morning/afternoon/evening slots, real venue recommendations, estimated costs per day, and transport suggestions — all within your budget.

---

## **🧠 NLP Pipeline**

```
User Input (Natural Language / Form)
        ↓
  Slot Filling & NER (spaCy / LLM-based)
  [destination, dates, budget, transport, mood]
        ↓
  Mood-to-Activity Mapping
  [sentence-transformers + cosine similarity]
        ↓
  RAG Retrieval (ChromaDB + Yelp Fusion API)
  [real venue data to ground generation]
        ↓
  Itinerary Generation (Groq / Llama 3)
  [constraint-aware, structured prompting]
        ↓
  Multi-turn Refinement
  [conversational edits via LangChain memory]
        ↓
  Output: Day-by-day itinerary (Streamlit UI)
```

---

## **✨ Features**

- 🗓️ **Day-by-day itinerary generation** from natural language or form input
- 🎭 **Mood-to-activity mapping** — maps vague descriptors like _"chill"_ or _"adventurous"_ to concrete activity categories using sentence embeddings
- 📍 **Real venue recommendations** grounded via Yelp Fusion / Google Places API
- 💰 **Budget & transport constraint enforcement** — stays within your limits
- 💬 **Multi-turn refinement** — tweak your plan conversationally (e.g., _"make Day 3 less packed"_)
- 🔍 **RAG-powered grounding** to minimize hallucinations

---

## **🛠️ Tech Stack**

| **Layer**     | **Tool**              |
| ------------- | --------------------- |
| LLM           | Llama 3 via Groq API  |
| Orchestration | LangChain             |
| Vector Store  | ChromaDB              |
| Embeddings    | sentence-transformers |
| NLP / NER     | spaCy                 |
| Venue Data    | Yelp Fusion API       |
| Frontend      | Streamlit             |
| Language      | Python 3.10+          |

---

## **🚀 Getting Started**

### **1. Clone the Repository**

```bash
git clone https://github.com/Aayush99000/TripCraft.git
cd TripCraft
```

### **2. Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Set Up Environment Variables**

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
YELP_API_KEY=your_yelp_fusion_api_key
```

### **5. Run the App**

```bash
streamlit run app.py
```

---

## **📁 Project Structure**

```
TripCraft/
├── app.py                  # Streamlit frontend
├── pipeline/
│   ├── slot_filler.py      # NER & structured input extraction
│   ├── mood_mapper.py      # Mood-to-activity semantic mapping
│   ├── retriever.py        # RAG retrieval from ChromaDB
│   └── generator.py        # LLM itinerary generation
├── data/
│   └── venue_data/         # Cached or scraped venue data
├── vectorstore/            # ChromaDB persistent store
├── requirements.txt
├── .env.example
└── README.md
```

---

## **📊 Evaluation**

TripCraft is evaluated on:

- **Constraint Satisfaction Rate** — how well the itinerary respects budget and transport inputs
- **BERTScore** — semantic similarity of generated itineraries vs. reference travel blogs
- **Human Preference Scoring** — user ratings on relevance, coherence, and personalization

---

## **🗺️ Roadmap**

- [x] Project proposal & architecture design
- [ ] Slot filling & input parsing module
- [ ] Mood-to-activity embedding layer
- [ ] RAG pipeline with Yelp API integration
- [ ] Itinerary generation with Groq / Llama 3
- [ ] Multi-turn refinement
- [ ] Streamlit UI
- [ ] Evaluation & final report

---

## **📄 License**

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## **🙋 Authors**

**Aayush** — MS Data Science, Northeastern University.
**Kaushal** — MS Data Science, Northeastern University
