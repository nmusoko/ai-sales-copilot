# 🤖 AI Sales Copilot  
### *LangChain · Whisper · Streamlit · Python*

---

## 📌 Overview
Sales representatives in small and medium businesses (SMBs) spend **10–20 minutes** after every client call summarizing discussions, drafting follow-up emails, and updating CRMs manually.  
**AI Sales Copilot** automates this process using generative AI — it listens, understands, and writes like a real sales assistant.

---

## 🎯 Problem → Solution → Impact

**Problem**  
Post-call administrative tasks are repetitive, time-consuming, and error-prone.  
Sales reps lose focus and valuable time that could be spent closing deals.

**Solution**  
AI Sales Copilot automatically:
1. Transcribes sales calls (via Whisper)
2. Summarizes discussions and detects sentiment
3. Generates a professional follow-up email in the rep’s tone
4. Logs key action items and next steps in a structured CRM format

**Impact**  
- Saves ~10 minutes per call  
- Ensures consistent, high-quality communication  
- Keeps CRM data accurate and up-to-date  

---

## 🧠 Key Features (MVP)

✅ Upload audio (.wav) or text transcript  
✅ Automatic call transcription (Whisper)  
✅ Smart summarization & sentiment analysis (LangChain + LLM)  
✅ AI-generated follow-up email (tone-matched)  
✅ Extracted “next steps” and action items  
✅ Save/export results to a local SQLite CRM  
✅ Streamlit dashboard for interactive use  

---

## 📊 Success Metrics

| Metric | Target |
|:--|:--|
| ⏱ Time saved per call      | ≥ 10 min |
| ✅ JSON valid outputs      | ≥ 95 % |
| 📨 Email quality rating (1-5) | ≥ 4.0 |
| ⚡ Latency (audio → email) | ≤ 2 min |
| 🎯 Data accuracy (facts retained) | ≥ 90 % |

---

## 👤 User Persona

**Role:** Sales Account Executive (SMB)  
**Needs:** Faster follow-ups, less admin time, consistent emails  

**User Stories**
1. Upload a sales call and receive a concise summary in < 30 s.  
2. Automatically generate a ready-to-send follow-up email.  
3. Extract next steps and deadlines for CRM logging.  
4. Download or export results as JSON or Markdown.  

---

## ⚙️ Tech Stack
- **LLM & Orchestration:** LangChain · IBM watsonx / OpenAI GPT  
- **Speech-to-Text:** Whisper  
- **UI:** Streamlit  
- **Data Storage:** SQLite · pandas · Chroma (optional memory)  
- **Validation:** Pydantic · JSON Schema  
- **Environment:** Python 3.10+

---