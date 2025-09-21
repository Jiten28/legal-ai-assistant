# 🧾 Legal AI Assistant — Demystifying Legal Documents

## 🚀 Overview
Legal contracts and agreements are often filled with complex jargon, making them difficult to understand for everyday users.  
Our solution leverages **Generative AI** to simplify legal documents into **plain language**, highlight **risky clauses**, and allow users to **ask questions directly** about their contracts.

This project was built for the challenge: **Generative AI for Demystifying Legal Documents**.

---

## ✨ Features
- 📂 **Upload Legal Documents** (PDF, DOCX, TXT)  
- 📝 **Plain-Language Summaries** of complex legal text  
- ⚠️ **Risk Flagging** (e.g., termination, liability, penalties)  
- ❓ **Question-Answering** about the document content  
- 🛡️ **Demo Mode** → works even if API quota is exhausted (using document snippet to generate realistic sample outputs)

---

## 🖥️ How to Run Locally

### 1️⃣ Clone the repo
```bash
git clone https://github.com/Jiten28/legal-ai-assistant.git
cd legal-ai-assistant
````

### 2️⃣ Create a virtual environment

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```


### 4️⃣ Set your OpenAI API Key

Add your API key to environment variables:

Windows (PowerShell):

```powershell
setx OPENAI_API_KEY "sk-xxxxxxxx"
```

---

## Screenshots

- Home Page
  <img width="1865" height="884" alt="image" src="https://github.com/user-attachments/assets/e206fa6d-e62a-4cf5-864b-25d447bda631" />


- Home Page with feature Testing
  <img width="1865" height="876" alt="image" src="https://github.com/user-attachments/assets/4996def3-4b3e-44f8-a6ee-1c1cfd935a0b" />


---
### 5️⃣ Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## ☁️ Deploy on Streamlit Cloud (Prototype Link)

1. Push your project to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Create a new app → select your repo and `app.py`.
4. In **App → Settings → Secrets**, add:

   ```toml
   OPENAI_API_KEY = "sk-xxxxxxxx"
   ```
5. Deploy → you’ll get a public link like:

   ```
   https://jiten-legal-ai-assistant.streamlit.app
   ```

---

## 📦 Requirements

```
streamlit
openai==0.28
pdfplumber
python-docx
```

---

## ⚠️ Notes & Limitations

* Risk detection is **naive keyword-based** (prototype only).
* Summaries/Q\&A use **OpenAI GPT models** when API key is present.
* In Demo Mode (no API), outputs are generated using document snippets.
* This is not a replacement for legal advice. Always consult a lawyer.

---

## 👨‍💻 Author

Built by Jiten Kumar for the **Generative AI Hackathon 2025**.


# 🛠️ How to Run (Quick Recap)

✅ **Local Development**  
1. `python -m venv venv && venv\Scripts\activate` (Windows)  
2. `pip install -r requirements.txt`  
3. `setx OPENAI_API_KEY "sk-xxxxx"` (Windows)
4. `streamlit run app.py`

✅ **Streamlit Cloud Deploy**  
1. Push repo to GitHub.  
2. Deploy new app on Streamlit Cloud.  
3. Add API key in **Secrets**.  
4. Share the **public link** in submission.  

