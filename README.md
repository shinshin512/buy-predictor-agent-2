# 🧠 Survey Analysis Agent (Beginner & No-Code Friendly)

This project is an **AI-powered survey analysis tool** with a simple web interface.
You do **not** need programming knowledge — just follow the steps carefully.

⚠️ **Important warning**
Processing can take **hours or even days**, depending on your CSV file size.
Your computer must stay **on and awake** while the agent is running.

---

## 🧩 What You’ll Do (Big Picture)

Inside one terminal, you will:

1. Check Python
2. Create a virtual environment
3. Install required packages
4. Download the AI model (LLaMA 3.1)
5. Run the web interface
6. Upload your CSV file via browser

---

# 🟢 STEP 1: Install Visual Studio Code (Editor)

1. Download and install **Visual Studio Code**
2. Open VS Code

You won’t write code — we just use it to run commands safely.

---

# 🟢 STEP 2: Download This Project

1. Go to this GitHub repository
2. Click **Code → Download ZIP**
3. Unzip the folder
4. Open VS Code
5. Click **File → Open Folder**
6. Select the unzipped project folder

---

# 🟢 STEP 3: Open the Terminal *Inside VS Code*

In VS Code:

* Click **Terminal → New Terminal**

⚠️ From now on, **all commands go here**.

---

# 🟢 STEP 4: Check Python Installation

In the VS Code terminal, run:

```bash
python --version
```

You should see something like:

```text
Python 3.10.x
```

❌ If Python is not found, install **Python 3.10+**, then restart VS Code and try again.

---

# 🟢 STEP 5: Create a Virtual Environment

In the same terminal:

```bash
python -m venv .venv
```

This creates a safe environment for the project.

---

# 🟢 STEP 6: Activate the Virtual Environment

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

If successful, you’ll see:

```text
(.venv)
```

at the start of the terminal line.

---

# 🟢 STEP 7: Install Required Python Packages

Still in the same terminal:

```bash
pip install -r requirements.txt
```

This installs everything the agent needs.

⏳ This may take a few minutes.

---

# 🟢 STEP 8: Install Ollama & Pull the AI Model

### 1️⃣ Install Ollama

Download and install **Ollama** from its official website (https://ollama.com/).

Once installed, **restart VS Code**.

---

### 2️⃣ Pull LLaMA 3.1 (Inside VS Code Terminal)

Back in the VS Code terminal:

```bash
ollama pull llama3.1
```

⏳ This may take several minutes depending on internet speed.

✅ Doing this inside the editor terminal is perfectly fine (and recommended).

---

# 🟢 STEP 9: Start the Web Interface

In the same terminal:

```bash
streamlit run frontend/app.py
```

You’ll see this or something similar:

```text
Local URL: http://localhost:8501
```

---

# 🟢 STEP 10: Use the Web Interface

1. Open the **local URL** in your browser
2. Upload your **CSV survey file**
3. Start the analysis

🎉 You’re officially running the agent.

---

## ⏳ Processing Time (Very Important)

* Small CSV → minutes to hours
* Medium CSV → hours
* Large CSV → **overnight or multiple days**

### ❌ Do NOT:

* Close VS Code
* Close the terminal
* Shut down your computer
* Let your computer sleep

If the terminal is still active, the agent is still working.

---

## 🛑 Stop the Agent (If Needed)

In the VS Code terminal:

```text
CTRL + C
```

---

## ❓ Common Issues

### `python` not found

* Install Python 3.10+
* Restart VS Code

### `ollama` not found

* Make sure Ollama is installed
* Restart VS Code
* Try:

```bash
ollama list
```

### Website doesn’t open

* Confirm you ran:

```bash
streamlit run frontend/app.py
```

* Make sure `(.venv)` is visible

---

## ✅ Final Notes

* Designed for **first-time users**
* One terminal, one workflow
* Long runtime is normal ⏳

If VS Code is open, the terminal shows `(.venv)`, and the browser loads — **you’re doing it right** 🚀
