# 🌟prompt-star-fantasy🌟

## What is prompt-star*-fantasy?
A turnbase mmoprg chat based game with 3 wifu. We have chat mode, battle mode and town mode.
wifu can speak ,listen our voice, send images(???) and they can interact to each other (taking long time to inference)

<img src="https://i.imgur.com/2CXyV6p.jpeg" width="1000">

Present,we only have 13 maps(ai generated) and our characters know the context of each map and monster details
<img src="https://i.imgur.com/ASXeZoY.png" width="600">

### ⚔️ Battle Mode Mechanics
In battle mode, the system uses three agents to enhance gameplay:

- Valuate Agent
- Summarizer
- Generator

The boss monster is randomly generated based on the map's characteristics.
<img src="https://i.imgur.com/GdNHPtL.png" width="300">

### 🏘️ town mode

<img src="https://i.imgur.com/SUcF0TM.png" width="300">

## ⚙️ technical
- ASR(api)
- 🐟TTS(fish speech 1.2 [link](https://github.com/fishaudio/fish-speech)) with optimized version for long text
- 🤗 flux1(api) as text to image generation
- RAG use sentence transformer for low level retrieval (faster than langchain)

## 💻 Computer Requirement
### CPU Only
- 12 GB ram or more

### GPU Inference
- VRAM 8 GB or more

### macOs Inference
- m1 base 8 GB or more

## Installation

clone repo & setup project
```bash
git clone https://github.com/Augustus2011/prompt-star-fantasy.git
cd prompt-star-fantast
conda create -n venv python=3.10 -y
conda activate venv
cd modules/fish_speech
pip install -e .

```
###  To run app
1. 🦙ollama  #after install ollama pull model(terminal)
```bash
ollama pull llama3.2 #llama3.1 or qwen2 also work
ollama serve
```
##### (activate env) 
```
conda activate venv
```
2. create .env and add your hf-token to access flux1 api
```bash
.
├── checkpoints
│   ├── file1
│   └── file2
├── images
├── .env
├── stream.py
└── README.md


ex .env
hf_token="hf_*****************"
```
3. run script

```bash
streamlit run stream.py
```



## 📚 CITEME

```bash
@misc{prompt_star_fantasy,
  title        = {Prompt-Star-Fantasy: A Turn-Based MMORPG with AI-Powered Waifus},
  author       = Kun Kerdthaisong,
  year         = {2024},
  howpublished = {\\url{https://github.com/Augustus2011/prompt-star-fantasy}},
  note         = {Interactive AI-powered game with chat, battle, and town modes.}
```


![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/github/license/Augustus2011/prompt-star-fantasy)
[![Star History Chart](https://api.star-history.com/svg?repos=Augustus2011/prompt-star-fantasy&type=Date)](https://star-history.com/#Augustus2011/prompt-star-fantasy&Date)
