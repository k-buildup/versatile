## 🤖 버사타일 (Versatile)

**Korean Bllossom** 모델을 활용한 AI 에이전트 프로젝트

---

### 🔥 Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![LlamaCPP](https://img.shields.io/badge/llama.cpp-%23000000.svg?style=for-the-badge&logo=ollama&logoColor=white)
![LangChain](https://img.shields.io/badge/langchain-%231C3C3C.svg?style=for-the-badge&logo=langchain&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Uvicorn](https://img.shields.io/badge/uvicorn-005571?style=for-the-badge&logo=fastapi)

---

### ✨ Setup

#### Clone Repository

```bash
$ git clone https://github.com/k-buildup/versatile.git
$ cd versatile
```

#### Install Dependencies

```bash
$ pip install -r requirements.txt
```

#### Install Model

1. [Reference](#reference) 섹션을 참고해서 모델을 다운로드해 주세요.
2. `models` 폴더에 다운로드 받은 모델을 넣어주세요.

#### Configure

`agent.py` 에서 `ModelConfig` 클래스를 본인의 컴퓨터 사양에 맞게 수정해 주세요.

- cli 환경
  - `cli.py` 에서 `main` 함수에 `model_path` 를 수정해 주세요.

- server 환경
  - `server.py` 에서 `lifespan` 함수에 `model_path` 를 수정해 주세요.

---

### 🚀 Run

```bash
# cli 환경
$ python cli.py

# server 환경
$ python server.py
# client.html 파일을 열어주세요.
```

---

### 📜 Commands

- `quit`: 종료
- `clear`: 대화 기록 초기화
- `<prompt>`: Q&A
- `@think <prompt>`: 사고 모드
- `@tool <prompt>`: 도구 모드

---

### 📚 Reference

- 모델: [QuantFactory / llama-3-Korean-Bllossom-8B-GGUF](https://huggingface.co/QuantFactory/llama-3-Korean-Bllossom-8B-GGUF) (Q8_0)

---

### ✅ Todo

- [x] Q&A
- [x] 대화 기억
- [x] LangChain 적용
- [x] 사고 모드
- [x] MCP 코어

- [ ] 사고 모드 개선
- [ ] MCP (툴) 개선
