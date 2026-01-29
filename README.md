## 🤖 버사타일 (Versatile)

**Korean Bllossom** 모델을 활용한 AI 에이전트 프로젝트

---

### 🔥 Stack

- Python 3.12.7
- llama-cpp-python
- langchain

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

1. [Reference](#reference)를 참고해서 모델을 다운로드해 주세요.
2. 프로젝트 내부에 다운로드 받은 모델을 넣어주세요.
3. `main.py` main 함수에서 `model_path` 변수를 다운로드 받은 모델의 경로로 수정해 주세요.

#### Configure

`main.py` 에서 `ModelConfig` 클래스를 본인의 컴퓨터 사양에 맞게 수정해 주세요.

---

### 🚀 Run

```bash
$ python main.py
```

---

### 📜 Commands

- `quit`: 종료
- `clear`: 대화 기록 초기화
- `<prompt>`: Q&A
- `@think <prompt>`: 사고 모드

---

### 📚 Reference

- 모델: [QuantFactory / llama-3-Korean-Bllossom-8B-GGUF](https://huggingface.co/QuantFactory/llama-3-Korean-Bllossom-8B-GGUF) (Q8_0)

---

### ✅ Todo

- [x] Q&A
- [x] 대화 기억
- [x] LangChain 적용
- [x] 사고 모드
- [ ] MCP 코어
