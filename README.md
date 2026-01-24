## 🤖 버사타일 (Versatile)

**Korean Bllossom** 모델을 활용한 AI 에이전트 프로젝트

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
2. 프로젝트 루트에 `models` 폴더를 생성해 주세요.
3. 다운로드 받은 모델을 `models` 폴더에 넣어주세요.

#### Configure

`ChatBot` 클래스의 `__init__` 메소드를 본인의 컴퓨터 사양에 맞게 수정해 주세요.

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

- 모델: [QuantFactory / llama-3.2-Korean-Bllossom-3B-GGUF](https://huggingface.co/QuantFactory/llama-3.2-Korean-Bllossom-3B-GGUF) (Q8_0)

---

### ✅ Todo

- [x] Q&A
- [x] 대화 기억
- [x] 사고 모드
- [ ] MCP 코어
