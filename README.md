# document_OCR — PDF OCR 엔진 비교 (exact match / CER / WER)

`data/` 의 PDF 를 **PaddleOCR-VL 1.6**, **Tesseract**, **EasyOCR** 로 읽고
정답 텍스트와 비교해 `result/result.csv` 를 만든다.

```
src/
  run_benchmark.py     실행 진입점 (OCR -> 채점 -> CSV)
  make_gt.py           PDF 텍스트 레이어로 정답 초안 만들기
  engines/             엔진 세 개 (공통 인터페이스, import 는 지연 로딩)
  pdf_io.py            PDF -> 페이지 이미지 렌더링
  normalize.py         채점 전 정규화
  metrics.py           레벤슈타인 기반 exact match / CER / WER
  languages.py         폴더 이름(ko, ch, ...) -> 엔진 언어 설정
data/                  언어 폴더별 PDF (ch/ en/ ko/ pil/ ru/ uz/ vn/) + 정답 텍스트
result/                result.csv, result_summary.csv, pred/<engine>/*.txt
```

## 1. 빠른 실행

```bash
python src/run_benchmark.py --device gpu:0          # 세 엔진 전부
python src/run_benchmark.py --engines tesseract --device cpu --limit 3
python src/run_benchmark.py --metrics-only --normalize nospace   # 채점만 다시
```

## 2. 정답 텍스트 두는 곳

`data/ko/2025고합394_판결문.pdf` 의 정답이라면 아래 중 아무 데나 두면 자동으로 찾는다.

| 배치 | 경로 |
| --- | --- |
| PDF 옆 | `data/ko/2025고합394_판결문.txt` |
| 하위 구조 미러링 | `data/gt/ko/2025고합394_판결문.txt` |
| 한 폴더에 몰기 | `data/gt/2025고합394_판결문.txt` |
| 직접 지정 | `--gt-dir <폴더>` |

확장자는 `.txt`, `.gt.txt`, `.md`, `.json` 을 본다. JSON 은 `{"text": "..."}` 또는
`{"pages": [...]}` 형태를 읽는다. 정답이 없는 문서는 OCR 결과만 저장하고 지표 칸을 비운다.

정답이 아직 없다면 초안을 뽑아 쓸 수 있다. `data/` 의 PDF 는 대부분 스캔본이 아니라
텍스트가 심긴 전자문서라 그대로 추출된다. 스캔본은 `--report` 가 따로 알려준다
(예: `en/23-cr-99_indictment.pdf` 는 텍스트 레이어가 없어 직접 만들어야 한다).

```bash
python src/make_gt.py --report      # 어떤 파일이 뽑히는지 먼저 확인
python src/make_gt.py               # data/gt/ 에 초안 저장
```

**초안은 반드시 눈으로 고쳐라.** 표·2단 편집은 읽는 순서와 다르게 뽑히고, 머리말과
쪽번호가 섞여 들어온다. 안 고치고 그대로 쓰면 OCR 성능이 아니라 "PDF 텍스트 레이어와
얼마나 같은가"를 재게 된다.

## 3. 설치

```bash
pip install -r requirements.txt

# tesseract 본체 (pytesseract 는 껍데기다)
sudo apt-get install -y tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng

# PaddleOCR-VL 1.6 — CUDA 버전을 타므로 requirements 에 안 넣었다
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install -U "paddleocr[doc-parser]>=3.6.0"
```

윈도우에서 tesseract 를 쓸 때는 설치 경로를 알려준다.
`--tesseract-cmd "C:/Program Files/Tesseract-OCR/tesseract.exe"`

## 4. 지표

| 지표 | 정의 | 비고 |
| --- | --- | --- |
| exact match | 정규화한 문서 전체가 글자 하나까지 같으면 1 | 문서 단위라 대개 0 이 많이 나온다 |
| CER | (치환+삭제+삽입) / 정답 **글자** 수 | 낮을수록 좋다 |
| WER | (치환+삭제+삽입) / 정답 **단어** 수 | 공백으로 자른 토큰 기준 |

`result_summary.csv` 에는 두 가지 평균이 같이 들어간다.

- `*_micro` — 편집 횟수 합 / 정답 길이 합. 긴 문서에 가중치가 실린다. **엔진 비교용은 이쪽.**
- `*_macro` — 문서별 비율의 평균. 문서 하나를 한 표로 센다.

`--normalize` 로 채점 기준을 바꾼다. 지표는 여기에 크게 흔들리니 결과를 공유할 때
어떤 값으로 쟀는지 같이 적어라 (CSV 의 `normalize` 열에 박힌다).

| 값 | 하는 일 | 쓰는 상황 |
| --- | --- | --- |
| `none` | 앞뒤 공백만 제거 | 원문 그대로 비교 |
| `basic` (기본) | NFKC + 따옴표/붙임표 통일 + 공백 1칸 축약 | 일반적인 비교 |
| `lower` | basic + 소문자화 | 영문 대소문자를 안 볼 때 |
| `nospace` | basic + 공백 전부 제거 | 한국어 띄어쓰기를 안 볼 때 (**CER 만 보라**. 문서가 한 단어가 돼 WER 이 무의미해진다) |
| `nopunct` | lower + 문장부호 제거 | 문장부호 차이를 안 볼 때 |

`data/` 가 언어 폴더로 나뉘어 있으므로 요약에 폴더별(`group`) 행이 함께 나온다.

## 4-1. 언어 설정

Tesseract 와 EasyOCR 은 어떤 언어를 읽을지 미리 정해줘야 한다. 폴더 이름을 언어 힌트로
삼아 문서마다 자동으로 바꿔 끼운다 (`src/languages.py`, CSV 의 `lang` 열에 기록된다).

| 폴더 | tesseract | easyocr |
| --- | --- | --- |
| `en` | `eng` | `en` |
| `ko` | `kor+eng` | `ko`, `en` |
| `ch` | `chi_sim+eng` | `ch_sim`, `en` |
| `ru` | `rus+eng` | `ru`, `en` |
| `vn` | `vie+eng` | `vi`, `en` |
| `uz` | `uzb+eng` | `uz`, `en` |
| `pil` | `tgl+eng` | `tl`, `en` |

표에 없는 폴더는 영어로 떨어진다. 바꾸려면 JSON 을 만들어 넘긴다.

```bash
python src/run_benchmark.py --lang-map my_langs.json
# {"pil": {"tesseract": "tgl+eng", "easyocr": ["tl", "en"]}}
```

- tesseract 는 언어별 traineddata 를 따로 깔아야 한다.
  `apt-get install -y tesseract-ocr-{kor,chi-sim,rus,vie,uzb,tgl}`
- EasyOCR 은 Reader 하나에 한국어와 중국어를 같이 못 넣는다. 그래서 그룹이 바뀔 때마다
  Reader 를 새로 만들고 이전 것은 버린다. 문서를 경로 순으로 돌기 때문에 교체는 그룹 수만큼만 일어난다.
- PaddleOCR-VL 은 다국어 모델 하나가 전부 처리해서 언어 설정이 없다.

## 5. 출력

| 파일 | 내용 |
| --- | --- |
| `result/result.csv` | 문서 × 엔진 한 줄. 지표, 소요 시간, 쪽 수, 에러 메시지 |
| `result/result_summary.csv` | 엔진별 집계 (`group=ALL` 전체 + 언어 폴더별 한 줄씩) |
| `result/pred/<engine>/<문서>.txt` | 인식 원문. 눈으로 확인하거나 재채점할 때 쓴다 |

문서 한 줄을 쓸 때마다 flush 하므로 중간에 끊겨도 그때까지의 결과는 남는다.

## 6. GPU 서버에서 돌릴 때

`.gitignore` 는 파이썬 캐시만 무시한다. `data/` 와 `result/` 는 일부러 커밋한다. GPU 서버에서 clone 만 하면 바로 돌아가고, 나온 CSV 를 그대로 받아오기 위해서다.

```bash
git clone <repo> && cd document_OCR
# (설치는 3번)
python src/run_benchmark.py --device gpu:0 --dpi 200 2>&1 | tee result/run.log
```

- 엔진을 바깥 루프에 두고 하나씩 올렸다 내린다. VRAM 에 모델이 동시에 올라가지 않는다.
- Tesseract 는 GPU 를 안 쓴다. `--device` 를 줘도 CPU 로 돌고 CSV 에도 `cpu` 로 적힌다.
- 중간에 끊겼으면 `--resume` 을 붙여 다시 돌린다. 이미 저장된 예측은 건너뛴다.
- 한 문서가 죽어도 나머지는 계속 돈다. 실패는 `error` 열에 남고 집계에서 빠진다.
- 엔진 하나가 아예 설치 안 됐으면 그 엔진만 건너뛰고 나머지로 CSV 를 만든다.

## 7. 엔진별 옵션

**PaddleOCR-VL** — 기본은 PDF 를 파이프라인에 그대로 넘기고(`--paddle-input pdf`),
레이아웃 블록의 텍스트를 이어 붙여 인식 결과로 본다(`--paddle-text blocks`).
`--paddle-text markdown` 으로 바꾸면 표를 마크다운으로 받는데, 정답이 평문이면
CER 이 불리해진다(기본적으로 채점 시 마크다운 기호는 제거한다. `--keep-markdown` 으로 끌 수 있다).
나머지 두 엔진과 완전히 같은 픽셀로 재고 싶으면 `--paddle-input images`.
vLLM 서버로 띄워 쓸 거면 `--paddle-backend vllm-server --paddle-server-url http://localhost:8000/v1`.

**Tesseract** — 언어는 폴더별 표를 따른다(4-1). `--tesseract-lang` 은 표에 없는 폴더의
기본값이고, `--tesseract-config "--oem 3 --psm 3"` 로 psm 을 바꾼다.

**EasyOCR** — 언어는 폴더별 표를 따른다(4-1). `--easyocr-langs` 는 그 표가 없을 때의
기본값이다. 기본은 `paragraph=True` 로 줄을 문단으로 묶는다.
`--easyocr-no-paragraph` 로 줄 단위 출력을 받는다.
