# EMMA: Event-Driven Multimodal Memory System with Multi-Agent Retrieval

EMMA builds an event-driven memory hierarchy for long videos and uses
multi-round retrieval to answer video questions. The public release has two
independent stages:

```
Stage 1 · Memory Building   video → ASR → STM / MTM / Event Table
Stage 2 · KQA               memory + question → multi-round retrieval → answer
```

The reference configuration uses local Qwen3-VL-8B-Instruct for visual memory
construction and raw-video fallback; DeepSeek-V4-Flash for memory text
reasoning, retrieval planning, and validation; `text-embedding-3-large` for
text retrieval; and DeepSeek R1 for final answers.

## Requirements

- Python 3.10+
- `ffmpeg` on `PATH` (for example, `apt install ffmpeg`)
- API access to an OpenAI-compatible DeepSeek and text-embedding endpoint

Install a compatible PyTorch build, then install the remaining packages:

```bash
pip install -r requirements.txt
```

Download these local models separately; model weights, videos, and outputs are
not included in this repository.

| Model | Role | Required format |
|---|---|---|
| Qwen3-VL-8B-Instruct | VLM memory construction and video fallback | Hugging Face checkpoint |
| whisper-large-v3-turbo | ASR | CTranslate2 / faster-whisper checkpoint |

To convert an existing Hugging Face Whisper checkpoint once (CPU is sufficient):

```bash
ct2-transformers-converter --model /path/to/whisper-large-v3-turbo --output_dir /path/to/whisper-large-v3-turbo-ct2 --quantization float16 --copy_files tokenizer.json preprocessor_config.json
```

## Configuration

Copy the examples; never commit the resulting `.env` files.

```bash
cp memory/.env.example memory/.env
cp KQA/project/.env.example KQA/project/.env
```

Set your own credentials and OpenAI-compatible endpoint in both files.  The
default memory text model is `deepseek-v4-flash`.  The public KQA configuration
is [`KQA/project/configs/deepseek_v4_flash_r1.yaml`](KQA/project/configs/deepseek_v4_flash_r1.yaml);
it uses `deepseek-v4-flash`, `text-embedding-3-large`, and
`ark-deepseek-r1-250528` respectively.  Adjust the model identifiers only if
your provider uses different aliases.

## Stage 1 — Memory building

`memory/run_memory.py` is the supported entry point. It first runs
faster-whisper/CTranslate2 ASR, then lets that process exit before starting a
local Qwen vLLM server. Qwen remains resident across all videos, so the report
separates startup time from per-video steady-state time.

```bash
python memory/run_memory.py \
  --input-dir /path/to/videos \
  --output-root /path/to/memory-results \
  --asr-model-dir /path/to/whisper-large-v3-turbo-ct2 \
  --qwen-model /path/to/Qwen3-VL-8B-Instruct \
  --device-id 0 \
  --fail-fast
```

For a reproducible subset, create a text file containing one absolute video
path (or one video filename) per line and pass `--video-list /path/to/list.txt`.
Use `--rerun-asr` when ASR should be measured again rather than reused.

Outputs for each video are written to:

```
{output_root}/{video_name}/outputs/
├── asr_segments.json
├── chapter_segmentation.json
├── short_term.json
├── medium_term.json
└── event_table.json
```

Timing reports are written to
`{output_root}/_runs/<timestamp>/summary.json` and
`per_video_timings.json`.  `memory_build` and each `videos[].seconds` value
exclude ASR and Qwen server startup, which is the useful production latency
when Qwen is already resident.

## Stage 2 — KQA

The KQA runner starts one Qwen vLLM server for RAW_VIDEO fallback and runs
questions sequentially. It reads the Stage 1 output tree via `--results-root`.

```bash
python KQA/project/run_batch.py \
  --videomme-json /path/to/videomme.json \
  --results-root /path/to/memory-results \
  --raw-video-root /path/to/videos \
  --qwen-model /path/to/Qwen3-VL-8B-Instruct \
  --device-id 0
```

Use `--question-id <id>` or `--max-questions 1` for a smoke test.  Results,
traces, errors, and a timing summary are written under
`KQA/project/outputs/batch_<timestamp>/` unless `--out-root` is supplied.

## Retrieval policy in the reference configuration

EMMA uses a grounded coarse-to-fine route: MTM localization, then STM or
RAW_VIDEO evidence before early termination.  After two stalled or insufficient
rounds, it forces the escalation `MTM → STM → RAW_VIDEO`; a no-evidence run
uses a closest-candidate fallback over 15 MTM and 25 STM memories.  This is the
core retrieval-control strategy carried into the public reference configuration.

## Project structure

```
EMMA/
├── memory/
│   ├── ASR.py                 # faster-whisper / CTranslate2 ASR
│   ├── run_memory.py          # supported memory runner
│   ├── AIO.py                 # memory construction pipeline
│   ├── chapter.py             # chapter segmentation
│   └── .env.example
├── KQA/project/
│   ├── run_batch.py           # supported KQA runner
│   ├── configs/deepseek_v4_flash_r1.yaml
│   ├── src/                   # retrieval agents, pipeline, and utilities
│   └── .env.example
├── qwen_vl_utils.py
├── requirements.txt
└── LICENSE                    # Apache-2.0
```

## Notes

- `.env`, datasets, model checkpoints, generated outputs, and logs are ignored
  by Git.  Scan your working tree before every public push.
- The supplied runner releases Whisper before starting Qwen to reduce model
  residency pressure.
