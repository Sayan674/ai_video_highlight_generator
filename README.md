# 🎬 AI Video Highlight Generator

> Automatically transform any long-form video into a polished, captioned highlight reel — powered by ResNet-18, CLIP, and OpenAI Whisper.

---

## 📌 What It Does

The AI Video Highlight Generator is a fully automated video intelligence pipeline that:

1. **Extracts** every frame from your source video
2. **Scores** each frame using a dual-model approach (ResNet-18 for visual richness + CLIP for semantic relevance)
3. **Selects** the single best frame per second of footage
4. **Stitches** selected frames into a compact highlight clip
5. **Transcribes** the original audio using OpenAI Whisper
6. **Overlays** time-aligned captions and applies a brightness/contrast enhancement
7. **Exports** the final polished reel — ready to share

---

## 🌍 Real-World Use Cases

| Scenario | How It Helps |
|---|---|
| 📹 Content Creators | Auto-generate YouTube Shorts or Instagram Reels from long videos |
| 🎓 E-Learning Platforms | Extract the most informative moments from lecture recordings |
| 🏟️ Sports & Events | Create instant highlight reels from raw footage |
| 📰 Journalism & Media | Surface key moments from hours of press conference footage |
| 🎥 Film Pre-Production | Quick rough-cut previews for editors to review |
| 🔒 Security / Surveillance | Flag high-activity moments in long CCTV recordings |

---

## ✨ Features

- **Dual-model frame scoring**: Combines low-level visual complexity (ResNet-18 feature activation) with high-level semantic relevance (CLIP image-text similarity) for smarter frame selection
- **Per-second temporal grouping**: Guarantees temporal coverage — no section of your video is skipped
- **Automatic speech transcription**: OpenAI Whisper generates accurate, time-aligned subtitles
- **Caption overlay**: Text rendered on a clean dark bar for maximum legibility
- **Visual enhancement**: Mild contrast and brightness boost for a polished look
- **GPU acceleration**: Automatically uses CUDA if available, falls back to CPU
- **Fully configurable**: All parameters (weights, FPS, captions, models) exposed in `config.yaml`
- **Batch processing**: Process entire folders of videos with a single command
- **Clean architecture**: Modular `src/` layout ready for extension or integration

---

## 🧠 Tech Stack

| Component | Technology |
|---|---|
| Frame extraction | OpenCV |
| Visual scoring | ResNet-18 (torchvision) |
| Semantic scoring | CLIP ViT-B/32 (OpenAI) |
| Speech transcription | Whisper (OpenAI) |
| Video I/O | MoviePy + OpenCV |
| Deep learning backend | PyTorch |
| Config management | PyYAML |
| Testing | pytest |

---

## ⚙️ Installation

### Prerequisites

- Python 3.9 or higher
- pip
- *(Optional but recommended)* NVIDIA GPU with CUDA for faster processing

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ai-video-highlight-generator.git
cd ai-video-highlight-generator

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Pre-download all model weights before first run
python scripts/download_models.py
```

> **Note:** CLIP is installed directly from GitHub. Ensure `git` is available on your system.  
> On first run, model weights (~500 MB total) are automatically downloaded and cached.

---

## 🚀 How to Run

### Basic Usage

```bash
# Place your video as input.mp4 in the project root, then:
python run.py
```

### Custom Input / Output

```bash
python run.py --input /path/to/your/video.mp4 --output /path/to/output_reel.mp4
```

### Custom Config

```bash
python run.py --config my_config.yaml
```

### Batch Processing

```bash
# Process a list of videos
python scripts/batch_process.py --inputs video1.mp4 video2.mp4 video3.mp4

# Process all .mp4 files in a folder
python scripts/batch_process.py --inputs_dir ./raw_videos/
```

### Run Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 📂 Project Structure

```
ai-video-highlight-generator/
│
├── run.py                        ← Main entry point
├── config.yaml                   ← All configurable parameters
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── core/
│   │   ├── frame_extractor.py    ← OpenCV frame extraction
│   │   ├── model_loader.py       ← ResNet-18 + CLIP loader
│   │   ├── frame_scorer.py       ← Dual-model scoring logic
│   │   └── frame_selector.py     ← Per-second best-frame selection
│   │
│   ├── pipelines/
│   │   ├── video_assembler.py    ← Stitch frames into silent clip
│   │   └── caption_pipeline.py   ← Whisper transcription + caption overlay
│   │
│   └── utils/
│       └── cleanup.py            ← Intermediate file cleanup
│
├── scripts/
│   ├── download_models.py        ← Pre-download all model weights
│   └── batch_process.py          ← Process multiple videos at once
│
├── tests/
│   └── test_pipeline.py          ← Unit tests (no GPU required)
│
├── models/                       ← Model cache info (weights auto-downloaded)
└── assets/
    ├── sample_input/             ← Place your input video here
    └── sample_output/            ← Generated reels saved here
```

---

## 🎛️ Configuration Reference (`config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `input_video` | `input.mp4` | Path to source video |
| `final_video_path` | `output/final_ai_reel.mp4` | Path for output highlight reel |
| `clip_model_name` | `ViT-B/32` | CLIP variant (`ViT-B/32` or `ViT-L/14`) |
| `clip_text_prompts` | `["interesting moment", ...]` | Semantic prompts for CLIP scoring |
| `whisper_model_size` | `base` | Whisper size: `tiny` / `base` / `small` / `medium` / `large` |
| `resnet_score_weight` | `0.5` | Weight for ResNet visual richness score |
| `clip_score_weight` | `0.5` | Weight for CLIP semantic score |
| `highlight_fps` | `2` | FPS of the output highlight reel |
| `contrast_alpha` | `1.2` | Contrast multiplier (1.0 = no change) |
| `brightness_beta` | `30` | Brightness offset (0 = no change) |
| `cleanup_intermediates` | `true` | Auto-delete temp frames/audio after run |

---

## 📥 Input / Output

### Input
- Any standard video file supported by OpenCV (`.mp4`, `.avi`, `.mov`, `.mkv`)
- Audio track is optional; if absent, captions are silently skipped

### Output
- **`output/final_ai_reel.mp4`** — the polished highlight reel with captions

### Intermediate Files (auto-cleaned by default)
| File/Folder | Purpose |
|---|---|
| `output/frames/` | All extracted raw frames |
| `output/selected_frames/` | Best-scored frames (one per source second) |
| `output/temp_highlight.mp4` | Silent highlight clip |
| `output/audio.wav` | Extracted audio for Whisper |

---

## 🔮 Future Improvements

- [ ] **Scene-change detection** — use histogram difference to avoid selecting near-duplicate frames
- [ ] **Adaptive highlight duration** — user-specified target reel length instead of fixed FPS
- [ ] **Multi-language captions** — Whisper supports 99 languages; surface language selection in config
- [ ] **Audio-visual sync** — stitch original audio segments into the reel instead of plain captions
- [ ] **Web UI** — Gradio or Streamlit interface for non-technical users
- [ ] **YouTube / Shorts export** — auto-crop to 9:16 aspect ratio with face-tracking
- [ ] **Emotion scoring** — integrate DeepFace to weight frames with strong emotional expression higher
- [ ] **Docker support** — containerised deployment for reproducible cross-platform runs

---

## 🙋 Author

**Sayan Mukherjee**  
B.Tech Mechanical Engineering, NIT Jamshedpur (2024–2028)  
[LinkedIn](https://www.linkedin.com/in/sayan-mukherjee-5b0654374) · [GitHub](https://github.com/your-username)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

> *Built as a portfolio project demonstrating applied deep learning, computer vision, and NLP engineering.*
