# speech-to-speech-with-lipsync

## 📂 Repository Structure

```bash
📂 demo
├── 1.py
├── 1.txt
├── 3.py
├── Advanced-RVC-Inference
│   ├── config.py
│   ├── configs
│   │   ├── 32k.json
│   │   ├── 40k.json
│   │   └── 48k.json
│   ├── hubert_base.pt
│   ├── lib
│   │   ├── audio.py
│   │   ├── commons.py
│   │   ├── data_utils.py
│   │   └── ...
│   ├── my_convert.py
│   ├── requirements.txt
│   ├── vc_infer_pipeline.py
│   └── weights
│       ├── modi.pth
│       ├── model.index
│       └── ...
├── dubbed_test.mp4
├── lip.py
├── main.py
├── models
│   ├── translation
│   │   └── nllb-en-te
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       └── ...
│   ├── tts
│   │   └── mms-tel
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       └── ...
│   └── whisper
│       └── base.en.pt
├── output.mp4
├── requirment.txt
├── temp
│   └── result.avi
├── test.mp4
├── test_converted.wav
├── test_translated.wav
└── wav2Lip
    ├── checkpoints
    │   ├── wav2lip.pth
    │   └── wav2lip_gan.pth
    ├── face_detection
    │   ├── api.py
    │   ├── core.py
    │   └── sfd
    │       ├── net_s3fd.py
    │       └── s3fd.pth
    ├── inference.py
    ├── models
    │   ├── conv.py
    │   ├── syncnet.py
    │   └── wav2lip.py
    ├── preprocess.py
    └── requirements.txt
