# Pulsar

Pulsar, [NeutronYazılım](https://neutronyazilim.com.tr) tarafından [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) üzerine QLoRA ile fine-tune edilmiş, TypeScript/Kotlin/Python odaklı bir kod üretim modelidir.

4GB VRAM'lik bir GPU'da (RTX 3050 Laptop) eğitilecek şekilde optimize edilmiştir.

## Kurulum

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

NixOS gibi ortamlarda CUDA/zlib paylaşımlı kütüphaneleri için `env.sh` dosyasını düzenleyip kullanın:

```bash
source env.sh
```

## Pipeline

1. **Veri seti çıkarımı** — yerel repolardan docstring/yorumu olan (veya isimden heuristik üretilen) fonksiyon/sınıfları toplar:
   ```bash
   python3 scripts/extract_dataset.py
   ```
2. **Public veri ile zenginleştirme** — `bigcode/commitpackft`'ten Python/TS/Kotlin örnekleri ekler:
   ```bash
   python3 scripts/augment_dataset.py
   ```
3. **Kimlik verisi** — modelin kendini tanıtma örnekleri:
   ```bash
   python3 scripts/identity_data.py
   ```
4. **Eğitim** — 4-bit QLoRA, sequence packing, 4GB VRAM'e göre ayarlı:
   ```bash
   python3 scripts/train.py
   ```
5. **LoRA merge** (CPU üzerinde, bf16):
   ```bash
   python3 scripts/merge.py
   ```
6. **GGUF dönüşümü** (llama.cpp klonlanmış olmalı, `models/pulsar-coder-1.5b-merged` girdisi ile):
   ```bash
   python3 llama.cpp/convert_hf_to_gguf.py models/pulsar-coder-1.5b-merged \
     --outfile models/pulsar-coder-1.5b-f16.gguf --outtype f16
   ./llama.cpp/build/bin/llama-quantize \
     models/pulsar-coder-1.5b-f16.gguf models/pulsar-coder-1.5b-Q4_K_M.gguf Q4_K_M
   ```

## Test

```bash
python3 scripts/infer.py "Sen kimsin?"
./llama.cpp/build/bin/llama-cli -m models/pulsar-coder-1.5b-Q4_K_M.gguf -cnv
```

## Lisans

Apache License 2.0 — baz model [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) de aynı lisansla dağıtıldığı için bu repo ve türetilmiş ağırlıklar da Apache 2.0 altındadır.

## Notlar

- Model ve GGUF dosyaları bu repoya dahil değildir (`.gitignore`), Hugging Face Hub'da `NeutronYazilim/pulsar-coder-1.5b` altında yayınlanacaktır.
- `data/` klasörü tamamen `.gitignore`'dadır: `dataset.jsonl` bazı özel (private) repolardan çıkarılmış kod içerdiğinden repoya dahil edilmemiştir. Yukarıdaki pipeline adımlarını kendi repolarınızla çalıştırarak kendi veri setinizi üretebilirsiniz.
