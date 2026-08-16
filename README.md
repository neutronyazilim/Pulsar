# Pulsar

Pulsar, [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) üzerine QLoRA ile fine-tune edilmiş, TypeScript, Kotlin ve Python odaklı bir kod üretim modelidir. 4GB VRAM'lik bir GPU'da (RTX 3050 Laptop) eğitilecek şekilde optimize edilmiştir.

## Model

| | |
|---|---|
| Ana model (safetensors) | [tuxkt/pulsar-coder-1.5b](https://huggingface.co/tuxkt/pulsar-coder-1.5b) |
| GGUF (Q4_K_M, llama.cpp) | [tuxkt/pulsar-coder-1.5b-GGUF](https://huggingface.co/tuxkt/pulsar-coder-1.5b-GGUF) |
| Baz model | [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) |

### Hızlı kullanım

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tuxkt/pulsar-coder-1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [
    {"role": "system", "content": "You are an expert software engineer. Given an instruction, write the requested code."},
    {"role": "user", "content": "Write a TypeScript function `debounce` that delays invoking a function until after a wait time."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=300, repetition_penalty=1.15)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Ya da `llama.cpp` ile GGUF sürümünü çalıştır:

```bash
hf download tuxkt/pulsar-coder-1.5b-GGUF pulsar-coder-1.5b-Q4_K_M.gguf --local-dir .
llama-cli -m pulsar-coder-1.5b-Q4_K_M.gguf -cnv --repeat-penalty 1.15 --temp 0.3
```

## Kendi kopyanı eğitmek

### Kurulum

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

NixOS gibi ortamlarda CUDA/zlib paylaşımlı kütüphaneleri için `env.sh` dosyasını düzenleyip kullanın:

```bash
source env.sh
```

### Pipeline

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
4. **Eğitim** — 4-bit QLoRA, sequence packing, 4GB VRAM'e göre ayarlı, 3 epoch:
   ```bash
   python3 scripts/train.py
   ```
5. **LoRA merge** (CPU üzerinde, bf16):
   ```bash
   python3 scripts/merge.py
   ```
6. **GGUF dönüşümü** (llama.cpp klonlanmış ve derlenmiş olmalı):
   ```bash
   git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
   cmake -B llama.cpp/build -S llama.cpp -DCMAKE_BUILD_TYPE=Release
   cmake --build llama.cpp/build --target llama-quantize llama-cli -j$(nproc)

   python3 llama.cpp/convert_hf_to_gguf.py models/pulsar-coder-1.5b-merged \
     --outfile models/pulsar-coder-1.5b-f16.gguf --outtype f16
   ./llama.cpp/build/bin/llama-quantize \
     models/pulsar-coder-1.5b-f16.gguf models/pulsar-coder-1.5b-Q4_K_M.gguf Q4_K_M
   ```

### Test

```bash
python3 scripts/infer.py "Sen kimsin?"
./llama.cpp/build/bin/llama-cli -m models/pulsar-coder-1.5b-Q4_K_M.gguf -cnv --repeat-penalty 1.15 --temp 0.3
```

`--repeat-penalty` olmadan (özellikle `--temp 0` greedy modda) kısa cevaplarda tekrar döngüsüne girebilir; bu bayrak ile risk azalır.

## Eğitim detayları

- ~1450 örnek: 9'u public bir repodan (`claws-mouse-linux`), 1350'si [bigcode/commitpackft](https://huggingface.co/datasets/bigcode/commitpackft) (Python/TypeScript/Kotlin), 92'si elle yazılmış kimlik verisi.
- Private/doğrulanamayan repolardan hiçbir kod eğitime dahil edilmemiştir.
- QLoRA: 4-bit NF4, r=16, alpha=32, 7 hedef modül, 3 epoch, ~44 dk (RTX 3050 Laptop, 4GB VRAM).

## Lisans

Apache License 2.0 — baz model [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) de aynı lisansla dağıtıldığı için bu repo ve türetilmiş ağırlıklar da Apache 2.0 altındadır.

## Notlar

- Model ve GGUF dosyaları bu repoya dahil değildir (`.gitignore`); yukarıdaki Hugging Face linklerinden indirilebilir.
- `data/` klasörü tamamen `.gitignore`'dadır: `dataset.jsonl` bazı özel (private) repolardan çıkarılmış kod içerebileceğinden repoya dahil edilmemiştir. Yukarıdaki pipeline adımlarını kendi repolarınızla çalıştırarak kendi veri setinizi üretebilirsiniz.
