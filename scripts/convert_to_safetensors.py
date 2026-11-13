#!/usr/bin/env python3
"""Save an HF masked-LM checkpoint that ships only pytorch_model.bin as safetensors.

transformers >= 4.50 refuses to torch.load a .bin under torch < 2.6 (CVE-2025-32434),
and the `retriever` env pairs transformers 5.4 with torch 2.4. naver/splade-v3 (and
SPLADE++) publish only .bin, so convert once in the `searchr1` env (transformers
4.47, loads with weights_only=True) and point the SPLADE paths at the output dir.

    conda activate searchr1 && source run/env.sh
    python scripts/convert_to_safetensors.py naver/splade-v3 \
        $STORE_ROOT/models/splade-v3
"""
import sys

from transformers import AutoModelForMaskedLM, AutoTokenizer

src, dst = sys.argv[1], sys.argv[2]
model = AutoModelForMaskedLM.from_pretrained(src)
model.save_pretrained(dst, safe_serialization=True)
AutoTokenizer.from_pretrained(src).save_pretrained(dst)
with open(f'{dst}/SOURCE', 'w') as f:
    f.write(f'converted from {src} by scripts/convert_to_safetensors.py\n')
print(f'{src} -> {dst}')
