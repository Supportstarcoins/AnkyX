# DeepSeek OCR weights

Put required files into this directory:
- `config.json`
- `model.safetensors`
- `tokenizer.json`

## Enable
1. Open app settings.
2. Set OCR provider to `deepseek`.
3. Save settings.

## Test OCR
Use Test OCR button (or call):
```python
from deepseek_ocr import test_ocr
ok, result = test_ocr("sample.png")
print(ok, result)
```

If files are missing, runner returns a clear offline error.
