from typing import Text
from pathlib import Path
import base64

# NOTE: generate additional images with https://latex2png.com/

def _img2b64(img_name: Text) -> bytes:
  p = Path(__file__).parent / "images" / img_name
  with p.open("rb") as infile:
    img_bytes = infile.read()
    return base64.b64encode(img_bytes).decode("utf-8")

img_b64_bayes_transparent = _img2b64("bayes-rule-transparent.png")
img_b64_bayes_white_bg = _img2b64("bayes-rule-white-bg.png")