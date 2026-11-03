"""Create a matching read-only PDF handout from finalized PPTX slide renders.

The PPTX retains editable charts, tables, overview and speaker notes.  This
image-backed PDF intentionally preserves the appearance of that verified PPTX.
It does not generate data, fit models, select loans or change prior deliveries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="01")
    args = parser.parse_args()
    if len(args.revision) != 2 or not args.revision.isdigit():
        parser.error("revision must contain two digits")
    root = Path(__file__).resolve().parents[1]
    task = root / "reports/professor_model/small_penalty"
    build = task / "_build"
    stem = f"professor_small_penalty_{args.revision}"
    output = task / "delivery" / f"{stem}.pdf"
    if output.exists():
        raise FileExistsError(output)
    pptx = output.with_suffix(".pptx")
    receipt = json.loads((build / f"{stem}.validation.json").read_text())
    if hashlib.sha256(pptx.read_bytes()).hexdigest() != receipt["finalSha256"]:
        raise ValueError("PPTX differs from its finalization receipt")
    pages = [build / f"previews_{args.revision}" / f"slide-{i:02d}.png" for i in range(1, 15)]
    for page in pages:
        with Image.open(page) as image:
            if image.size != (1920, 1080):
                raise ValueError(f"unexpected rendered slide size: {page}")
    document = canvas.Canvas(str(output), pagesize=(960, 540), pageCompression=1)
    document.setTitle("Lending Under Imperfect Information: Smaller-Penalty Final Update")
    document.setAuthor("Nguyen Phan Nguyen")
    document.setSubject("Two matched synthetic scenarios, three methods, and historical appendix")
    for page in pages:
        document.drawImage(ImageReader(str(page)), 0, 0, width=960, height=540)
        document.showPage()
    document.save()
    reader = PdfReader(output)
    if len(reader.pages) != 14:
        raise ValueError("PDF must contain fourteen pages")
    for page, source in zip(reader.pages, pages):
        with Image.open(source) as expected:
            if len(page.images) != 1 or page.images[0].image.convert("RGB").tobytes() != expected.convert("RGB").tobytes():
                raise ValueError("PDF page pixels do not match the finalized slide preview")
    print(output)


if __name__ == "__main__":
    main()
