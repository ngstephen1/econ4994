"""Export eight verified slide renders without PDF font substitution.

The PPTX retains native editable objects. The companion PDF deliberately uses
144-dpi slide renders, preserving exactly the reviewed appearance and signs.
No experiment or scientific result is read/modified by this exporter.
"""
import argparse
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader
from PIL import Image

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():
        raise FileExistsError('Use a new revision PDF filename: '+str(out))
    deck_dir=Path(__file__).resolve().parent
    c=canvas.Canvas(str(out),pagesize=(960,540),pageCompression=1)
    c.setTitle('Lending Under Imperfect Information: Research Progress Through Step 8B')
    c.setAuthor('Nguyen Phan Nguyen')
    c.setSubject('Eight slides. Conditional noise-only synthetic lending research.')
    for i in range(1,9):
        path=deck_dir/f'previews/slide-{i:02d}.png'
        with Image.open(path) as im:
            assert im.size==(1920,1080), (path,im.size)
        c.drawImage(ImageReader(str(path)),0,0,width=960,height=540)
        c.showPage()
    c.save()
    assert len(PdfReader(out).pages)==8
    print('Saved eight-page appearance-matched PDF:',out)

if __name__=='__main__':main()
