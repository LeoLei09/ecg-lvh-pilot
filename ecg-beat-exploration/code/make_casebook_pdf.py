from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import matplotlib.pyplot as plt
p=Path('outputs/ecg-beat-exploration'); figs=sorted((p/'figures').glob('case_*.png'))
# Existing images contain missing glyphs, so create a clean contact PDF with metadata titles outside image.
out=p/'casebook'/'waveform_casebook.pdf'
with PdfPages(out) as pdf:
 for f in figs:
  im=Image.open(f)
  fig,ax=plt.subplots(figsize=(11.7,8.3)); ax.imshow(im); ax.axis('off'); ax.set_title(f.stem.replace('case_','ECG '),fontsize=11); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
print(out, len(figs))
