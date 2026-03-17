"""Quick dataset stats — writes to report.txt"""
import json, os, sys
import numpy as np
from PIL import Image

OUT = "output"
IMG = os.path.join(OUT, "images")
LBL = os.path.join(OUT, "labels")
ANN = os.path.join(OUT, "annotations.json")
RPT = "report.txt"

lines = []
def p(s=""): lines.append(s)

images = sorted([f for f in os.listdir(IMG) if f.endswith(('.jpg','.png'))])
labels = sorted([f for f in os.listdir(LBL) if f.endswith('.txt')])

with open(ANN, "r", encoding="utf-8") as f:
    data = json.load(f)
ann_imgs = data["images"]
anns = data["annotations"]

p("="*60)
p("  DATASET VERIFICATION REPORT")
p("="*60)
p(f"\n[1] COUNTS")
p(f"  Images: {len(images)}")
p(f"  Labels: {len(labels)}")
p(f"  JSON images: {len(ann_imgs)}")
p(f"  JSON annotations: {len(anns)}")
p(f"  Avg annotations/image: {len(anns)/max(1,len(ann_imgs)):.1f}")

# Sizes
widths = [im["width"] for im in ann_imgs]
heights = [im["height"] for im in ann_imgs]
p(f"\n[2] SIZE DIVERSITY")
p(f"  Width range: {min(widths)}-{max(widths)}")
p(f"  Height range: {min(heights)}-{max(heights)}")
p(f"  Unique sizes: {len(set(zip(widths, heights)))}")

# Background diversity - sample 200
p(f"\n[3] BACKGROUND DIVERSITY (200 samples)")
idxs = np.linspace(0, len(images)-1, 200, dtype=int)
mcs, sds, lums = [], [], []
for i in idxs:
    arr = np.array(Image.open(os.path.join(IMG, images[i])))
    mc = arr.mean(axis=(0,1))
    sd = arr.std(axis=(0,1)).mean()
    lum = 0.299*mc[0]+0.587*mc[1]+0.114*mc[2]
    mcs.append(mc); sds.append(sd); lums.append(lum)
mcs=np.array(mcs); sds=np.array(sds); lums=np.array(lums)

dark = (lums<80).sum()
med = ((lums>=80)&(lums<170)).sum()
brt = (lums>=170).sum()
simp = (sds<15).sum()
modr = ((sds>=15)&(sds<40)).sum()
cplx = (sds>=40).sum()

p(f"  Luminance: Dark={dark}({dark*100//200}%) Med={med}({med*100//200}%) Bright={brt}({brt*100//200}%)")
p(f"  Complexity: Simple={simp}({simp*100//200}%) Moderate={modr}({modr*100//200}%) Complex={cplx}({cplx*100//200}%)")
p(f"  R range: {mcs[:,0].min():.0f}-{mcs[:,0].max():.0f}")
p(f"  G range: {mcs[:,1].min():.0f}-{mcs[:,1].max():.0f}")
p(f"  B range: {mcs[:,2].min():.0f}-{mcs[:,2].max():.0f}")
p(f"  Avg StdDev: {sds.mean():.1f}")

# Text
p(f"\n[4] TEXT DIVERSITY")
texts = [a["utf8_string"] for a in anns]
p(f"  Total text boxes: {len(texts)}")
p(f"  Unique texts: {len(set(texts))}")
p(f"  Avg text len: {np.mean([len(t) for t in texts]):.1f} chars")
p(f"  All Devanagari: {all(a['script']=='Devanagari' for a in anns)}")
p(f"  All Hindi: {all(a['language']=='hi' for a in anns)}")

# Bbox
p(f"\n[5] BBOX SANITY")
valid=invalid=0
img_map = {im["id"]: im for im in ann_imgs}
for a in anns:
    x,y,bw,bh = a["bbox"]
    im = img_map.get(a["image_id"])
    if im:
        if x>=0 and y>=0 and bw>0 and bh>0 and x+bw<=im["width"]+5 and y+bh<=im["height"]+5:
            valid+=1
        else:
            invalid+=1
p(f"  Valid: {valid}")
p(f"  Invalid: {invalid}")
p(f"  Rate: {valid*100/max(1,valid+invalid):.1f}%")

# Samples
p(f"\n[6] SAMPLE ANNOTATIONS")
for a in anns[:5]:
    p(f"  id={a['image_id']} bbox={a['bbox']} text=\"{a['utf8_string'][:40]}\"")

# Disk
total = sum(os.path.getsize(os.path.join(IMG,f)) for f in images)
p(f"\n[7] DISK")
p(f"  Images total: {total/(1024*1024):.1f} MB")
p(f"  Avg image: {total/max(1,len(images))/1024:.1f} KB")
p(f"  Annotations: {os.path.getsize(ANN)/(1024*1024):.1f} MB")

# Verdict
p(f"\n{'='*60}")
issues=[]
if len(images)<10000: issues.append(f"Only {len(images)} images")
if invalid>len(anns)*0.05: issues.append(f"High invalid bbox: {invalid}")
if len(set(texts))<1000: issues.append("Low text diversity")
if dark<5 and brt<5: issues.append("Low luminance diversity")
if cplx<5: issues.append("Low texture complexity")
if issues:
    p("  ISSUES:")
    for i in issues: p(f"    - {i}")
else:
    p("  ALL CHECKS PASSED - Dataset ready for OCR training!")
p("="*60)

report = "\n".join(lines)
with open(RPT, "w", encoding="utf-8") as f:
    f.write(report)
print(report)
