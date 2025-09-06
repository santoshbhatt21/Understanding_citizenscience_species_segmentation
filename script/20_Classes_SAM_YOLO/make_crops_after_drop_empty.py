# save as make_crops.py and run once
from pathlib import Path
import cv2, numpy as np

ROOT = Path(r"E:/Santosh_master_thesis/DATA_YOLO11_strict_CAM_setting")

def make_split(split, outsize=768, margin=0.35):
    imdir = ROOT / "images" / split
    lbdir = ROOT / "labels" / split
    dst_im = ROOT / f"images_{split}_crops"
    dst_lb = ROOT / f"labels_{split}_crops"
    dst_im.mkdir(parents=True, exist_ok=True)
    dst_lb.mkdir(parents=True, exist_ok=True)

    def poly_to_bbox(poly, w, h):
        xs = np.array(poly[0::2])*w; ys = np.array(poly[1::2])*h
        return xs.min(), ys.min(), xs.max(), ys.max()

    n_im = 0
    for p in imdir.rglob("*"):
        if p.suffix.lower() not in {".jpg",".jpeg",".png",".bmp"}: 
            continue
        rel = p.relative_to(imdir)
        lab = (lbdir / rel).with_suffix(".txt")
        if not lab.exists() or lab.stat().st_size==0:
            continue
        img = cv2.imread(str(p)); 
        if img is None: continue
        h,w = img.shape[:2]
        lines = [ln.strip() for ln in open(lab, "r", encoding="utf-8") if ln.strip()]
        # make one crop per object (cap to 8)
        for k,ln in enumerate(lines[:8]):
            parts = ln.split()
            cls = int(parts[0]); poly = list(map(float, parts[1:]))
            if len(poly) < 6: continue
            x1,y1,x2,y2 = poly_to_bbox(poly, w, h)
            cx,cy = (x1+x2)/2,(y1+y2)/2
            side = int(max(x2-x1, y2-y1)*(1.0+margin))
            xA = max(0, int(cx - side/2)); yA = max(0, int(cy - side/2))
            xB = min(w, xA+side); yB = min(h, yA+side)
            xA = max(0, xB-side); yA = max(0, yB-side)
            crop = img[yA:yB, xA:xB]
            if crop.size==0: continue
            crop = cv2.resize(crop, (outsize, outsize))
            # remap all polys that land in crop
            new = []
            for l in lines:
                ps = l.split(); c = int(ps[0]); pp = list(map(float, ps[1:]))
                xs = np.array(pp[0::2])*w - xA
                ys = np.array(pp[1::2])*h - yA
                inside = (xs>=0)&(ys>=0)&(xs<=xB-xA)&(ys<=yB-yA)
                if inside.mean()<0.3: 
                    continue
                xs = np.clip(xs,0,xB-xA)/(xB-xA)
                ys = np.clip(ys,0,yB-yA)/(yB-yA)
                flat = []
                for xi,yi in zip(xs,ys): flat += [f"{xi:.6f}", f"{yi:.6f}"]
                new.append(" ".join([str(c)] + flat))
            if not new: continue
            base = (p.stem + f"_crop{k:02d}")
            (dst_im / rel.parent).mkdir(parents=True, exist_ok=True)
            (dst_lb / rel.parent).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str((dst_im/rel.parent)/(base+".jpg")), crop)
            open(str((dst_lb/rel.parent)/(base+".txt")), "w", encoding="utf-8").write("\n".join(new))
            n_im += 1
    print(f"{split}: wrote {n_im} crop images.")

make_split("train")
make_split("val")
