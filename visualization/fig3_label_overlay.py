"""Draw residue labels on the Figure-3 panels rendered by ``fig3_render.py``.

``fig3_render.py`` writes ``panel_nolabel.png`` (structure only) plus
``labels.json`` holding the object-space -> pixel matrix ``A``, the pixel
position ``p0`` of the pocket centre and the residues to annotate.  This script
projects the C-alpha atoms with ``A``, lays the labels out radially around the
pocket with a collision-free 2-D packing step, and draws them together with a
short leader line, writing ``panel.png``.

Usage:
    python fig3_label_overlay.py --fig3_dir ../paper/fig3
"""
import os
import json
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
]
FONT_SIZE = 78        # px, at the 1400 px panel size
RADIAL = 0.16         # initial label distance from the pocket centre (fraction of the panel)
PAD = 10              # px padding inside a label box (at 1400 px)
GAP = 12              # px gap enforced between boxes (at 1400 px)


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    raise RuntimeError('no usable TTF font found')


def text_box(draw, text, font):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l, b - t


def place_labels(draw, font, texts, anchors, p0, size, radius, pad, gap):
    """Pack the label boxes radially around the pocket without overlapping."""
    pos, boxes = [], []
    for text, base in zip(texts, anchors):
        d = base - p0
        n = np.linalg.norm(d) or 1.0
        pos.append(base + radius * size * d / n)
        w, h = text_box(draw, text, font)
        boxes.append(np.array([w + 2 * pad, h + 2 * pad], dtype=float))

    rmin = radius * size
    for _ in range(800):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                sx = 0.5 * (boxes[i][0] + boxes[j][0]) + gap
                sy = 0.5 * (boxes[i][1] + boxes[j][1]) + gap
                ox = sx - abs(dx)
                oy = sy - abs(dy)
                if ox <= 0 or oy <= 0:
                    continue
                if ox / sx <= oy / sy:
                    step = np.array([np.sign(dx) or 1.0, 0.0]) * ox * 0.5
                else:
                    step = np.array([0.0, np.sign(dy) or 1.0]) * oy * 0.5
                pos[i] = pos[i] + step
                pos[j] = pos[j] - step
                moved = True
        for k, p in enumerate(pos):
            d = p - p0
            n = np.linalg.norm(d)
            if n < rmin:
                pos[k] = p0 + (d / n if n > 1e-9 else np.array([1.0, 0.0])) * rmin
                moved = True
        for k, p in enumerate(pos):
            half = boxes[k] / 2.0
            p[0] = min(max(p[0], half[0] + 30), size - half[0] - 30)
            p[1] = min(max(p[1], half[1] + 30), size - half[1] - 30)
            pos[k] = p
        if not moved:
            break
    return pos, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig3_dir', default='../paper/fig3')
    ap.add_argument('--radius', type=float, default=RADIAL)
    ap.add_argument('--out_suffix', default='panel.png')
    args = ap.parse_args()

    font = None
    manifest = [json.loads(l) for l in
                open(os.path.join(args.fig3_dir, 'manifest.jsonl'))]

    for m in manifest:
        case_dir = os.path.join(args.fig3_dir, f"case_{m['case']:02d}")
        info = json.load(open(os.path.join(case_dir, 'labels.json')))
        A = np.asarray(info['A'], dtype=float)
        p0 = np.asarray(info['p0'], dtype=float)
        cen = np.asarray(info['cen'], dtype=float)
        size = int(info['img'])
        residues = info['residues']

        s = size / 1400.0                 # scale factor for all stroke/box sizes
        font = load_font(int(FONT_SIZE * s))
        pad, gap = int(PAD * s), int(GAP * s)
        lw, dot, halo = max(2, int(3 * s)), max(3, int(7 * s)), max(2, int(3 * s))

        img = Image.open(os.path.join(case_dir, 'panel_nolabel.png'))
        if img.mode == 'RGBA':          # ray-traced images keep an alpha channel
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert('RGB')
        draw = ImageDraw.Draw(img)

        texts = [r['text'] for r in residues]
        anchors = [p0 + A @ (np.asarray(r['ca'], dtype=float) - cen)
                   for r in residues]

        pos, boxes = place_labels(draw, font, texts, anchors, p0, size,
                                  args.radius, pad, gap)

        for text, base, p, b in zip(texts, anchors, pos, boxes):
            delta = p - base
            edge = p
            if np.linalg.norm(delta) > 1e-9:
                tx = (b[0] / 2.0) / abs(delta[0]) if abs(delta[0]) > 1e-9 else 1e9
                ty = (b[1] / 2.0) / abs(delta[1]) if abs(delta[1]) > 1e-9 else 1e9
                edge = p - delta * min(tx, ty)
            draw.line([tuple(base), tuple(edge)], fill=(50, 50, 50), width=lw)
            draw.ellipse([base[0] - dot, base[1] - dot, base[0] + dot, base[1] + dot],
                         fill=(50, 50, 50))
            w, h = text_box(draw, text, font)
            xy = (p[0] - w / 2.0, p[1] - h / 2.0)
            for ox, oy in ((-halo, 0), (halo, 0), (0, -halo), (0, halo)):
                draw.text((xy[0] + ox, xy[1] + oy), text, font=font,
                          fill=(255, 255, 255))
            draw.text(xy, text, font=font, fill=(0, 0, 0))

        out = os.path.join(case_dir, args.out_suffix)
        img.save(out)
        print('saved', out)


if __name__ == '__main__':
    main()
