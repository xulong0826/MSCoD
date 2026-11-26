"""Render Figure-3 panels with PyMOL (open-source build, no watermark).

Run from the repository root (consistent with the other Fig. 3 scripts):

    python visualization/fig3_render.py

Each case is written as `<case>/panel.png` and shows
  * the protein as a faint cartoon (global fold),
  * a semi-transparent molecular surface of the pocket (cavity shape),
  * the docked ligand as thick sticks (green carbon),
  * the residues that make polar contacts to the ligand (cyan sticks, labelled
    with residue name + number),
  * polar contacts (hydrogen bonds) as yellow dashes.
"""
import os
import glob
import json

import numpy as np
from pymol import cmd

LIG = 'resn LIG'
POLAR_CUT = 3.5      # A, ligand <-> protein polar contact distance
POCKET_CUT = 5.0     # A, residues whose surface shapes the pocket

IMG = 2000           # final raster size (px)
CAL = 500            # calibration raster size (px, unshaded, fast)
CAL_PNG = '/tmp/fig3_calib.png'
N_LABELS = 4         # residues annotated per panel


def color_by_element(sel, carbon='green'):
    """Standard element colouring with a chosen carbon colour."""
    cmd.color(carbon,   f'({sel}) and elem C')
    cmd.color('red',    f'({sel}) and elem O')
    cmd.color('blue',   f'({sel}) and elem N')
    cmd.color('yellow', f'({sel}) and elem S')
    cmd.color('orange', f'({sel}) and elem P')
    cmd.color('purple', f'({sel}) and elem Cl')
    cmd.color('cyan',   f'({sel}) and elem F')


def _marker_pixels(path):
    """Locate the four coloured calibration markers in a rendered PNG."""
    from PIL import Image

    im = np.asarray(Image.open(path).convert('RGB')).astype(float) / 255.0
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    masks = {
        'red':     (r - np.maximum(g, b)) > 0.25,
        'blue':    (b - np.maximum(r, g)) > 0.25,
        'yellow':  (np.minimum(r, g) - b) > 0.25,
        'magenta': (np.minimum(r, b) - g) > 0.25,
    }
    out = {}
    for name, m in masks.items():
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            raise RuntimeError(f'calibration marker {name} not found')
        # image y axis points down; keep it, we work in image coordinates
        out[name] = np.array([xs.mean(), ys.mean()])
    return out


def calibrate(cen):
    """Measure the object-space -> pixel mapping at the current camera.

    Four coloured pseudo-atoms are placed 5 A apart along the three object axes
    and their centroids are measured in a quick unshaded rendering, giving the
    2x3 matrix ``A`` (pixels per Angstrom) and the pixel position ``p0`` of
    ``cen`` at the final image resolution.
    """
    axes = {'red': np.zeros(3), 'blue': np.array([1.0, 0.0, 0.0]),
            'yellow': np.array([0.0, 1.0, 0.0]),
            'magenta': np.array([0.0, 0.0, 1.0])}
    for name, ax in axes.items():
        cmd.pseudoatom(f'cal_{name}', pos=[float(v) for v in cen + 5.0 * ax])
        cmd.show('spheres', f'cal_{name}')
        cmd.set('sphere_scale', 0.35, f'cal_{name}')
        cmd.color(name, f'cal_{name}')

    cmd.png(CAL_PNG, width=CAL, height=CAL, ray=0)
    cmd.delete('cal_*')

    px = _marker_pixels(CAL_PNG)
    p0 = px['red']
    A = np.column_stack([(px['blue'] - p0) / 5.0,
                         (px['yellow'] - p0) / 5.0,
                         (px['magenta'] - p0) / 5.0])
    scale = IMG / CAL
    return A * scale, p0 * scale


def contact_residues(n_keep=4, fallback_cut=4.2):
    """Annotated residues: polar contacts first, then the closest neighbours."""
    lig = np.array([a.coord for a in cmd.get_model(LIG).atom], dtype=float)

    def rank(sel):
        out, seen = [], set()
        for a in cmd.get_model(f'{sel} and name CA').atom:
            text = f'{a.resn}{a.resi}'
            if text in seen or a.alt not in ('', 'A'):
                continue          # skip alternate locations / duplicate chains
            seen.add(text)
            side = np.array([x.coord for x in cmd.get_model(
                f'chain {a.chain} and resi {a.resi} and resn {a.resn} '
                f'and not (name N+C+O)').atom], dtype=float)
            pool = side if len(side) else np.array([a.coord], dtype=float)
            d = np.sqrt(((pool[:, None, :] - lig[None, :, :]) ** 2).sum(-1)).min()
            out.append({'text': f'{a.resn}{a.resi}',
                        'chain': a.chain, 'resi': a.resi,
                        'ca': [float(v) for v in a.coord],
                        'dist': round(float(d), 2)})
        return sorted(out, key=lambda r: r['dist'])

    chosen = rank('polar_res')[:n_keep]
    if len(chosen) < n_keep:
        seen = {r['text'] for r in chosen}
        extra = [r for r in rank(f'byres (polymer within {fallback_cut} of {LIG})')
                 if r['text'] not in seen]
        chosen += extra[:n_keep - len(chosen)]
    return chosen


def orient_pocket(cen, bulk_cut=10.0):
    """Rotate the model so the camera looks into the pocket opening.

    The opening direction is approximated by the vector from the centroid of the
    protein bulk (atoms far from the ligand) to the ligand centroid. Rotating
    the whole model about the ligand centre until that vector points at the
    camera yields a consistent ``pocket mouth'' view.
    """
    cmd.select('_bulk', f'polymer beyond {bulk_cut} of {LIG}')
    bulk = cmd.get_model('_bulk').atom
    if not bulk:
        cmd.select('_bulk', 'polymer')
        bulk = cmd.get_model('_bulk').atom
    if not bulk:
        return
    bulk_c = np.array([a.coord for a in bulk], dtype=float).mean(axis=0)

    fwd = cen - bulk_c
    n = np.linalg.norm(fwd)
    if n < 1e-6:
        return
    fwd = fwd / n

    v = cmd.get_view()
    cam = np.array(v[9:12], dtype=float)
    cur = cen - cam
    cur = cur / (np.linalg.norm(cur) or 1.0)

    axis = np.cross(fwd, cur)
    n = np.linalg.norm(axis)
    if n < 1e-6:
        if fwd @ cur > 0:
            return                     # already aligned
        axis = np.array([0.0, 0.0, 1.0])   # anti-parallel: flip 180 deg
    else:
        axis = axis / n
    ang = np.degrees(np.arccos(float(np.clip(fwd @ cur, -1.0, 1.0))))
    cmd.rotate([float(x) for x in axis], float(ang), selection='all',
               origin=[float(x) for x in cen], camera=0)


def render(comp):
    cmd.reinitialize()
    cmd.bg_color('white')
    cmd.load(comp, 'cplx')
    cmd.hide('everything', 'all')

    # ---- camera: look down the pocket opening ---------------------------
    cen = np.array([a.coord for a in cmd.get_model(LIG).atom]).mean(axis=0)
    cmd.orient(LIG)
    orient_pocket(cen)
    # fixed ligand scale for all panels (consistent molecular size)
    cmd.zoom(LIG, 6.0)
    # then centre the pocket+ligand content to remove asymmetric empty space
    extent = cmd.get_extent(f'({LIG} or byres (polymer within {POCKET_CUT} of {LIG}))')
    if extent:
        (x0, y0, z0), (x1, y1, z1) = extent
        cmd.origin(position=[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])

    # ---- object space -> pixel mapping for label placement --------------
    # (done while nothing else is displayed, so the markers are never occluded)
    A, p0 = calibrate(cen)

    # ---- protein backbone: faint ribbon to give the global context -----
    cmd.show('cartoon', 'polymer')
    cmd.set('cartoon_transparency', 0.68)
    cmd.set('cartoon_side_chain_helper', 1)
    cmd.color('gray40', 'polymer')

    # ---- pocket surface: makes the shape of the cavity visible ----------
    cmd.select('pocket', f'byres (polymer within {POCKET_CUT} of {LIG})')
    cmd.show('surface', 'pocket')
    cmd.set('surface_quality', 2)
    cmd.set('solvent_radius', 1.4)
    cmd.set("transparency", 0.55, "pocket")
    cmd.color('gray80', 'pocket')     # whole pocket one neutral grey tone
    cmd.set('two_sided_lighting', 1)

    # ---- docked ligand --------------------------------------------------
    cmd.show('sticks', LIG)
    cmd.set('stick_radius', 0.26, LIG)
    color_by_element(LIG, carbon='green')

    # ---- residues that make polar contacts to the ligand ---------------
    cmd.select('polar_res',
               f'byres (polymer and (elem N+O+S) within {POLAR_CUT} '
               f'of ({LIG} and (elem N+O+S)))')

    # ---- highlight every residue that will be annotated -----------------
    ann = contact_residues(N_LABELS)
    if ann:
        parts = ' or '.join(f"(chain {r['chain']} and resi {r['resi']})" for r in ann)
        cmd.select('annotated', parts)
    else:
        cmd.select('annotated', 'none')
    cmd.show('sticks', 'annotated and not (hydro or name N+C+O)')
    cmd.set('stick_radius', 0.17, 'annotated and not (hydro or name N+C+O)')
    cmd.color('cyan', 'annotated')      # uniform cyan for the whole residue

    # ---- hydrogen bonds as yellow dashes -------------------------------
    cmd.distance('polar', LIG, 'polymer', POLAR_CUT, mode=2)
    cmd.hide('labels', 'polar')
    cmd.set('dash_color', 'yellow')
    cmd.set('dash_gap', 0.30)
    cmd.set('dash_length', 0.22)
    cmd.set('dash_width', 2.5)

    # ---- export geometry for the text overlay drawn afterwards ----------
    case_dir = os.path.dirname(comp)
    with open(os.path.join(case_dir, 'labels.json'), 'w') as fh:
        json.dump({'A': A.tolist(), 'p0': p0.tolist(), 'cen': cen.tolist(),
                   'img': IMG, 'residues': ann}, fh, indent=1)

    # ---- camera / ray trace --------------------------------------------
    cmd.set('ray_opaque_background', 0)
    cmd.set('antialias', 2)
    cmd.set('depth_cue', 0)
    cmd.set('ray_shadows', 1)       # soft shadows give the pocket depth
    cmd.set('ambient', 0.35)
    cmd.set('specular', 0.20)
    cmd.set('spec_power', 40)
    cmd.png(os.path.join(case_dir, 'panel_nolabel.png'),
            width=IMG, height=IMG, dpi=600, ray=1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default=None,
                    help='comma separated case indices, e.g. 0,1,2')
    args = ap.parse_args()

    cases = sorted(glob.glob(os.path.join('../paper/fig3', 'case_*', 'complex.pdb')))
    if args.cases:
        keep = {int(c) for c in args.cases.split(',')}
        cases = [c for c in cases if int(os.path.basename(os.path.dirname(c)).split('_')[1]) in keep]
    print(f'found {len(cases)} complexes')
    for comp in cases:
        render(comp)
        print('saved', os.path.join(os.path.dirname(comp), 'panel_nolabel.png'))


main()
