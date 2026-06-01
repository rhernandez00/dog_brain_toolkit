#!/usr/bin/env python
"""Failsafe static-site exporter for the EmoC dashboard.

Generates a self-contained ``docs/`` folder (GitHub Pages root) that reproduces
as much of the Normal-mode viewer as possible WITHOUT a Python server:

    docs/
      index.html, app.js           NiiVue (2D+3D) + Plotly viewer
      manifest.json                describes every available result
      data/atlas/{D,H}_low.nii.gz  low-res atlas backgrounds
      data/results/.../*.nii.gz    z-maps (unthresholded + corrected)
      data/tables/*.json           cluster tables
      data/matrices/*.json         RSA model dissimilarity matrices

Why self-contained: committing the (small, low-res) NIfTI files makes the site
work offline on GitHub Pages with no Google Drive auth or CORS issues. The
low-res atlas is used as the background (NiiVue aligns the overlay via affines),
which also matches the low-resolution result space.

Usage:
  & "C:\\ProgramData\\anaconda3\\python.exe" export_static.py --dataset EmoC
  & "C:\\ProgramData\\anaconda3\\python.exe" export_static.py --dataset EmoC --modality RSA GLM
"""

import os
import sys
import json
import shutil
import argparse
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz import datasource, stimuli, niftiutil

THRESHOLD_PRESETS = [2.3, 3.1, 3.9]
SPECIES = ["D", "H"]
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Region labelling sources — must match searchlight.py step 10 / rsa_utils.create_tables.
# Dog: Czeibert labels (2mm) + dictionary. Human: AAL3 (per-dataset ROI) + AAL dictionary.
LABEL_ATLAS = {
    "D": os.path.join(REPO_ROOT, "Atlas", "Dog", "Nitzsche", "Czeibert_labels2mm.nii.gz"),
    "H": None,  # resolved per-dataset below ({datafolder}/{dataset}/ROI/AAL3.nii.gz)
}
LABEL_DICT_CSV = {
    "D": os.path.join(REPO_ROOT, "Atlas", "Dog", "Czeibert_dictionary.csv"),
    "H": os.path.join(REPO_ROOT, "Atlas", "Hum", "AAL_dictionary.csv"),
}


def _safe(name):
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))


def _table_to_json(path):
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    return {"columns": [str(c) for c in df.columns],
            "data": [{str(k): _jsonable(v) for k, v in row.items()} for row in df.to_dict("records")]}


def _jsonable(v):
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    return v


def _matrix_to_json(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    return {"index": [str(i) for i in df.index],
            "columns": [str(c) for c in df.columns],
            "z": [[_jsonable(x) for x in row] for row in df.values]}


def export(dataset, modalities, out_dir):
    datafolder = datasource.resolve_datafolder(dataset)
    print(f"Source : {datasource.describe_source(dataset)}")
    print(f"Output : {out_dir}")

    data_dir = os.path.join(out_dir, "data")
    for sub in ("atlas", "results", "tables", "matrices"):
        os.makedirs(os.path.join(data_dir, sub), exist_ok=True)

    # --- atlases (low-res backgrounds) ---
    atlases = {}
    for sp in SPECIES:
        src = niftiutil.ATLAS_PATHS[sp]["low"]
        if os.path.exists(src):
            rel = f"data/atlas/{sp}_low.nii.gz"
            shutil.copyfile(src, os.path.join(out_dir, rel))
            atlases[sp] = rel
            print(f"  atlas {sp}: {os.path.basename(src)}")

    results = []
    for modality in modalities:
        for roi in _rois_union(datafolder, dataset, modality):
            for model in _models_union(datafolder, dataset, modality, roi):
                entry = {"modality": modality, "roi": roi, "model": model, "species": {}}
                for sp in SPECIES:
                    sp_info = _export_species_result(datafolder, dataset, modality, roi, model, sp, out_dir)
                    if sp_info:
                        entry["species"][sp] = sp_info
                if not entry["species"]:
                    continue
                # RSA dissimilarity matrix (shared across species)
                if modality == "RSA":
                    csv = os.path.join(datafolder, dataset, "rsa_models", f"{model}.csv")
                    if os.path.exists(csv):
                        rel = f"data/matrices/{_safe(model)}.json"
                        with open(os.path.join(out_dir, rel), "w") as f:
                            json.dump(_matrix_to_json(csv), f)
                        entry["matrix"] = rel
                results.append(entry)
                print(f"  result: {modality}/{roi}/{model}  species={list(entry['species'])}")

    label_atlas, label_dict = _export_labels(out_dir, datafolder, dataset)

    manifest = {
        "dataset": dataset,
        "generated": datetime.now(timezone.utc).isoformat(),
        "thresholds": THRESHOLD_PRESETS,
        "atlases": atlases,
        "label_atlas": label_atlas,   # per-species region label NIfTI (for click-to-name)
        "label_dict": label_dict,     # per-species {number: region name} JSON
        "label_def": stimuli.LABEL_DEF,
        "results": results,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    _write_site(out_dir)
    print(f"\nDone. {len(results)} result(s). Open {os.path.join(out_dir, 'index.html')}")
    print("Deploy: commit docs/ and enable GitHub Pages (Settings -> Pages -> /docs).")


def _export_labels(out_dir, datafolder, dataset):
    """Copy each species' region-label atlas + dictionary so the failsafe site
    can name the region under a tapped voxel (mirrors create_tables labelling)."""
    label_atlas, label_dict = {}, {}
    sources = dict(LABEL_ATLAS)
    sources["H"] = os.path.join(datafolder, dataset, "ROI", "AAL3.nii.gz")
    for sp in SPECIES:
        src = sources.get(sp)
        if src and os.path.exists(src):
            rel = f"data/atlas/{sp}_labels.nii.gz"
            shutil.copyfile(src, os.path.join(out_dir, rel))
            label_atlas[sp] = rel
            print(f"  labels {sp}: {os.path.basename(src)}")
        dcsv = LABEL_DICT_CSV.get(sp)
        if dcsv and os.path.exists(dcsv):
            df = pd.read_csv(dcsv)
            mapping = {}
            for _, row in df.iterrows():
                num = row.get("Number")
                if pd.isna(num):
                    continue
                mapping[str(int(num))] = str(row.get("Region", "Unknown"))
            rel = f"data/atlas/{sp}_labels.json"
            with open(os.path.join(out_dir, rel), "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False)
            label_dict[sp] = rel
    return label_atlas, label_dict


def _rois_union(datafolder, dataset, modality):
    rois = set()
    for sp in SPECIES:
        rois.update(datasource.scan_roi_types(datafolder, dataset, modality, sp))
    return sorted(rois)


def _models_union(datafolder, dataset, modality, roi):
    models = set()
    for sp in SPECIES:
        models.update(datasource.scan_models(datafolder, dataset, modality, sp, roi))
    return sorted(models)


def _export_species_result(datafolder, dataset, modality, roi, model, sp, out_dir):
    overlay, kind = datasource.overlay_path(datafolder, dataset, modality, sp, roi, model)
    if overlay is None:
        return None
    rel_dir = os.path.join("data", "results", modality, sp, _safe(roi))
    os.makedirs(os.path.join(out_dir, rel_dir), exist_ok=True)

    info = {"overlay_kind": kind, "corrected": {}}
    rel_overlay = os.path.join(rel_dir, f"{sp}_{_safe(model)}_z.nii.gz").replace("\\", "/")
    shutil.copyfile(overlay, os.path.join(out_dir, rel_overlay))
    info["overlay"] = rel_overlay

    # per-threshold corrected maps + tables
    for zt in THRESHOLD_PRESETS:
        corr = datasource.corrected_path(datafolder, dataset, modality, sp, roi, model, z_threshold=zt)
        tab = datasource.table_path(datafolder, dataset, modality, sp, roi, model, z_threshold=zt)
        slot = {}
        if corr:
            rc = os.path.join(rel_dir, f"{sp}_{_safe(model)}_zt{zt}_corrected.nii.gz").replace("\\", "/")
            shutil.copyfile(corr, os.path.join(out_dir, rc))
            slot["corrected"] = rc
        if tab:
            rt = f"data/tables/{modality}__{sp}__{_safe(roi)}__{_safe(model)}__zt{zt}.json"
            with open(os.path.join(out_dir, rt), "w") as f:
                json.dump(_table_to_json(tab), f)
            slot["table"] = rt
        if slot:
            info["corrected"][str(zt)] = slot
    return info


def _write_site(out_dir):
    # index.html = stable landing page (QR target); viewer.html = the failsafe app.
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(_LANDING_HTML)
    with open(os.path.join(out_dir, "viewer.html"), "w", encoding="utf-8") as f:
        f.write(_VIEWER_HTML)
    with open(os.path.join(out_dir, "app.js"), "w", encoding="utf-8") as f:
        f.write(_APP_JS)
    # live.json holds the laptop tunnel URL; never clobber a URL the user set.
    live_path = os.path.join(out_dir, "live.json")
    if not os.path.exists(live_path):
        with open(live_path, "w", encoding="utf-8") as f:
            json.dump({"live_url": "", "note": "Set live_url to the laptop tunnel URL, then push."}, f, indent=2)


# --- static site source ---------------------------------------------------

_LANDING_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Networks of the social brain — comparative neuroimaging</title>
<style>
  :root{--bg:#ffffff;--panel:#f3f5f9;--ink:#222222;--muted:#667085;--line:#d5dbe5;--accent:#4472C4;}
  *{box-sizing:border-box;}
  html,body{margin:0;background:var(--bg);color:var(--ink);
            font-family:Georgia,'Times New Roman',serif;line-height:1.5;}
  .wrap{max-width:760px;margin:0 auto;padding:40px 24px 56px;}
  h1{font-size:26px;line-height:1.3;margin:0 0 10px;font-weight:600;}
  .authors{color:var(--muted);font-style:italic;font-size:15px;margin:0 0 4px;}
  .venue{color:var(--muted);font-size:13px;margin:0 0 22px;}
  hr{border:none;border-top:1px solid var(--line);margin:22px 0;}
  .abstract{font-size:15px;text-align:justify;}
  .abstract p{margin:0 0 12px;}
  .actions{display:flex;gap:14px;flex-wrap:wrap;margin:26px 0 6px;}
  a.btn{flex:1 1 240px;text-align:center;padding:16px 18px;border-radius:8px;text-decoration:none;
        font-family:Arial,Helvetica,sans-serif;font-weight:bold;font-size:16px;border:1px solid var(--line);}
  .live{background:var(--accent);color:#fff;border-color:var(--accent);}
  .live.off{background:#eef1f6;color:#9aa3b8;pointer-events:none;}
  .fail{background:#ED7D31;color:#fff;border-color:#ED7D31;}
  a.btn:hover{filter:brightness(1.06);}
  .sub{display:block;font-weight:normal;font-size:12px;opacity:.9;margin-top:4px;}
  .foot{color:var(--muted);font-size:12px;font-family:Arial,Helvetica,sans-serif;margin-top:22px;}
</style>
</head>
<body>
<div class="wrap">
  <h1>Networks of the social brain. Connecting dog to human social brain function using comparative neuroimaging</h1>
  <p class="authors">Raúl Hernández-Pérez, Laura V. Cuaya, Julia Meier, Ludwig Huber, Claus Lamm</p>
  <p class="venue">Comparative neuroimaging of affective processing in dogs and humans</p>
  <hr>
  <div class="abstract">
    <p>The human superior temporal sulcus (STS) is critical for perceiving emotions and navigating
       complex social interactions. While higher-level association cortices such as the STS are absent
       in carnivores, indirect neuroimaging evidence suggests the caudal sylvian gyrus (cSG) as a hub
       for the processing of affective information in dogs. Our aim is to compare the role of the human
       STS and dog cSG in affective processing while controlling for the effect of species and valence.
       We hypothesize that the human STS and dog cSG are functionally analogous in affective processing.</p>
    <p>We plan a fully comparative fMRI study in which humans (n = 40) and awake pet dogs (n = 24) will
       observe the same videos (13 s) of humans or dogs expressing happiness, excitement, fear, and
       anger. Additionally, we will acquire eye-tracking data simultaneously. We will acquire six runs
       per participant, each lasting 305 s. Data acquisition is ongoing.</p>
    <p>The planned dataset will allow us to examine brain networks related to affective processing and
       the interaction between emotions, valence, and species. In addition, the eye-tracking data will
       be correlated with the BOLD response at the individual and the group level. We will focus our
       analyses on the human STS and the dog cSG to determine the degree of similarity in their brain
       responses, connectivity patterns, and representational geometries.</p>
    <p>By demonstrating how visual representations of emotions expressed in others are processed in both
       species, we expect these findings to provide compelling evidence for understanding the evolution
       of social brain networks.</p>
  </div>
  <div class="actions">
    <a id="live" class="btn live off" href="#">Live interactive dashboard
       <span class="sub" id="livesub">checking availability…</span></a>
    <a class="btn fail" href="viewer.html">Results viewer
       <span class="sub">always available · 2D / 3D maps and tables</span></a>
  </div>
  <p class="foot" id="stamp"></p>
</div>
<script>
fetch("live.json?_="+Date.now()).then(r=>r.json()).then(j=>{
  const a=document.getElementById("live"), sub=document.getElementById("livesub");
  if(j && j.live_url){ a.href=j.live_url; a.classList.remove("off"); sub.textContent=j.live_url; }
  else { sub.textContent="currently offline — use the results viewer"; }
}).catch(()=>{document.getElementById("livesub").textContent="currently offline — use the results viewer";});
fetch("manifest.json").then(r=>r.json()).then(m=>{
  document.getElementById("stamp").textContent="Results generated " + (m.generated||"").slice(0,10) + " · dataset " + (m.dataset||"");
}).catch(()=>{});
</script>
</body>
</html>
"""

_VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>EmoC Results</title>
<style>
  :root{--bg:#ffffff;--panel:#f3f5f9;--ink:#222222;--muted:#667085;--line:#d5dbe5;--accent:#4472C4;}
  *{box-sizing:border-box;-webkit-tap-highlight-color:transparent;}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,Helvetica,sans-serif;}
  header{display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--panel);
         border-bottom:1px solid var(--line);}
  header a{color:var(--accent);text-decoration:none;font-size:22px;}
  .title{font-weight:bold;font-size:15px;}
  .badge{margin-left:auto;background:#fdeede;color:#b5651d;font-size:11px;padding:2px 8px;border-radius:10px;}
  .controls{padding:10px 14px;display:flex;flex-direction:column;gap:10px;}
  .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;}
  .grow{flex:1 1 auto;}
  label{font-size:12px;color:var(--muted);}
  select,input[type=range]{font-size:16px;background:#fff;color:var(--ink);
        border:1px solid var(--line);border-radius:8px;padding:8px;width:100%;}
  .seg{display:flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;}
  .seg button{flex:1;padding:10px 6px;background:#fff;color:var(--muted);border:none;
              font-size:13px;cursor:pointer;}
  .seg button.on{background:var(--accent);color:#fff;font-weight:bold;}
  .tabs{display:flex;gap:8px;}
  .tabs button{flex:1;padding:12px;border-radius:8px;border:1px solid var(--line);background:#fff;
               color:var(--ink);font-size:15px;font-weight:bold;cursor:pointer;}
  .tabs button.on{background:var(--accent);color:#fff;border-color:var(--accent);}
  .tabs button:disabled{opacity:.4;}
  .navbtn{min-width:54px;padding:12px;border-radius:8px;border:1px solid var(--line);
          background:#fff;color:var(--ink);font-size:18px;cursor:pointer;}
  #glwrap{position:relative;margin:0 14px;}
  canvas{width:100%;height:56vh;min-height:320px;background:#fff;border:1px solid var(--line);
         border-radius:10px;display:block;}
  #readout{margin:8px 14px;padding:10px 12px;background:var(--panel);border-radius:8px;font-size:14px;
           min-height:20px;}
  #readout .region{font-size:16px;font-weight:bold;color:var(--accent);}
  details{margin:10px 14px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 12px;}
  summary{cursor:pointer;font-weight:bold;padding:6px 0;}
  .mxlegend{display:flex;flex-wrap:wrap;gap:2px 14px;margin:6px 0 10px;}
  .mxlegend span.lg{display:inline-flex;align-items:center;font-size:12px;white-space:nowrap;}
  .mxlegend i{width:12px;height:12px;border-radius:3px;margin-right:5px;display:inline-block;}
  .mxgrid{display:grid;gap:4px;justify-content:start;overflow-x:auto;}
  .mxgrid .lab{font-size:10px;color:var(--ink);font-family:Consolas,monospace;text-align:right;
               align-self:center;padding-right:4px;}
  .mxgrid .cell{width:26px;height:26px;border-radius:7px;border:1px solid var(--line);}
  table{border-collapse:collapse;font-size:12px;width:100%;overflow-x:auto;display:block;}
  th,td{border:1px solid var(--line);padding:4px 6px;text-align:center;white-space:nowrap;}
  th{background:#eef1f6;}
  .muted{color:var(--muted);font-size:12px;}
  .hide{display:none!important;}
</style>
</head>
<body>
<header>
  <a href="index.html" title="Back">&#8592;</a>
  <span class="title">EmoC results</span>
  <span class="badge">offline viewer</span>
</header>

<div class="controls">
  <div class="tabs" id="sptabs">
    <button id="tabD" data-sp="D">Dog</button>
    <button id="tabH" data-sp="H">Human</button>
  </div>
  <div class="row">
    <div class="grow"><label>Model</label><select id="model"></select></div>
  </div>
  <div class="seg" id="viewseg">
    <button data-view="render" class="on">3D</button>
    <button data-view="axial">Axial</button>
    <button data-view="coronal">Coronal</button>
    <button data-view="sagittal">Sagittal</button>
  </div>
  <div class="row" id="slicenav">
    <button class="navbtn" id="prev">&#9664;</button>
    <span class="grow muted" id="sliceinfo" style="text-align:center;"></span>
    <button class="navbtn" id="next">&#9654;</button>
  </div>
  <div class="row">
    <label>z-threshold</label><span class="muted" id="ztval">3.1</span>
    <input class="grow" id="zt" type="range" min="0" max="8" step="0.1" value="3.1">
  </div>
</div>

<div id="glwrap"><canvas id="gl"></canvas></div>
<div id="readout"><span class="muted">Switch to a slice view and tap the brain to identify a region.</span></div>

<details id="matrixpanel">
  <summary>RSA model matrix</summary>
  <div id="mxlegend" class="mxlegend"></div>
  <div id="matrix" class="mxgrid"></div>
</details>

<details>
  <summary>Cluster table</summary>
  <div class="row"><label>Table @ z</label><select id="tablezt" style="width:auto;"></select></div>
  <div id="table"></div>
</details>

<script type="module" src="app.js"></script>
</body>
</html>
"""

_APP_JS = """import {Niivue} from "https://unpkg.com/@niivue/niivue@0.44.0/dist/index.js";

let manifest=null, nv=null;
const dictCache={};                       // species -> {number: region}
const state={specie:"D", view:"render", idx:0};   // idx = result index
let labelVolIdx=-1;                       // index of the label volume (or -1)

const el=id=>document.getElementById(id);
const results=()=>manifest.results;

async function boot(){
  manifest=await (await fetch("manifest.json")).json();
  // White background; no crosshair / origin lines (crosshairWidth 0 hides the
  // bright lines that mark the origin in slice views).
  nv=new Niivue({backColor:[1,1,1,1], show3Dcrosshair:false, crosshairWidth:0,
                 crosshairColor:[0,0,0,0], dragAndDropEnabled:false});
  nv.attachToCanvas(el("gl"));
  nv.onLocationChange=onLoc;

  // thresholds
  for(const t of manifest.thresholds){ const o=document.createElement("option"); o.value=t; o.textContent="z = "+t; el("tablezt").appendChild(o); }
  el("tablezt").value=3.1;

  // model list (value = result index; prefix modality/roi only if ambiguous)
  const multiMod=new Set(results().map(r=>r.modality)).size>1;
  const multiRoi=new Set(results().map(r=>r.roi)).size>1;
  el("model").innerHTML="";
  results().forEach((r,i)=>{
    const o=document.createElement("option"); o.value=i;
    o.textContent=r.model+((multiMod||multiRoi)?` (${r.modality}/${r.roi})`:"");
    el("model").appendChild(o);
  });

  // events
  el("model").addEventListener("change", e=>{ state.idx=+e.target.value; pickSpecies(); load(); });
  el("sptabs").addEventListener("click", e=>{ const sp=e.target.dataset.sp; if(sp){ state.specie=sp; load(); }});
  el("viewseg").addEventListener("click", e=>{ const v=e.target.dataset.view; if(v){ state.view=v; applyView(); }});
  el("prev").addEventListener("click", ()=>step(-1));
  el("next").addEventListener("click", ()=>step(1));
  el("zt").addEventListener("input", ()=>{ el("ztval").textContent=el("zt").value; applyThreshold(); });
  el("tablezt").addEventListener("change", renderTable);

  pickSpecies();
  await load();
}

function entry(){ return results()[state.idx]; }

function pickSpecies(){
  const e=entry();
  for(const sp of ["D","H"]){
    const has=!!(e && e.species[sp]);
    el("tab"+sp).disabled=!has;
  }
  if(!entry().species[state.specie]){ state.specie = entry().species.D ? "D" : (entry().species.H ? "H" : state.specie); }
  for(const sp of ["D","H"]) el("tab"+sp).classList.toggle("on", sp===state.specie);
}

async function loadDict(sp){
  if(dictCache[sp]!==undefined) return dictCache[sp];
  const rel=(manifest.label_dict||{})[sp];
  dictCache[sp]= rel ? await (await fetch(rel)).json() : null;
  return dictCache[sp];
}

async function load(){
  const e=entry(); if(!e) return;
  for(const sp of ["D","H"]) el("tab"+sp).classList.toggle("on", sp===state.specie);
  renderMatrix(e); renderTable();
  const info=e.species[state.specie];
  const bg=manifest.atlases[state.specie];
  if(!info || !bg){ await nv.loadVolumes([]); el("readout").innerHTML='<span class="muted">No result for this species.</span>'; return; }
  const zt=parseFloat(el("zt").value);
  const vols=[{url:bg, colormap:"gray"},
              {url:info.overlay, colormap:"warm", cal_min:zt, cal_max:6, opacity:0.85}];
  const labAtlas=(manifest.label_atlas||{})[state.specie];
  labelVolIdx = labAtlas ? 2 : -1;
  if(labAtlas) vols.push({url:labAtlas, colormap:"gray", opacity:0});  // hidden; sampled for region names
  await nv.loadVolumes(vols);
  await loadDict(state.specie);
  applyView(); applyThreshold();
}

function applyThreshold(){ if(nv && nv.volumes.length>1){ nv.volumes[1].cal_min=parseFloat(el("zt").value); nv.updateGLVolume(); } }

function applyView(){
  if(!nv) return;
  const m={render:nv.sliceTypeRender, axial:nv.sliceTypeAxial, coronal:nv.sliceTypeCoronal, sagittal:nv.sliceTypeSagittal};
  nv.setSliceType(m[state.view]);
  [...el("viewseg").children].forEach(b=>b.classList.toggle("on", b.dataset.view===state.view));
  el("slicenav").classList.toggle("hide", state.view==="render");
  el("sliceinfo").textContent = state.view==="render" ? "" : "step through "+state.view+" slices";
}

function step(d){
  if(!nv || state.view==="render") return;
  const ax={sagittal:[d,0,0], coronal:[0,d,0], axial:[0,0,d]}[state.view];
  nv.moveCrosshairInVox(ax[0], ax[1], ax[2]);
}

function onLoc(data){
  if(!data || !data.values) return;
  const ov = data.values[1] ? data.values[1].value : null;
  let region="—";
  if(labelVolIdx>=0 && data.values[labelVolIdx]){
    const num=Math.round(data.values[labelVolIdx].value);
    const dict=dictCache[state.specie];
    region = (dict && dict[num]) ? dict[num] : (num===0?"outside atlas":"label "+num);
  }
  const mm=data.mm? `(${data.mm.map(v=>v.toFixed(0)).join(", ")}) mm` : "";
  const ztxt = (ov!=null && isFinite(ov)) ? ` &middot; z = ${ov.toFixed(2)}` : "";
  el("readout").innerHTML = `<span class="region">${region}</span> ${ztxt}<br><span class="muted">${mm}</span>`;
}

function cellColor(v, vmax){
  const t = vmax ? Math.min(Math.abs(v)/vmax, 1) : 0;   // white -> Office blue
  const r=Math.round(255+t*(68-255)), g=Math.round(255+t*(114-255)), b=Math.round(255+t*(196-255));
  return `rgb(${r},${g},${b})`;
}

function renderMatrix(e){
  const p=el("matrixpanel");
  if(!e.matrix){ p.classList.add("hide"); return; }
  p.classList.remove("hide");
  fetch(e.matrix).then(r=>r.json()).then(mx=>{
    // category-name legend on top
    const leg=el("mxlegend"); leg.innerHTML="";
    for(const [lab,d] of Object.entries(manifest.label_def||{})){
      const s=document.createElement("span"); s.className="lg";
      s.innerHTML=`<i style="background:${d.color}"></i>${d.name}`; leg.appendChild(s);
    }
    // rounded-square grid with row labels (matrix is symmetric, so columns omitted)
    let vmax=0; mx.z.forEach(row=>row.forEach(v=>{ if(v!=null) vmax=Math.max(vmax,Math.abs(v)); }));
    const n=mx.index.length, grid=el("matrix");
    grid.style.gridTemplateColumns=`52px repeat(${n}, 26px)`;
    grid.innerHTML="";
    mx.index.forEach((name,i)=>{
      const lab=document.createElement("div"); lab.className="lab"; lab.textContent=name; grid.appendChild(lab);
      mx.z[i].forEach((v,j)=>{
        const c=document.createElement("div"); c.className="cell";
        c.style.background=(v==null)?"#fff":cellColor(v,vmax);
        c.title=`${name} × ${mx.columns[j]} = ${v==null?"":(+v).toFixed(2)}`;
        grid.appendChild(c);
      });
    });
  });
}

function renderTable(){
  const e=entry(), zt=el("tablezt").value, div=el("table");
  const info=e && e.species[state.specie];
  const slot=info && info.corrected && info.corrected[zt];
  if(!slot || !slot.table){ div.innerHTML='<div class="muted">No table at z='+zt+' for this species &mdash; schedule a job for this threshold.</div>'; return; }
  fetch(slot.table).then(r=>r.json()).then(t=>{
    let h="<table><thead><tr>"+t.columns.map(c=>"<th>"+c+"</th>").join("")+"</tr></thead><tbody>";
    h+=t.data.map(row=>"<tr>"+t.columns.map(c=>"<td>"+(row[c]??"")+"</td>").join("")+"</tr>").join("");
    div.innerHTML=h+"</tbody></table>";
  });
}

boot();
"""


def main():
    ap = argparse.ArgumentParser(description="Export the failsafe static results site")
    ap.add_argument("--dataset", default="EmoC")
    ap.add_argument("--modality", nargs="+", default=["RSA", "GLM"])
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "docs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    export(args.dataset, args.modality, args.out)


if __name__ == "__main__":
    main()
