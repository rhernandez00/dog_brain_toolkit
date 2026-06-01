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

    manifest = {
        "dataset": dataset,
        "generated": datetime.now(timezone.utc).isoformat(),
        "thresholds": THRESHOLD_PRESETS,
        "atlases": atlases,
        "label_def": stimuli.LABEL_DEF,
        "results": results,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    _write_site(out_dir)
    print(f"\nDone. {len(results)} result(s). Open {os.path.join(out_dir, 'index.html')}")
    print("Deploy: commit docs/ and enable GitHub Pages (Settings -> Pages -> /docs).")


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
<title>EmoC Dashboard</title>
<style>
  html,body{margin:0;height:100%;background:#0f0f23;color:#e0e0ff;
            font-family:'Segoe UI',Arial,sans-serif;display:flex;align-items:center;justify-content:center;}
  .card{max-width:520px;text-align:center;padding:28px;}
  h1{margin:0 0 6px;letter-spacing:1px;}
  p{color:#9aa6c8;margin:6px 0 22px;}
  a.btn{display:block;margin:12px auto;padding:16px 20px;border-radius:12px;text-decoration:none;
        font-size:18px;font-weight:bold;max-width:360px;}
  .live{background:#4a90d9;color:#fff;}
  .live.off{background:#2a3450;color:#7a86a8;pointer-events:none;}
  .fail{background:#7a4fa0;color:#fff;}
  .sub{font-size:12px;font-weight:normal;display:block;opacity:.85;margin-top:4px;}
  .muted{font-size:11px;color:#667;margin-top:18px;}
</style>
</head>
<body>
<div class="card">
  <h1>🧠 EmoC Dashboard</h1>
  <p>Dog &amp; human neuroimaging — RSA results</p>
  <a id="live" class="btn live off" href="#">Live demo (laptop)
     <span class="sub" id="livesub">checking…</span></a>
  <a class="btn fail" href="viewer.html">Failsafe results
     <span class="sub">always available · 2D/3D, no scheduling</span></a>
  <div class="muted" id="stamp"></div>
</div>
<script>
fetch("live.json?_="+Date.now()).then(r=>r.json()).then(j=>{
  const a=document.getElementById("live"), sub=document.getElementById("livesub");
  if(j && j.live_url){ a.href=j.live_url; a.classList.remove("off"); sub.textContent=j.live_url; }
  else { sub.textContent="offline right now — use Failsafe"; }
}).catch(()=>{document.getElementById("livesub").textContent="offline right now — use Failsafe";});
</script>
</body>
</html>
"""

_VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EmoC Results (Failsafe)</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { margin:0; background:#0f0f23; color:#e0e0ff; font-family:'Segoe UI',Arial,sans-serif; }
  header { padding:8px 14px; background:#16213e; border-bottom:1px solid #2a2a44;
           display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  .brand { font-weight:bold; letter-spacing:1px; }
  .badge { background:#7a4fa0; color:#fff; font-size:11px; padding:2px 8px; border-radius:10px; }
  label { font-size:11px; color:#aaa; margin-right:4px; }
  select, input { background:#16213e; color:#fff; border:1px solid #333; border-radius:4px; padding:3px 6px; }
  .panel { background:#1a1a2e; border-radius:8px; margin:8px 14px; padding:8px 12px; }
  .species { display:flex; gap:8px; flex-wrap:wrap; }
  .canvaswrap { flex:1 1 420px; min-height:380px; position:relative; }
  canvas { width:100%; height:380px; background:#000; border-radius:6px; }
  .chip { color:#000; border-radius:6px; padding:4px 8px; margin:2px; font-family:Consolas,monospace;
          font-size:11px; font-weight:bold; display:inline-block; }
  table { border-collapse:collapse; font-size:11px; width:100%; }
  th,td { border:1px solid #222; padding:3px 6px; text-align:center; }
  th { background:#16213e; }
  .muted { color:#888; font-size:11px; }
  h4 { margin:4px 0; color:#e0e0ff; }
</style>
</head>
<body>
<header>
  <span class="brand">🧠 EmoC Results</span><span class="badge">Failsafe (read-only)</span>
  <span><label>Modality</label><select id="modality"></select></span>
  <span><label>ROI</label><select id="roi"></select></span>
  <span><label>Model</label><select id="model"></select></span>
  <span><label>z-threshold</label><input id="zt" type="range" min="0" max="8" step="0.1" value="3.1">
        <span id="ztval">3.1</span></span>
  <span><label>3D</label><input id="render3d" type="checkbox"></span>
  <span><label>Dog</label><input id="showD" type="checkbox" checked>
        <label>Human</label><input id="showH" type="checkbox" checked></span>
  <span class="muted" id="genstamp"></span>
</header>

<div class="panel" id="matrixpanel">
  <div id="chips"></div>
  <div id="matrix" style="height:320px;"></div>
</div>

<div class="panel">
  <div class="species">
    <div class="canvaswrap" id="wrapD"><h4>Dog</h4><canvas id="glD"></canvas>
      <div class="muted" id="statusD"></div></div>
    <div class="canvaswrap" id="wrapH"><h4>Human</h4><canvas id="glH"></canvas>
      <div class="muted" id="statusH"></div></div>
  </div>
</div>

<div class="panel">
  <div style="display:flex;gap:10px;align-items:center;">
    <h4>Cluster tables</h4>
    <span><label>Table @ z</label><select id="tablezt"></select></span>
  </div>
  <div class="species">
    <div style="flex:1 1 320px"><b>Dog</b><div id="tableD"></div></div>
    <div style="flex:1 1 320px"><b>Human</b><div id="tableH"></div></div>
  </div>
</div>

<script type="module" src="app.js"></script>
</body>
</html>
"""

_APP_JS = """import {Niivue} from "https://unpkg.com/@niivue/niivue@0.44.0/dist/index.js";

let manifest = null;
const nv = {D: null, H: null};
const SP = ["D", "H"];

function el(id){ return document.getElementById(id); }

async function boot(){
  manifest = await (await fetch("manifest.json")).json();
  el("genstamp").textContent = "generated " + (manifest.generated||"").slice(0,19);
  for (const t of manifest.thresholds){
    const o=document.createElement("option"); o.value=t; o.textContent=t; el("tablezt").appendChild(o);
  }
  el("tablezt").value = 3.1;
  for (const s of SP){
    nv[s] = new Niivue({backColor:[0,0,0,1], show3Dcrosshair:true});
    nv[s].attachToCanvas(el("gl"+s));
  }
  populateModality();
  ["modality","roi","model","tablezt"].forEach(id=>el(id).addEventListener("change", onSelectChange));
  el("zt").addEventListener("input", ()=>{ el("ztval").textContent=el("zt").value; applyThreshold(); });
  el("render3d").addEventListener("change", applyViewMode);
  el("showD").addEventListener("change", ()=>toggleSpecies("D"));
  el("showH").addEventListener("change", ()=>toggleSpecies("H"));
}

function uniq(a){ return [...new Set(a)]; }
function results(){ return manifest.results; }

function populateModality(){
  const mods = uniq(results().map(r=>r.modality));
  fill("modality", mods); populateRoi();
}
function populateRoi(){
  const m=el("modality").value;
  fill("roi", uniq(results().filter(r=>r.modality===m).map(r=>r.roi))); populateModel();
}
function populateModel(){
  const m=el("modality").value, roi=el("roi").value;
  fill("model", uniq(results().filter(r=>r.modality===m && r.roi===roi).map(r=>r.model)));
  loadCurrent();
}
function fill(id, vals){
  const s=el(id); s.innerHTML="";
  vals.forEach(v=>{ const o=document.createElement("option"); o.value=v; o.textContent=v; s.appendChild(o); });
}
function onSelectChange(e){
  if(e.target.id==="modality") populateRoi();
  else if(e.target.id==="roi") populateModel();
  else if(e.target.id==="model") loadCurrent();
  else if(e.target.id==="tablezt") renderTables();
}

function currentEntry(){
  const m=el("modality").value, roi=el("roi").value, model=el("model").value;
  return results().find(r=>r.modality===m && r.roi===roi && r.model===model);
}

async function loadCurrent(){
  const entry = currentEntry();
  if(!entry) return;
  renderMatrix(entry);
  for(const s of SP){ await loadSpecies(s, entry); }
  applyThreshold(); applyViewMode(); renderTables();
}

async function loadSpecies(s, entry){
  const info = entry.species[s];
  const status = el("status"+s);
  if(!info || !manifest.atlases[s]){ status.textContent="no result"; await nv[s].loadVolumes([]); return; }
  const zt = parseFloat(el("zt").value);
  await nv[s].loadVolumes([
    {url: manifest.atlases[s], colormap:"gray"},
    {url: info.overlay, colormap:"warm", cal_min: zt, cal_max: 6, opacity: 0.8}
  ]);
  status.textContent = info.overlay_kind + " z-map";
}

function applyThreshold(){
  const zt=parseFloat(el("zt").value);
  for(const s of SP){ const v=nv[s]; if(v && v.volumes.length>1){ v.volumes[1].cal_min=zt; v.updateGLVolume(); } }
}
function applyViewMode(){
  const t = el("render3d").checked;
  for(const s of SP){ const v=nv[s]; if(v){ v.setSliceType(t?v.sliceTypeRender:v.sliceTypeMultiplanar); } }
}
function toggleSpecies(s){ el("wrap"+s).style.display = el("show"+s).checked ? "" : "none"; }

function renderMatrix(entry){
  const panel=el("matrixpanel");
  if(!entry.matrix){ panel.style.display="none"; return; }
  panel.style.display="";
  fetch(entry.matrix).then(r=>r.json()).then(mx=>{
    Plotly.newPlot("matrix", [{z:mx.z, x:mx.columns, y:mx.index, type:"heatmap", colorscale:"Viridis"}],
      {margin:{l:60,r:10,t:10,b:60}, paper_bgcolor:"#1a1a2e", plot_bgcolor:"#1a1a2e",
       font:{color:"#fff"}, yaxis:{autorange:"reversed"}}, {displayModeBar:false});
    const chips=el("chips"); chips.innerHTML="";
    mx.index.forEach(c=>{
      const lab=c.slice(-1), col=(manifest.label_def[lab]||{}).color||"#888";
      const span=document.createElement("span"); span.className="chip";
      span.style.background=col; span.textContent=c; chips.appendChild(span);
    });
  });
}

function renderTables(){
  const entry=currentEntry(); const zt=el("tablezt").value;
  for(const s of SP){
    const div=el("table"+s); const info=entry && entry.species[s];
    const slot = info && info.corrected && info.corrected[zt];
    if(!slot || !slot.table){ div.innerHTML='<div class="muted">no table at z='+zt+' — schedule a job for this threshold.</div>'; continue; }
    fetch(slot.table).then(r=>r.json()).then(t=>{
      let h="<table><thead><tr>"+t.columns.map(c=>"<th>"+c+"</th>").join("")+"</tr></thead><tbody>";
      h+=t.data.map(row=>"<tr>"+t.columns.map(c=>"<td>"+(row[c]??"")+"</td>").join("")+"</tr>").join("");
      div.innerHTML=h+"</tbody></table>";
    });
  }
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
