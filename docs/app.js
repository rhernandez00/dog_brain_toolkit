import {Niivue} from "https://unpkg.com/@niivue/niivue@0.44.0/dist/index.js";

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
