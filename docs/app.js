import {Niivue} from "https://unpkg.com/@niivue/niivue@0.44.0/dist/index.js";

let manifest=null, nv=null;
const dictCache={};                       // species -> {number: region}
const state={specie:"D", view:"render", idx:0, maptype:"z"};   // idx = result index; maptype: z | mean
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
  el("mapseg").addEventListener("click", e=>{ const mp=e.target.dataset.map; if(mp && !e.target.disabled){ state.maptype=mp; load(); }});
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
  // enable/disable the Mean button per available map; reflect the active map
  const meanBtn=[...el("mapseg").children].find(b=>b.dataset.map==="mean");
  if(meanBtn) meanBtn.disabled = !(info && info.mean);
  if(state.maptype==="mean" && !(info && info.mean)) state.maptype="z";
  [...el("mapseg").children].forEach(b=>b.classList.toggle("on", b.dataset.map===state.maptype));
  // the z-threshold slider applies to the z-map only
  el("ztrow").classList.toggle("hide", state.maptype==="mean");
  if(!info || !bg){ await nv.loadVolumes([]); el("readout").innerHTML='<span class="muted">No result for this species.</span>'; return; }
  const zt=parseFloat(el("zt").value);
  const useMean = state.maptype==="mean" && info.mean;
  const ov = useMean
    ? {url:info.mean, colormap:"warm", cal_min:0, opacity:0.85}
    : {url:info.overlay, colormap:"warm", cal_min:zt, cal_max:6, opacity:0.85};
  const vols=[{url:bg, colormap:"gray"}, ov];
  const labAtlas=(manifest.label_atlas||{})[state.specie];
  labelVolIdx = labAtlas ? 2 : -1;
  if(labAtlas) vols.push({url:labAtlas, colormap:"gray", opacity:0});  // hidden; sampled for region names
  await nv.loadVolumes(vols);
  // mean map: let NiiVue auto-scale the top (Kendall tau range), clamp floor to 0
  if(useMean && nv.volumes.length>1){ nv.volumes[1].cal_min=0; nv.updateGLVolume(); }
  await loadDict(state.specie);
  applyView(); applyThreshold();
}

function applyThreshold(){ if(nv && nv.volumes.length>1 && state.maptype!=="mean"){ nv.volumes[1].cal_min=parseFloat(el("zt").value); nv.updateGLVolume(); } }

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
  // Region naming is for slice views only — tapping in the 3D render produces
  // unreliable coordinates that crash the lookup, so skip it there entirely.
  if(state.view==="render") return;
  if(!data || !data.values) return;
  const ov = data.values[1] ? data.values[1].value : null;
  let region="—";
  if(labelVolIdx>=0 && data.values[labelVolIdx]){
    const num=Math.round(data.values[labelVolIdx].value);
    const dict=dictCache[state.specie];
    region = (dict && dict[num]) ? dict[num] : (num===0?"outside atlas":"label "+num);
  }
  const mm=data.mm? `(${data.mm.map(v=>v.toFixed(0)).join(", ")}) mm` : "";
  const unit = state.maptype==="mean" ? "&tau;" : "z";
  const vtxt = (ov!=null && isFinite(ov)) ? ` &middot; ${unit} = ${ov.toFixed(2)}` : "";
  el("readout").innerHTML = `<span class="region">${region}</span> ${vtxt}<br><span class="muted">${mm}</span>`;
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
