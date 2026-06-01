import {Niivue} from "https://unpkg.com/@niivue/niivue@0.44.0/dist/index.js";

let manifest=null, nv=null;
const dictCache={};                       // species -> {number: region}
const state={specie:"D", view:"render", idx:0};   // idx = result index
let labelVolIdx=-1;                       // index of the label volume (or -1)

const el=id=>document.getElementById(id);
const results=()=>manifest.results;

async function boot(){
  manifest=await (await fetch("manifest.json")).json();
  nv=new Niivue({backColor:[0,0,0,1], show3Dcrosshair:true, crosshairColor:[0,1,1,1], dragAndDropEnabled:false});
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

function renderMatrix(e){
  const p=el("matrixpanel");
  if(!e.matrix){ p.classList.add("hide"); return; }
  p.classList.remove("hide");
  fetch(e.matrix).then(r=>r.json()).then(mx=>{
    Plotly.newPlot("matrix", [{z:mx.z, x:mx.columns, y:mx.index, type:"heatmap", colorscale:"Viridis"}],
      {margin:{l:50,r:6,t:6,b:50}, paper_bgcolor:"#181c28", plot_bgcolor:"#181c28",
       font:{color:"#e8ebf2",size:10}, yaxis:{autorange:"reversed"}},
      {displayModeBar:false, responsive:true});
    const chips=el("chips"); chips.innerHTML="";
    mx.index.forEach(c=>{ const col=(manifest.label_def[c.slice(-1)]||{}).color||"#888";
      const s=document.createElement("span"); s.className="chip"; s.style.background=col; s.textContent=c; chips.appendChild(s); });
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
