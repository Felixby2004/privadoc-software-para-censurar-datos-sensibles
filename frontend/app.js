const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const SETTINGS_KEY = 'privadoc.settings.v1';
// Reemplaza únicamente este valor cuando tengas el enlace definitivo de Google Forms.
const FEEDBACK_FORM_URL = 'https://forms.gle/r8gJMAmQ57iB7TW79';
const DOCUMENT_TYPES = ['dni_azul','dni_amarillo','dni_electronico','comprobantes'];
const DEFAULT_SETTINGS = Object.freeze({
  defaultRedactionMode:'black',
  autoSelectRecommended:true,
  rememberDocumentType:true,
  lastDocumentType:'dni_azul'
});
function loadSettings(){
  try{
    const raw=JSON.parse(localStorage.getItem(SETTINGS_KEY)||'{}');
    return {
      defaultRedactionMode:['black','blur'].includes(raw.defaultRedactionMode)?raw.defaultRedactionMode:DEFAULT_SETTINGS.defaultRedactionMode,
      autoSelectRecommended:typeof raw.autoSelectRecommended==='boolean'?raw.autoSelectRecommended:DEFAULT_SETTINGS.autoSelectRecommended,
      rememberDocumentType:typeof raw.rememberDocumentType==='boolean'?raw.rememberDocumentType:DEFAULT_SETTINGS.rememberDocumentType,
      lastDocumentType:DOCUMENT_TYPES.includes(raw.lastDocumentType)?raw.lastDocumentType:DEFAULT_SETTINGS.lastDocumentType
    };
  }catch(_){ return {...DEFAULT_SETTINGS}; }
}
function saveSettings(next=settings){
  settings={...settings,...next};
  try{ localStorage.setItem(SETTINGS_KEY,JSON.stringify(settings)); }catch(_){}
  applySettingsToControls();
}
let settings = loadSettings();
let selectedFile = null;
let selectedDocumentType = settings.rememberDocumentType && DOCUMENT_TYPES.includes(settings.lastDocumentType) ? settings.lastDocumentType : 'dni_azul';
let analysis = null;
let activeDocIndex = 0;
let manualMode = false;
let progressTimer = null;
let analysisInFlight = false;
let resizeBound = false;
let lastProtectionScreen = 'uploadScreen';
let currentView = 'protection';
const selectedByDoc = new Map();
const manualByDoc = new Map();

const protectionScreens = ['uploadScreen','progressScreen','reviewScreen','protectScreen','doneScreen'];
const screens = [...protectionScreens,'settingsScreen'];
function showScreen(id){ if(protectionScreens.includes(id))lastProtectionScreen=id; screens.forEach(s => $('#'+s)?.classList.toggle('hidden', s!==id)); window.scrollTo({top:0,behavior:'smooth'}); }
function setStep(n){ for(let i=1;i<=3;i++){const el=$('#step'+i); el.classList.remove('active','done'); if(i<n)el.classList.add('done'); if(i===n)el.classList.add('active');} }
function escapeHtml(s){return String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function humanSize(n){ return n<1024*1024 ? `${(n/1024).toFixed(0)} KB` : `${(n/1024/1024).toFixed(2)} MB`; }
function treatmentLabel(f){if(f?.recommendation==='partial')return 'Censura parcial';if(f?.recommendation==='partial_phone')return 'Teléfono: deja 3 dígitos';if(f?.recommendation==='visible')return 'Mantener visible';return 'Censura total';}
function riskClass(risk){return risk==='Medio'?'medium':(risk==='Contexto'?'context':'');}
function doc(){ return analysis?.documents?.[activeDocIndex] || null; }
function selectedSet(d=doc()){ if(!d)return new Set(); if(!selectedByDoc.has(d.id))selectedByDoc.set(d.id,new Set()); return selectedByDoc.get(d.id); }
function manualBoxes(d=doc()){ if(!d)return []; if(!manualByDoc.has(d.id))manualByDoc.set(d.id,[]); return manualByDoc.get(d.id); }

async function readApiResponse(response, fallbackMessage){
  const raw=await response.text();
  let data=null;
  if(raw){
    try{
      data=JSON.parse(raw);
    }catch(error){
      console.error('Respuesta no JSON del servidor:',{status:response.status,raw:raw.slice(0,1000),error});
    }
  }
  if(!response.ok){
    const detail=data?.detail || fallbackMessage || `El servidor no pudo completar la solicitud (${response.status}).`;
    const error=new Error(detail);
    error.status=response.status;
    throw error;
  }
  if(!data){
    throw new Error('El servidor interrumpió el procesamiento antes de responder. Puede estar temporalmente ocupado, reiniciándose o sin recursos suficientes. Intenta nuevamente.');
  }
  return data;
}

function documentTypeLabel(type){
  if(type==='dni_amarillo') return 'DNI amarillo · menor de edad';
  if(type==='dni_electronico') return 'DNI electrónico';
  if(type==='comprobantes') return 'Comprobantes bancarios';
  return 'DNI azul peruano';
}
function shortDocumentType(type){
  if(type==='dni_amarillo') return 'DNI amarillo';
  if(type==='dni_electronico') return 'DNI electrónico';
  if(type==='comprobantes') return 'comprobante';
  return 'DNI azul';
}
function setDocumentType(type){
  if(!DOCUMENT_TYPES.includes(type)) return;
  selectedDocumentType=type;
  if(settings.rememberDocumentType){settings.lastDocumentType=type;saveSettings(settings);}
  $$('.doc-type[data-type]').forEach(btn=>btn.classList.toggle('selected',btn.dataset.type===type));
  const typeShort=type==='comprobantes'?'comprobante':'documento';
  const uploadTitle=$('#uploadDocTitle'); if(uploadTitle) uploadTitle.textContent=`Suelta aquí una foto, imagen o PDF de tu ${typeShort}`;
  const analyzeLabel=type==='comprobantes'?'comprobante':shortDocumentType(type);
  const analyze=$('#analyzeBtn'); if(analyze) analyze.innerHTML=`Analizar ${analyzeLabel} <span>→</span>`;
}
$$('.doc-type[data-type]').forEach(btn=>btn.addEventListener('click',()=>setDocumentType(btn.dataset.type)));

function pickFile(file){
  if(!file)return;
  const ok=['image/jpeg','image/png','application/pdf'].includes(file.type)||/\.(jpe?g|png|pdf)$/i.test(file.name);
  if(!ok){alert('Selecciona una imagen JPG/PNG o un PDF.');return;}
  if(file.size>30*1024*1024){alert('El archivo supera el límite de 30 MB.');return;}
  selectedFile=file; $('#fileName').textContent=file.name; $('#fileMeta').textContent=`${humanSize(file.size)} · ${file.type||'archivo'}`;
  $('#pickedFile').classList.remove('hidden'); $('#analyzeBtn').disabled=false;
}
$('#fileInput').addEventListener('change',e=>pickFile(e.target.files[0]));
$('#clearFile').addEventListener('click',()=>{selectedFile=null;$('#pickedFile').classList.add('hidden');$('#analyzeBtn').disabled=true;$('#fileInput').value='';});
['dragenter','dragover'].forEach(ev=>$('#dropZone').addEventListener(ev,e=>{e.preventDefault();$('#dropZone').classList.add('drag')}));
['dragleave','drop'].forEach(ev=>$('#dropZone').addEventListener(ev,e=>{e.preventDefault();$('#dropZone').classList.remove('drag')}));
$('#dropZone').addEventListener('drop',e=>pickFile(e.dataTransfer.files[0]));

function startProgress(){
  showScreen('progressScreen'); setStep(1); let idx=0;
  const states=selectedDocumentType==='comprobantes' ? [
    [16,'Preparando captura…','Normalizamos la imagen para mejorar la lectura del comprobante.',1],
    [34,'Leyendo estructura…','Ubicamos los bloques principales del voucher digital.',1],
    [54,'Reconociendo texto…','El OCR local interpreta nombres, cuentas y teléfonos.',2],
    [72,'Analizando secciones…','Relacionamos etiquetas como Enviado a, Desde y Código de operación.',2],
    [88,'Mapeando datos sensibles…','Marcamos nombres para censura parcial y números para censura total.',3],
    [95,'Verificando posiciones…','Comprobamos las regiones antes de mostrártelas.',3]
  ] : [
    [16,'Buscando documentos…','Detectamos uno o varios documentos dentro de la foto o página.',1],
    [34,'Corrigiendo perspectiva…','Enderezamos cada documento y normalizamos su tamaño.',1],
    [54,'Reconociendo estructura…','Probamos orientación, lado y estructura del documento.',2],
    [72,'Leyendo campos…','El OCR local interpreta únicamente el contenido necesario.',2],
    [88,'Mapeando datos sensibles…','Combinamos plantilla geométrica, OCR y validaciones.',3],
    [95,'Verificando posiciones…','Comprobamos las regiones antes de mostrártelas.',3]
  ];
  $('#progressBar').style.width='8%'; $('#progressPercent').textContent='8%';
  progressTimer=setInterval(()=>{
    if(idx>=states.length){
      $('#progressPercent').textContent='…';
      $('#progressTitle').textContent='Procesando documento…';
      $('#progressText').textContent='El servidor está terminando el OCR. No cierres esta ventana.';
      return;
    }
    const [p,t,d,step]=states[idx++];
    $('#progressBar').style.width=p+'%';
    $('#progressPercent').textContent=p+'%';
    $('#progressTitle').textContent=t;
    $('#progressText').textContent=d;
    for(let i=1;i<=3;i++){const el=$('#p'+i);el.classList.toggle('done',i<step);el.classList.toggle('active',i===step);}
  },700);
}
function finishProgress(){clearInterval(progressTimer);$('#progressBar').style.width='100%';$('#progressPercent').textContent='100%';$('#progressTitle').textContent='Análisis completado';$('#progressText').textContent='Ya puedes revisar cada documento detectado.';$$('.process-item').forEach(x=>{x.classList.remove('active');x.classList.add('done')});}

const wait=(ms)=>new Promise(resolve=>setTimeout(resolve,ms));
async function waitForAnalysis(jobId){
  const deadline=Date.now()+10*60*1000;
  while(Date.now()<deadline){
    const response=await fetch(`/api/analyze/${encodeURIComponent(jobId)}`,{cache:'no-store'});
    const data=await readApiResponse(response,'No se pudo consultar el estado del análisis.');
    if(data.status==='completed')return data;
    if(data.status==='error'){
      const error=new Error(data.error||'El servidor no pudo analizar el documento.');
      error.status=data.error_status;
      throw error;
    }
    await wait(1500);
  }
  throw new Error('El análisis superó el tiempo máximo de espera. El servidor no respondió a tiempo.');
}

$('#analyzeBtn').addEventListener('click',async()=>{
  if(!selectedFile || analysisInFlight)return;
  analysisInFlight=true;
  $('#analyzeBtn').disabled=true;
  startProgress();
  const fd=new FormData(); fd.append('file',selectedFile); fd.append('document_type',selectedDocumentType);
  try{
    const r=await fetch('/api/analyze',{method:'POST',body:fd});
    const queued=await readApiResponse(r,'No se pudo poner el documento en cola.');
    const d=await waitForAnalysis(queued.job_id);
    analysis=d; activeDocIndex=0; selectedByDoc.clear(); manualByDoc.clear();
    if(!Array.isArray(analysis.documents) || analysis.documents.length===0) throw new Error('El backend no devolvió ningún documento procesado.');
    analysis.documents.forEach(x=>{
      x.fields=Array.isArray(x.fields)?x.fields:[];
      selectedByDoc.set(x.id,new Set(settings.autoSelectRecommended?x.fields.filter(f=>f.default_redact).map(f=>f.id):[]));
    });
    finishProgress();
    setTimeout(()=>{
      try{
        lastProtectionScreen='reviewScreen';
        setStep(2);
        renderAnalysis();
        if(currentView==='protection')showScreen('reviewScreen');
      }catch(err){
        console.error('Error al renderizar el análisis:',err,analysis);
        showScreen('uploadScreen');
        alert('El análisis terminó, pero la interfaz no pudo mostrar el resultado: '+(err?.message||err));
      }
    },250);
  }catch(e){
    clearInterval(progressTimer);
    showScreen('uploadScreen');
    if(e?.status===429){
      alert('El servidor ya está procesando otro documento. Espera unos segundos y vuelve a intentarlo.');
    }else if(e?.status===502 || e?.status===503){
      alert('El servidor se reinició o se quedó sin recursos durante el análisis. Prueba con una imagen más pequeña o vuelve a intentarlo en unos segundos.');
    }else{
      alert(e.message);
    }
  }finally{
    analysisInFlight=false;
    $('#analyzeBtn').disabled=!selectedFile;
  }
});

function renderTabs(){
  const wrap=$('#documentTabs'); wrap.innerHTML='';
  analysis.documents.forEach((d,i)=>{
    const b=document.createElement('button'); b.className='document-tab'+(i===activeDocIndex?' active':'');
    const conf=Math.round((d.classification_confidence||0)*100);
    b.innerHTML=`<b>${i+1}</b><span class="tab-side">${escapeHtml(d.side_label||'DNI')}</span><span class="tab-conf">${conf}%</span>`;
    b.addEventListener('click',()=>{activeDocIndex=i;setManualMode(false);renderAnalysis();}); wrap.appendChild(b);
  });
  wrap.classList.toggle('hidden',analysis.documents.length<=1);
}

function renderWarnings(){
  const d=doc(), box=$('#documentWarnings'); const warnings=d?.warnings||[];
  if(!warnings.length){box.classList.add('hidden');box.innerHTML='';return;}
  box.classList.remove('hidden'); box.innerHTML=`<ul>${warnings.map(w=>`<li>${escapeHtml(w)}</li>`).join('')}</ul>`;
}

function renderAnalysis(){
  const d=doc(); if(!d)return;
  renderTabs(); renderWarnings();
  const preview = typeof d.preview==='string' && d.preview.length>20 ? d.preview : null;
  if(!preview) throw new Error('El backend no devolvió una vista previa válida del documento.');
  $('#previewImg').src=preview; $('#protectImg').src=preview;
  const reg=Math.round((d.registration_confidence||0)*100), cls=Math.round((d.classification_confidence||0)*100);
  const docLabel=d.type==='dni_amarillo'?'DNI amarillo':(d.type==='dni_electronico'?'DNI electrónico':(d.type==='comprobante'?'Comprobante digital':'DNI azul'));
  $('#engineText').textContent=`${analysis.ocr_engine} · ${docLabel} · ${d.side_label} · clasificación ${cls}% · registro ${reg}%`;
  $('#rectifiedText').textContent=d.type==='comprobante'?'Análisis estructural + OCR':(d.reference_used?'Plantilla de referencia registrada':'Normalización geométrica + OCR');
  $('#fieldCount').textContent=d.fields.length;
  const totalFields=analysis.documents.reduce((a,x)=>a+x.fields.length,0);
  $('#summaryText').textContent=`${analysis.document_count} documento${analysis.document_count===1?'':'s'} detectado${analysis.document_count===1?'':'s'} · ${totalFields} campos en total. Revisa cada pestaña.`;
  const high=d.fields.filter(f=>f.risk==='Alto').length; $('#riskScore').textContent=high?`${high} alto${high===1?'':'s'}`:'Sin riesgo alto';
  const list=$('#fieldList'); list.innerHTML=''; $('#emptyFields').classList.toggle('hidden',d.fields.length>0);
  const selected=selectedSet(d);
  d.fields.forEach(f=>{
    const row=document.createElement('label'); row.className='field'+(selected.has(f.id)?' checked':'');
    const checked=selected.has(f.id)?'checked':''; const confidence=Math.max(0,Math.min(100,Math.round((f.confidence||0)*100)));
    const confClass=confidence<55?'confidence-low':confidence>=80?'confidence-good':'';
    row.innerHTML=`<input class="field-check" data-id="${escapeHtml(f.id)}" type="checkbox" ${checked}><div><div class="field-name">${escapeHtml(f.name)}</div><div class="field-value">${escapeHtml(f.value)}</div><div class="field-meta"><span class="${confClass}">Confianza ${confidence}%</span><span>${escapeHtml(treatmentLabel(f))}</span><span class="field-source">${escapeHtml(f.source||'')}</span></div></div><span class="pill ${riskClass(f.risk)}">${escapeHtml(f.risk)}</span>`;
    const input=row.querySelector('input'); input.addEventListener('change',()=>{if(input.checked)selected.add(f.id);else selected.delete(f.id);row.classList.toggle('checked',input.checked);drawAllBoxes();updateSelectionSummary()}); list.appendChild(row);
  });
  updateManualList(); updateSelectionSummary();
  $('#previewImg').onload=drawAllBoxes; $('#protectImg').onload=drawProtectBoxes;
  if(!resizeBound){window.addEventListener('resize',()=>{drawAllBoxes();drawProtectBoxes()},{passive:true});resizeBound=true;}
  requestAnimationFrame(()=>{drawAllBoxes();drawProtectBoxes()});
}

function totalSelectedAuto(){let n=0; for(const s of selectedByDoc.values())n+=s.size; return n;}
function totalManual(){let n=0; for(const b of manualByDoc.values())n+=b.length; return n;}
function updateSelectionSummary(){const a=totalSelectedAuto(),m=totalManual();$('#selectedAuto').textContent=a;$('#selectedManual').textContent=m;$('#selectedTotal').textContent=a+m;$('#goProtectBtn').disabled=(a+m)===0;drawProtectBoxes();}
$('#selectRecommended').addEventListener('click',()=>{const d=doc(),s=selectedSet(d);d.fields.filter(f=>f.default_redact).forEach(f=>s.add(f.id));renderAnalysis();});
$('#clearSelection').addEventListener('click',()=>{selectedSet().clear();renderAnalysis();});

function boxStyle(box,img,wrap){const d=doc();const ir=img.getBoundingClientRect(),wr=wrap.getBoundingClientRect();const sx=ir.width/d.width,sy=ir.height/d.height;const [x1,y1,x2,y2]=box;return{left:ir.left-wr.left+x1*sx,top:ir.top-wr.top+y1*sy,width:(x2-x1)*sx,height:(y2-y1)*sy};}
function addVisualBox(layer,box,label,cls,img,wrap,selected=true){const s=boxStyle(box,img,wrap);const b=document.createElement('div');const compactLabel=['Sexo','Estado civil','Dígito verificador'].includes(label);b.className=`bbox ${cls||''} ${selected?'selected':''} ${compactLabel?'compact-label':''}`;Object.assign(b.style,{left:s.left+'px',top:s.top+'px',width:s.width+'px',height:s.height+'px'});if(label){if(compactLabel){b.innerHTML=`<span class="bbox-label compact-outside">${escapeHtml(label)}</span>`;}else{b.innerHTML=`<span class="bbox-label">${escapeHtml(label)}</span>`;}}if(label)b.setAttribute('aria-label',label);layer.appendChild(b);}
function drawAllBoxes(){const d=doc();if(!d||!$('#previewImg').naturalWidth)return;const layer=$('#boxes');layer.innerHTML='';const ids=selectedSet(d);d.fields.forEach(f=>addVisualBox(layer,f.box,f.name,riskClass(f.risk),$('#previewImg'),$('#imageWrap'),ids.has(f.id)));manualBoxes(d).forEach((b,i)=>addVisualBox(layer,b,`Zona manual ${i+1}`,'manual',$('#previewImg'),$('#imageWrap'),true));}
function drawProtectBoxes(){const d=doc();if(!d||!$('#protectImg').naturalWidth)return;const layer=$('#protectBoxes');layer.innerHTML='';const ids=selectedSet(d);d.fields.filter(f=>ids.has(f.id)).forEach(f=>addVisualBox(layer,f.box,f.name,riskClass(f.risk),$('#protectImg'),$('#protectImageWrap'),true));manualBoxes(d).forEach((b,i)=>addVisualBox(layer,b,`Zona manual ${i+1}`,'manual',$('#protectImg'),$('#protectImageWrap'),true));$('#protectDocumentLabel').textContent=`${d.type==='dni_amarillo'?'DNI amarillo':(d.type==='dni_electronico'?'DNI electrónico':(d.type==='comprobante'?'Comprobante':'DNI azul'))} ${activeDocIndex+1} · ${d.side_label}`;}

function setManualMode(on){manualMode=on;$('#manualLayer').classList.toggle('drawing',on);$('#manualHint').classList.toggle('hidden',!on);$('#manualBtn').textContent=on?'Seleccionando…':'＋ Añadir zona manual';}
$('#manualBtn').addEventListener('click',()=>setManualMode(!manualMode)); $('#cancelManual').addEventListener('click',()=>setManualMode(false));
let dragStart=null,drawEl=null;
$('#manualLayer').addEventListener('pointerdown',e=>{if(!manualMode||!doc())return;e.preventDefault();const r=$('#previewImg').getBoundingClientRect();dragStart={x:Math.max(r.left,Math.min(e.clientX,r.right)),y:Math.max(r.top,Math.min(e.clientY,r.bottom)),r};drawEl=document.createElement('div');drawEl.className='draw-box';$('#manualLayer').appendChild(drawEl);$('#manualLayer').setPointerCapture(e.pointerId)});
$('#manualLayer').addEventListener('pointermove',e=>{if(!dragStart||!drawEl)return;const r=dragStart.r,x=Math.max(r.left,Math.min(e.clientX,r.right)),y=Math.max(r.top,Math.min(e.clientY,r.bottom));const left=Math.min(dragStart.x,x)-r.left,top=Math.min(dragStart.y,y)-r.top,w=Math.abs(x-dragStart.x),h=Math.abs(y-dragStart.y);Object.assign(drawEl.style,{left:left+'px',top:top+'px',width:w+'px',height:h+'px'});});
$('#manualLayer').addEventListener('pointerup',e=>{if(!dragStart)return;const d=doc(),r=dragStart.r,x=Math.max(r.left,Math.min(e.clientX,r.right)),y=Math.max(r.top,Math.min(e.clientY,r.bottom));const left=Math.min(dragStart.x,x)-r.left,top=Math.min(dragStart.y,y)-r.top,w=Math.abs(x-dragStart.x),h=Math.abs(y-dragStart.y);drawEl?.remove();drawEl=null;dragStart=null;if(w>8&&h>8){const sx=d.width/r.width,sy=d.height/r.height;manualBoxes(d).push([Math.round(left*sx),Math.round(top*sy),Math.round((left+w)*sx),Math.round((top+h)*sy)]);updateManualList();updateSelectionSummary();drawAllBoxes()}setManualMode(false)});
function updateManualList(){const boxes=manualBoxes(),wrap=$('#manualListWrap'),list=$('#manualList');wrap.classList.toggle('hidden',boxes.length===0);$('#manualCount').textContent=boxes.length;list.innerHTML='';boxes.forEach((b,i)=>{const el=document.createElement('div');el.className='manual-chip';el.innerHTML=`<span></span><strong>Zona manual ${i+1}</strong><button data-i="${i}">Eliminar</button>`;el.querySelector('button').addEventListener('click',()=>{boxes.splice(i,1);updateManualList();updateSelectionSummary();drawAllBoxes()});list.appendChild(el)});}

$('#fitBtn').addEventListener('click',()=>{$('#imageStage').scrollTo({top:0,left:0,behavior:'smooth'})});
$('#goProtectBtn').addEventListener('click',()=>{updateSelectionSummary();drawProtectBoxes();showScreen('protectScreen');setStep(3)});
$('#backReviewBtn').addEventListener('click',()=>{showScreen('reviewScreen');setStep(2)});
$$('input[name="mode"]').forEach(x=>x.addEventListener('change',()=>$$('.mode-option').forEach(l=>l.classList.toggle('selected',l.querySelector('input').checked))));

function applySettingsToControls(){
  const mode=$('#defaultModeSetting'),auto=$('#autoSelectSetting'),remember=$('#rememberTypeSetting');
  if(mode)mode.value=settings.defaultRedactionMode;
  if(auto)auto.checked=settings.autoSelectRecommended;
  if(remember)remember.checked=settings.rememberDocumentType;
  $$('input[name="mode"]').forEach(input=>{input.checked=input.value===settings.defaultRedactionMode;input.closest('.mode-option')?.classList.toggle('selected',input.checked)});
}
function updatePageContext(view){
  const configs={
    protection:['Protección de identidad','Censura datos sensibles antes de compartir un documento','Analiza documentos de identidad peruanos, revisa lo encontrado y decide exactamente qué ocultar.'],
    settings:['Preferencias','Configuración de privacidad','Define cómo quieres trabajar con PRIVADOC sin modificar la precisión de los motores de detección.']
  };
  const cfg=configs[view]||configs.protection;
  $('#pageKicker').textContent=cfg[0];$('#pageTitle').textContent=cfg[1];$('#pageSubtitle').textContent=cfg[2];
  $('#mainStepper').classList.toggle('hidden',view!=='protection');
}
function setMainView(view){
  if(!['protection','settings'].includes(view))return;
  currentView=view;
  $$('.nav-item[data-view]').forEach(btn=>btn.classList.toggle('active',btn.dataset.view===view));
  updatePageContext(view);
  if(view==='settings'){applySettingsToControls();showScreen('settingsScreen');}
  else showScreen(lastProtectionScreen||'uploadScreen');
}
$$('.nav-item[data-view]').forEach(btn=>btn.addEventListener('click',()=>setMainView(btn.dataset.view)));
$('#defaultModeSetting')?.addEventListener('change',e=>saveSettings({defaultRedactionMode:e.target.value}));
$('#autoSelectSetting')?.addEventListener('change',e=>saveSettings({autoSelectRecommended:e.target.checked}));
$('#rememberTypeSetting')?.addEventListener('change',e=>saveSettings({rememberDocumentType:e.target.checked,lastDocumentType:selectedDocumentType}));
$('#resetSettingsBtn')?.addEventListener('click',()=>{settings={...DEFAULT_SETTINGS};saveSettings(settings);setDocumentType('dni_azul');});

function feedbackFormReady(){
  try {
    const url = new URL(FEEDBACK_FORM_URL);
    return (url.protocol === "https:" || url.protocol === "http:") &&
           (url.hostname === "docs.google.com" && url.pathname.startsWith("/forms/") ||
            url.hostname === "forms.gle");
  } catch (e) {
    return false;
  }
}
function revealFeedbackPrompt(){
  if(sessionStorage.getItem('privadoc.feedback.dismissed')==='1')return;
  const card=$('#feedbackCard');
  if(!card)return;
  card.classList.remove('hidden');
  requestAnimationFrame(()=>{
    card.classList.add('visible');
    window.setTimeout(()=>card.scrollIntoView({behavior:'smooth',block:'nearest'}),180);
  });
}
$('#downloadBtn')?.addEventListener('click',()=>{
  window.setTimeout(revealFeedbackPrompt,350);
});
$('#feedbackBtn')?.addEventListener('click',()=>{
  if(!feedbackFormReady()){
    alert('El formulario todavía no tiene un enlace configurado. Cuando nos pases el Google Forms, solo habrá que reemplazar FEEDBACK_FORM_URL en frontend/app.js.');
    return;
  }
  sessionStorage.setItem('privadoc.feedback.dismissed','1');
  window.open(FEEDBACK_FORM_URL,'_blank','noopener,noreferrer');
});
$('#feedbackLaterBtn')?.addEventListener('click',()=>{
  sessionStorage.setItem('privadoc.feedback.dismissed','1');
  $('#feedbackCard')?.classList.add('hidden');
});

function buildSelections(){return analysis.documents.map(d=>({document_id:d.id,field_ids:[...(selectedByDoc.get(d.id)||new Set())],manual_boxes:manualByDoc.get(d.id)||[]})).filter(x=>x.field_ids.length||x.manual_boxes.length);}
$('#redactBtn').addEventListener('click',async()=>{
  if(!analysis)return;const selections=buildSelections();if(!selections.length){alert('Selecciona al menos un dato o añade una zona manual.');return;}
  const btn=$('#redactBtn');btn.disabled=true;btn.textContent='Generando copia…';
  try{
    const mode=$('input[name="mode"]:checked').value;
    const r=await fetch('/api/redact',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({analysis_id:analysis.analysis_id,selections,mode})});
    const d=await readApiResponse(r,'No se pudo generar la copia censurada.');
    const outputFormat=(d.mime||'image/png').includes('pdf')?'PDF':'PNG';
    const protectionCount=totalSelectedAuto()+totalManual();
    $('#redactedImg').src=d.preview;$('#downloadBtn').href=d.download_data;$('#downloadBtn').download=d.filename;$('#doneCount').textContent=protectionCount;$('#doneFormat').textContent=outputFormat;
    showScreen('doneScreen');setStep(3);
  }catch(e){alert(e.message)}finally{btn.disabled=false;btn.textContent='Generar copia censurada'}
});
$('#redoBtn').addEventListener('click',()=>{renderAnalysis();showScreen('reviewScreen');setStep(2)}); $('#newBtn').addEventListener('click',()=>location.reload()); $('#anotherBtn').addEventListener('click',()=>location.reload());

applySettingsToControls();
setDocumentType(selectedDocumentType);
setMainView('protection');
