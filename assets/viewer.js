const $=id=>document.getElementById(id);
const colors=["#9cbbff","#ffc087","#a4dfaf","#d3adf1","#ff9e9d","#7edbd2","#f6d777","#d8acc5"];
let scene,stage=0,beat=0,timer,serial=0,flows=[];
const requestCount=1;
const communicationRoundMs=500;
const layout=()=>$("layout").value;
const layerGroup=()=>scene.layerClasses[Number($("layer").value)||0];
const allowed=i=>scene.layerAllows(i,layerGroup(),layout());
const ranks=()=>Array.from({length:scene.ranks},(_,i)=>i);
function collective(i=stage){
 const op=scene.op(i,layout());
 if(op.includes("AllReduce"))return "ar";
 if(op.includes("ReduceScatter"))return "rs";
 if(op.includes("AllGather"))return "ag";
 return "";
}
function lastBeat(){if(scene.service(stage))return 0;const op=collective();return op==="ar"?14:op?7:2;}
function ring(op,round){
 const N=scene.ranks, mod=x=>(x+N)%N;
 const mem=ranks().map(r=>ranks().map(k=>op==="ag" && k!==r?[]:[r]));
 let edges=[];
 for(let tick=1;tick<=round;tick++){
   const gather=op==="ag"||tick>7, t=tick>7?tick-7:tick;
   edges=ranks().map(r=>{
     const k=mod(r-t+(gather?1:0));
     return {src:r,dst:mod(r+1),k,origins:[...mem[r][k]],gather};
   });
   for(const e of edges){
     mem[e.dst][e.k]=gather?[...e.origins]:[...mem[e.dst][e.k],...e.origins];
     if(!gather)mem[e.src][e.k]=[];
   }
 }
 return {mem,edges};
}
const esc=s=>String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll('"',"&quot;");
function bridgeView(){
 const senders=!scene.sharedDecode && layout()==="tp"?[0]:ranks();
 const links=senders.flatMap(r=>{
   const dst=scene.sharedDecode?r:0;
   const blocks=scene.cacheBlocks?.(r);
   const payloads=blocks?blocks.map(b=>b.name+" "+b.shape+"。"+b.detail):[scene.transfer?.(r,layerGroup(),layout()) || "MLA 内容与位置键，另含 DSA 索引缓存"];
   return payloads.map((payload,k)=>{
     const info="P"+r+" → D"+dst+" · 请求 A · "+payload;
     const offset=64+(k-(payloads.length-1)/2)*32, color=colors[blocks?k:r];
     const sx=gx(r)+offset, dx=gx(dst)+offset, lane=27+3*r;
     const d="M"+sx+" 20 V"+lane+" H"+dx+" V62";
     return '<g class="wire" role="button" tabindex="0" aria-label="'+esc(info)+'" data-info="'+esc(info)+'"><title>'+esc(info)+'</title><path class="hit" d="'+d+'"/><path class="path" d="'+d+'" stroke="'+color+'"/><path d="M'+(dx-4)+' 56 l4 6 4 -6" fill="none" stroke="'+color+'"/></g>';
   });
 }).join("");
 const labels=ranks().map(r=>txt(gx(r)+51,16,"P"+r,"hint")+txt(gx(r)+51,79,"D"+r,"hint")).join("");
 return '<section class="transfer"><strong>'+esc(scene.bridge)+(scene.cacheBlocks?' · 每卡从左至右为递归状态、卷积历史、K/V':' · 逻辑端点')+'</strong><div class="scroll"><svg class="deck" role="group" aria-label="跨节点缓存传输" viewBox="0 0 1160 86">'+labels+links+'</svg></div></section>';
}
const txt=(x,y,s,cl="")=>'<text x="'+x+'" y="'+y+'" class="'+cl+'">'+esc(s)+'</text>';
const gx=r=>8+144*r;
function slot(x,y,w,h,label,color,tip,sub=""){
 return '<g tabindex="0" role="button" aria-label="'+esc(tip)+'" data-info="'+esc(tip)+'" class="datum"><title>'+esc(tip)+'</title><rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="4" fill="'+color+'" fill-opacity=".14" stroke="'+color+'"/>'+txt(x+6,y+16,label)+(sub?txt(x+6,y+34,sub,"hint"):"")+'</g>';
}
function drawNode(side,stage,beat){
 const phase=scene.phase(stage);
 const op=collective(stage), active=!scene.service(stage) && (side===(phase<13?"P":"D")||phase===13);
 const hidden=scene.hidden||6144, channelBlock=hidden/scene.ranks;
 const chunk=scene.chunk?.(stage);
 const shape=chunk?"["+chunk.rows+","+chunk.width+"]":side==="D"?"["+requestCount+","+channelBlock+"]":"["+(requestCount*2)+","+hidden+"]";
 const state=active && op?ring(op,beat):null;
 const hybridTransfer=phase===13 && scene.cacheBlocks;
 const weightsY=hybridTransfer?205:state?251:active?176:67;
 const h=weightsY+100;
 const group="N"+side;
 let cards="",paths="";
 for(const r of ranks()){
  const x=gx(r), boxes=(active && (side==="P" || r<requestCount || scene.sharedDecode) && scene.layerPresentation(stage,layerGroup())?.blocks) || scene.describe(stage,r,side,requestCount,layout(),layerGroup());
  cards+='<rect class="device" x="'+x+'" y="19" width="128" height="'+(h-24)+'" rx="7"/>'+txt(x+7,39,side+r,"rank");
  const q=scene.sharedDecode?"A":String.fromCharCode(65+r);
  const owner=side==="D" && phase>=13 && (r<requestCount||scene.sharedDecode) && stage!==scene.stages.length-1;
  const weight=scene.weightView(stage,side,r,layout(),active,layerGroup());
  cards+='<line x1="'+(x+7)+'" x2="'+(x+121)+'" y1="'+weightsY+'" y2="'+weightsY+'" stroke="#41515b"/>'+txt(x+7,weightsY+18,"常驻权重 · 摘选","hint")+txt(x+7,weightsY+39,weight[0],"weightName")+txt(x+7,weightsY+58,weight[1],"hint");
  let status=owner?(phase===13 && beat<2?"请求 "+q+" · 待交接":"请求 "+q+" · KV 在卡"):(phase>=13 && side==="D"?"无本批 KV":"");
  if(hybridTransfer)status=side==="P"?"请求 A · 源缓存可读":["目标槽已分配","请求 A · 接收中","请求 A · 已接收"][beat];
  cards+=txt(x+7,weightsY+83,status,"hint");
  if(state){
   state.mem[r].forEach((orig,k)=>{
    const y=49+k*23;
    if(!orig.length){cards+='<rect x="'+(x+6)+'" y="'+y+'" width="115" height="20" rx="3" fill="#141d22"/>';return;}
    const summed=op!=="ag" && orig.length===8, col=op==="ag"||summed?colors[k]:colors[orig[0]];
    const desc=side+r+" · B"+k+" "+shape+" · "+(op==="ag"?"复制自 "+side+k:"贡献卡 "+orig.map(v=>side+v).join(" + "))+(chunk?"。"+chunk.axis+"区间 ["+k*chunk.width+","+(k+1)*chunk.width+")。":side==="D"?"。隐藏通道 ["+channelBlock*k+","+channelBlock*(k+1)+")。":"。B"+k+" 表示位置 "+k+", "+(k+8)+" … 的行。");
    const embedding=phase===1 || scene.stages[stage][1]==="合并词向量";
    const hit=orig.includes(side==="D"?0:k);
    const status=summed?"已求和":op==="ag"?"完整块":embedding && !hit?"零":orig.length===1?"本地":"部分和";
    cards+=slot(x+6,y,115,20,"B"+k+" · "+status,col,desc+(summed?" 八份贡献已合并。":embedding && !hit?" 该块目前只含未命中词表的零贡献。":""));
   });
  }else if(hybridTransfer){
   scene.cacheBlocks(r).forEach((b,k)=>{
    const tip=side+r+" · "+(side==="P"?"源缓存":"目的缓存")+" · "+b.name+" "+b.shape+"。"+b.detail;
    cards+=slot(x+6,50+48*k,115,42,b.name,side==="D" && beat===0?"#73848a":colors[k],tip,b.shape);
   });
  }else if(active){
   // A single GPU buffer evolves in place: operands coexist for a multiply and are released after it.
   const waiting=!scene.sharedDecode && side==="D" && (phase===14||phase===13) && r>=requestCount;
   const visible=waiting?[boxes[0]]:beat===0?[boxes[0]]:beat===1?[boxes[0],boxes[1]]:[boxes[2]];
   if(phase===0 && beat!==1)visible.unshift(boxes[1]);
   visible.forEach((item,row)=>{
     const tip=side+r+" · "+item.name+" "+item.value+"。"+item.detail;
     cards+=slot(x+6,51+row*50,115,44,item.name,colors[r],tip,item.value);
   });
   if(!waiting && beat===1)cards+=txt(x+8,160,"↓ "+boxes[2].name,"hint");
  }else cards+=txt(x+7,57,"本步未参与","hint");
 }
 if(state){
  flows=state.edges;
  for(const e of flows){
   const sy=59+e.k*23, sx=gx(e.src)+122, dx=gx(e.dst)+6;
   const p=e.dst!==0?'M'+sx+' '+sy+' H'+dx:'M'+sx+' '+sy+' h9 V10 H3 V'+sy+' H'+dx;
   const col=colors[e.k], info=side+e.src+" → "+side+e.dst+" · "+scene.stages[stage][1]+" · B"+e.k+" "+shape+" · "+(e.gather?"传完整块，目的卡保留副本":"传部分和，在目的卡逐元素相加");
   paths+='<g tabindex="0" role="button" class="wire" data-info="'+info+'" aria-label="'+info+'"><title>'+info+'</title><path class="hit" d="'+p+'"/><path class="path" d="'+p+'" stroke="'+col+'"/><path d="M'+(dx-4)+' '+(sy-4)+' l4 4 -4 4" fill="none" stroke="'+col+'"/><circle class="packet" r="3.3" fill="'+col+'"><animateMotion dur="0.4s" path="'+p+'" repeatCount="1" fill="remove"/></circle></g>';
  }
 }
 const equation=active?scene.equation(stage,side,requestCount,layout(),layerGroup()):null;
 const math=equation?'<div class="nodeFormula"><div class="mathDisplay">'+katex.renderToString(equation[0],{displayMode:true,throwOnError:true,trust:false})+'</div><span class="mathCaption">'+esc(equation[1])+'</span></div>':"";
 const node=side==="P"?"01 · Prefill":"02 · Decode";
 return '<section class="node"><h2>'+node+'</h2>'+math+'<div class="scroll"><svg class="deck" viewBox="0 0 1160 '+h+'" role="group" aria-label="'+group+' GPU 缓冲区">'+cards+paths+'</svg></div></section>';
}
function renderParallelConfig(){
 const config=scene.parallel;
 $("hardware").textContent=config.hardware;
 $("profile").textContent=scene.profile;
 const header=config.columns.map(label=>'<th scope="col">'+esc(label)+'</th>').join("");
 const body=config.rows(layout()).map(([node,...values])=>'<tr><th scope="row">'+esc(node)+'</th>'+values.map(value=>'<td>'+esc(value)+'</td>').join("")+'</tr>').join("");
 $("parallelConfig").innerHTML='<thead><tr>'+header+'</tr></thead><tbody>'+body+'</tbody>';
}
let tocKey="", tocStage=-1;
function renderToc(){
 const key=scene.id+"-"+layout()+"-"+$("layer").value;
 const nav=$("toc");
 if(key!==tocKey){
   const openBranches=new Set([...nav.querySelectorAll("details[open]")].map(el=>el.dataset.branch));
   const branches=new Map();
   nav.replaceChildren();
   scene.stages.forEach((item,i)=>{
     const path=scene.path(i);
     if(!allowed(i) && path[1]!=="FFN")return;
     let parent=nav;
     const labels=scene.tocPath(i);
     labels.forEach((label,depth)=>{
       const branchKey=JSON.stringify([scene.id,...labels.slice(0,depth+1)]);
       let branch=branches.get(branchKey);
       if(!branch){
         branch=document.createElement("details");
         branch.dataset.branch=branchKey;
         const summary=document.createElement("summary");
         summary.textContent=label;
         branch.append(summary);
         parent.append(branch);
         branch.open=openBranches.has(branchKey);
         branches.set(branchKey,branch);
       }
       parent=branch;
     });
     if(path[2]==="MoE" && !parent.querySelector("[data-trace-entry]")){
       const entry=document.createElement("button");
       entry.dataset.traceEntry=path[0];
       entry.textContent="EP8 通信对照 · Dispatch / Combine";
       parent.append(entry);
     }
     const button=document.createElement("button");button.dataset.step=i;
     button.textContent=scene.layerPresentation(i,layerGroup())?.title || item[1];
     parent.append(button);
   });
   tocKey=key;tocStage=-1;
 }
 const selected=trace?.phase || stage;
 if(selected!==tocStage){
   nav.querySelector("[aria-current]")?.removeAttribute("aria-current");
   const base=nav.querySelector('[data-step="'+stage+'"]');
   const active=trace?nav.querySelector('[data-trace-entry="'+trace.phase+'"]'):base;
   if(active){
     active.setAttribute("aria-current","step");
     for(let branch=active.closest("details");branch;branch=branch.parentElement.closest("details"))branch.open=true;
     revealTocStep();
   }
   tocStage=selected;
 }
}
function revealTocStep(){
 const nav=$("toc"), current=nav.querySelector("[aria-current=step]");
 if(nav.hidden || !current)return;
 const item=current.getBoundingClientRect(), panel=nav.getBoundingClientRect();
 if(item.top<panel.top)nav.scrollTop+=item.top-panel.top;
 else if(item.bottom>panel.bottom)nav.scrollTop+=item.bottom-panel.bottom;
}
function toggleToc(open){
 $("workspace").classList.toggle("compact",!open);
 $("sidebar").hidden=!open;
 $("toc").hidden=!open;$("toggleToc").setAttribute("aria-expanded",open);
 $("toggleToc").textContent=open?"收起目录":"展开目录";
 if(open)revealTocStep();
}
$("toggleToc").onclick=()=>toggleToc($("toc").hidden);
toggleToc(!matchMedia("(max-width:760px)").matches);
$("toc").onclick=e=>{
 const button=e.target.closest("button[data-step],button[data-trace-entry]");if(!button)return;
 if(button.dataset.traceEntry){
   openTrace(button.dataset.traceEntry);
 }else{
   const i=Number(button.dataset.step);
   if(!allowed(i))$("layer").value=scene.layerClasses.findIndex(g=>g.ffn===scene.path(i)[2]);
   enter(i);
 }
 if(matchMedia("(max-width:760px)").matches){toggleToc(false);$("toggleToc").focus();}
};
function render(){
 const stages=scene.stages,s=stages[stage],op=collective();
 const current=scene.path(stage);
 $("layerControl").hidden=!!trace || !["Attention","Mamba-2","FFN"].includes(current[1]);
 $("traceControls").hidden=!trace;
 $("moeControl").hidden=!trace || traceStep===1;
 renderToc();
 if(trace){renderTrace();return;}
 const layerView=scene.layerPresentation(stage,layerGroup());
 $("title").textContent=layerView?.title || s[1];
 const phase=scene.service(stage)?"请求软件路径":op?(beat===0?"初始缓冲区":(op==="ag"||beat>7?"AllGather":"ReduceScatter")+"　轮 "+((beat-1)%7+1)+" / 7") : ["读取","计算 / 搬运","写回"][beat];
 if(op==="ar"){
   $("beat").textContent="AllReduce（逐元素求和）";
   const segment=beat===0?0:beat<=7?1:2;
   $("ringStages").hidden=false;
   $("ringStages").innerHTML=["初始分块","① ReduceScatter · 求和并分块","② AllGather · 复制已求和块"].map((name,i)=>'<span class="'+(i===segment?"now":"")+'">'+name+(i===segment && i>0?"　"+((beat-1)%7+1)+" / 7":"")+'</span>').join("<i>›</i>");
 }else{
   $("ringStages").hidden=true;
   $("beat").textContent=(scene.op(stage,layout())||s[2])+"　·　"+phase;
 }
 const svc=scene.service(stage);
 $("service").hidden=!svc;
 if(svc)$("service").innerHTML='<small>服务端路径</small>'+[svc.from,svc.via,svc.to].map(x=>'<b>'+esc(x)+'</b>').join('<span>→</span>');
 $("map").innerHTML=drawNode("P",stage,beat)+(scene.phase(stage)===13 ? bridgeView() : "")+drawNode("D",stage,beat);
 $("play").hidden=!isCommunication();
 $("prev").disabled=stage===0 && beat===0;$("next").disabled=stage===stages.length-1 && beat===lastBeat();
 $("play").textContent="重新播放通信";
}
function stop(){clearInterval(timer);timer=null;}
function isCommunication(){return !!collective();}
function enter(i,initial=0,resume=false){
 stop(); $("inspect").hidePopover(); trace=null; stage=i;
 beat=isCommunication() && !resume?0:Math.min(initial,lastBeat());
 if(isCommunication() && beat<lastBeat()){
   timer=setInterval(()=>{beat++;if(beat>=lastBeat())stop();render();},communicationRoundMs);
 }
 render();
}
function move(dir){
 if(trace){const target=traceStep+dir;if(target>=0 && target<traceSteps.length)startTrace(target);return;}
 if(!isCommunication() && beat+dir>=0 && beat+dir<=lastBeat()){beat+=dir;render();return;}
 let target=stage+dir;
 // On the default forward walkthrough, advance from the last dense example to the first MoE layer.
 if(dir>0 && scene.path(stage)[2]==="Dense" && scene.path(target)?.[2]==="MoE"){
   $("layer").value=scene.layerClasses.findIndex(g=>g.ffn==="MoE");
   target=scene.stages.findIndex((s,i)=>scene.path(i)[0]===scene.path(stage)[0] && scene.path(i)[1]==="Attention");
 }
 while(target>=0 && target<scene.stages.length && !allowed(target))target+=dir;
 if(target>=0 && target<scene.stages.length)enter(target,dir>0?0:2);
}
const scripts=new Map();
function loadScript(url){
 if(!scripts.has(url))scripts.set(url,new Promise((resolve,reject)=>{
   const tag=document.createElement("script");tag.src=url;tag.onload=resolve;
   tag.onerror=()=>{scripts.delete(url);reject(Error(url));};document.head.append(tag);
 }));
 return scripts.get(url);
}
async function loadModel(id){
 const version=++serial,entry=catalog.find(m=>m.id===id);stop();$("engine").disabled=true;
 for(const url of entry.scripts)await loadScript(url);
 if(version!==serial)return;
 const variants=window.inferenceModels[id];
 $("engine").innerHTML=Object.keys(variants).map(name=>'<option>'+esc(name)+'</option>').join("");
 $("engine").value=entry.defaultEngine;
 selectEngine();$("engine").disabled=false;
}
function selectEngine(){
 stop();const variants=window.inferenceModels[$("model").value];
 const previous=scene, oldStage=stage, oldBeat=beat, oldLayer=previous && layerGroup(), oldLayout=layout();
 scene=variants[$("engine").value];stage=beat=0;tocKey="";
 $("comparison").innerHTML=Object.entries(variants).map(([engine,s])=>'<p><b>'+esc(engine)+'</b>　'+esc(s.profile)+' · '+esc(s.bridge)+'</p>').join("");
 $("schemeLabel").textContent=scene.parallel.schemeLabel || "部署拓扑";
 $("layout").innerHTML=scene.parallel.schemes.map(scheme=>'<option value="'+esc(scheme.id)+'">'+esc(scheme.label)+'</option>').join("");
 if(scene.parallel.schemes.some(s=>s.id===oldLayout))$("layout").value=oldLayout;
 renderParallelConfig();
 $("layer").innerHTML=scene.layerClasses.map((group,i)=>'<option value="'+i+'" title="'+esc(group.detail || group.label)+'">'+esc(group.label)+'</option>').join("");
 if(previous){
   const exact=scene.layerClasses.findIndex(g=>g.label===oldLayer.label);
   const similar=scene.layerClasses.findIndex(g=>g.ffn===oldLayer.ffn);
   $("layer").selectedIndex=exact>=0?exact:Math.max(0,similar);
   const path=previous.path(oldStage), candidates=scene.stages.map((_,i)=>i).filter(i=>allowed(i) && scene.path(i)[0]===path[0]);
   stage=candidates.find(i=>scene.stages[i][1]===previous.stages[oldStage][1])
     ?? candidates.find(i=>scene.path(i).join("/")===path.join("/")) ?? candidates[0] ?? 0;
 }
 const sameStep=previous && scene.stages[stage][1]===previous.stages[oldStage][1];
 if(trace){
   if(!sameStep)trace.returnBeat=0;
   startTrace(traceStep,oldBeat);
 }else enter(stage,sameStep?oldBeat:0,true);
}
$("model").innerHTML=catalog.map(m=>'<option value="'+m.id+'">'+m.name+'</option>').join("");
$("layer").onchange=()=>{
 const [phase,section]=scene.path(stage);
 const match=scene.stages.findIndex((s,i)=>scene.path(i)[0]===phase && scene.path(i)[1]===section && allowed(i));
 const nextInSection=scene.path(stage+1)?.[1]===section && allowed(stage+1);
 const target=allowed(stage)?stage:nextInSection?stage+1:match;
 const samePhase=scene.stages.findIndex((_,i)=>scene.path(i)[0]===phase && scene.path(i)[1]==="Mamba-2" && allowed(i));
 enter(target>=0?target:samePhase>=0?samePhase:0);
};
$("engine").onchange=selectEngine;
$("model").onchange=()=>loadModel($("model").value).catch(error=>{$("title").textContent="场景加载失败："+error.message;});

$("layout").onchange=()=>{
 renderParallelConfig();
 if(!allowed(stage)){
   const phase=scene.path(stage)[0];
   const candidates=scene.stages.map((_,i)=>i).filter(i=>allowed(i) && scene.path(i)[0]===phase);
   stage=candidates.reduce((best,i)=>Math.abs(i-stage)<Math.abs(best-stage)?i:best,candidates[0] ?? 0);
   if(trace)trace.returnBeat=0;
 }
 if(trace)render();else enter(stage,beat,true);
};
$("prev").onclick=()=>{stop();move(-1);};$("next").onclick=()=>{stop();move(1);};
$("play").onclick=()=>trace?startTrace(traceStep):enter(stage);
$("moePath").onchange=changeTracePath;
$("traceControls").addEventListener("click",e=>{
 const step=e.target.closest("[data-trace-step]");
 if(step)startTrace(Number(step.dataset.traceStep));
 else if(e.target.closest("[data-close-trace]"))closeTrace();
});

document.addEventListener("click",e=>{
 const tile=e.target.closest("[data-info]");
 if(tile){
   const tip=$("inspect");
   tip.textContent=tile.dataset.info;
   tip.showPopover();
   const rect=tile.getBoundingClientRect();
   const x=e.clientX || rect.left, y=e.clientY || rect.bottom;
   tip.style.left=Math.max(8,Math.min(x,innerWidth-tip.offsetWidth-8))+"px";
   tip.style.top=Math.max(8,Math.min(y+8,innerHeight-tip.offsetHeight-8))+"px";
 }
});
document.addEventListener("keydown",e=>{if(e.target.closest("[data-info]") && ["Enter"," "].includes(e.key)){e.preventDefault();e.target.dispatchEvent(new MouseEvent("click",{bubbles:true}));}});
const requestedModel=new URLSearchParams(location.search).get("model");
$("model").value=catalog.find(m=>m.id===requestedModel)?.id || catalog[0].id;
loadModel($("model").value).then(()=>{
 if(location.hash==="#moe"){
   const first=scene.stages.findIndex((_,i)=>scene.path(i)[0]==="Prefill" && scene.path(i)[2]==="MoE");
   if(first>=0){
     $("layer").selectedIndex=scene.layerClasses.findIndex(g=>g.ffn==="MoE");
     stage=first;
     openTrace("Prefill");
   }
 }
}).catch(()=>{$("title").textContent="模型场景加载失败";});
