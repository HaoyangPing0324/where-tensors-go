const $=id=>document.getElementById(id);
const colors=["#9cbbff","#ffc087","#a4dfaf","#d3adf1","#ff9e9d","#7edbd2","#f6d777","#d8acc5"];
let scene,stage=0,beat=0,timer,serial=0,flows=[];
const requestCount=1;
const communicationRoundMs=800;
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
 const links=senders.map(r=>{
   const dst=scene.sharedDecode?r:0;
   const payload=scene.transfer?.(r,layerGroup(),layout()) || "MLA 内容与位置键，另含 DSA 索引缓存";
   const info="P"+r+" → D"+dst+" · 请求 A · "+payload;
   const sx=gx(r)+64, dx=gx(dst)+64, lane=27+3*r;
   const d="M"+sx+" 20 V"+lane+" H"+dx+" V62";
   return '<g class="wire" role="button" tabindex="0" aria-label="'+esc(info)+'" data-info="'+esc(info)+'"><title>'+esc(info)+'</title><path class="hit" d="'+d+'"/><path class="path" d="'+d+'" stroke="'+colors[r]+'"/><path d="M'+(dx-4)+' 56 l4 6 4 -6" fill="none" stroke="'+colors[r]+'"/></g>';
 }).join("");
 const labels=ranks().map(r=>txt(gx(r)+51,16,"P"+r,"hint")+txt(gx(r)+51,79,"D"+r,"hint")).join("");
 return '<section class="transfer"><strong>'+esc(scene.bridge)+' · 逻辑端点</strong><div class="scroll"><svg class="deck" role="group" aria-label="跨节点缓存传输" viewBox="0 0 1160 86">'+labels+links+'</svg></div></section>';
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
 const weightsY=state?251:active?176:67;
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
  cards+=txt(x+7,weightsY+83,owner?(phase===13 && beat<2?"请求 "+q+" · 待交接":"请求 "+q+" · KV 在卡"):(phase>=13 && side==="D"?"无本批 KV":""),"hint");
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
   paths+='<g tabindex="0" role="button" class="wire" data-info="'+info+'" aria-label="'+info+'"><title>'+info+'</title><path class="hit" d="'+p+'"/><path class="path" d="'+p+'" stroke="'+col+'"/><path d="M'+(dx-4)+' '+(sy-4)+' l4 4 -4 4" fill="none" stroke="'+col+'"/><circle class="packet" r="3.3" fill="'+col+'"><animateMotion dur="0.65s" path="'+p+'" repeatCount="1" fill="remove"/></circle></g>';
  }
 }
 const equation=active?scene.equation(stage,side,requestCount,layout(),layerGroup()):null;
 const math=equation?'<div class="nodeFormula"><div class="mathDisplay">'+katex.renderToString(equation[0],{displayMode:true,throwOnError:true,trust:false})+'</div><span class="mathCaption">'+esc(equation[1])+'</span></div>':"";
 const node=side==="P"?"01 · Prefill":"02 · Decode";
 return '<section class="node"><h2>'+node+'</h2>'+math+'<div class="scroll"><svg class="deck" viewBox="0 0 1160 '+h+'" role="group" aria-label="'+group+' GPU 缓冲区">'+cards+paths+'</svg></div></section>';
}
let displayedConfig="";
function renderParallelConfig(){
 const local=!!traceMode(), phase=scene.path(stage)[0];
 const key=[scene.id,$("engine").value,layout(),traceMode(),local?phase:""].join("/");
 if(key===displayedConfig)return;
 displayedConfig=key;
 const model=scene.moe;
 const config=local ? {
   hardware: phase+" · 路由专家算子",
   columns: ["节点","EP","单专家 TP","路由专家","每卡专家","每 token 命中","专家输入维"],
   rows: ()=>[[phase,8,1,model.experts,model.experts/8,model.top,model.width]],
 } : scene.parallel;
 $("layout").closest("label").hidden=local;
 $("engineDiff").hidden=local;
 $("hardware").textContent=config.hardware;
 $("profile").textContent=local?traceProfile($("model").value,$("engine").value,phase):scene.profile;
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
       entry.dataset.traceEntry=i;
       entry.textContent="专家通信 · Dispatch / Combine";
       parent.append(entry);
     }
     const button=document.createElement("button");button.dataset.step=i;
     button.textContent=scene.layerPresentation(i,layerGroup())?.title || item[1];
     parent.append(button);
   });
   tocKey=key;tocStage=-1;
 }
 if(stage!==tocStage){
   nav.querySelector("[aria-current]")?.removeAttribute("aria-current");
   const base=nav.querySelector('[data-step="'+stage+'"]');
   const active=traceMode()?base?.parentElement.querySelector("[data-trace-entry]"):base;
   if(active){
     active.setAttribute("aria-current","step");
     for(let branch=active.closest("details");branch;branch=branch.parentElement.closest("details"))branch.open=true;
     revealTocStep();
   }
   tocStage=stage;
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
 const trace=button.hasAttribute("data-trace-entry");
 const i=Number(trace?button.dataset.traceEntry:button.dataset.step);
 if(!allowed(i))$("layer").value=scene.layerClasses.findIndex(g=>g.ffn===scene.path(i)[2]);
 if(trace){
   stage=i; $("moePath").value="direct";tocStage=-1;startTrace(0);
 }else enter(i);
 if(matchMedia("(max-width:760px)").matches){toggleToc(false);$("toggleToc").focus();}
};
function render(){
 const stages=scene.stages,s=stages[stage],op=collective();
 const current=scene.path(stage);
 $("layerControl").hidden=!["Attention","FFN"].includes(current[1]);
 $("moeControl").hidden=current[2]!=="MoE";
 if(current[2]==="MoE"){
   const row=scene.parallel.rows(layout()).find(r=>r[0]===current[0]);
   const route=row[6]===1?"专家 TP · AllReduce":row[3]===1 && row[4]===8?"AllGather / ReduceScatter":"复制输入 · AllReduce";
   $("moePath").options[0].textContent="当前部署 · "+route;
 }
 renderParallelConfig();
 renderToc();
 if(traceMode()){renderTrace();return;}
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
 $("inspect").hidePopover();
 $("play").hidden=!isCommunication();
 $("prev").disabled=stage===0 && beat===0;$("next").disabled=stage===stages.length-1 && beat===lastBeat();
 $("play").textContent="重新播放通信";
}
function stop(){clearInterval(timer);timer=null;}
function isCommunication(){return !!collective();}
function enter(i,initial=0){
 stop(); if(traceMode()){ $("moePath").value="";tocStage=-1; }
 stage=i;beat=Math.min(initial,lastBeat());
 if(isCommunication()){
   beat=0;
   timer=setInterval(()=>{beat++;if(beat>=lastBeat())stop();render();},communicationRoundMs);
 }
 render();
}
function move(dir){
 if(traceMode()){const target=traceStep+dir;if(target<0 || target>=traceSteps.length)closeTrace();else startTrace(target);return;}
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
 $("moePath").value="";
 scene=variants[$("engine").value];stage=beat=0;tocKey="";
 $("comparison").innerHTML=Object.entries(variants).map(([engine,s])=>'<p><b>'+esc(engine)+'</b>　'+esc(s.profile)+' · '+esc(s.bridge)+'</p>').join("");
 $("schemeLabel").textContent=scene.parallel.schemeLabel || "部署拓扑";
 $("layout").innerHTML=scene.parallel.schemes.map(scheme=>'<option value="'+esc(scheme.id)+'">'+esc(scheme.label)+'</option>').join("");
 renderParallelConfig();
 $("layer").innerHTML=scene.layerClasses.map((group,i)=>'<option value="'+i+'">'+group.label+'</option>').join("");
 render();
}
$("model").innerHTML=catalog.map(m=>'<option value="'+m.id+'">'+m.name+'</option>').join("");
$("layer").onchange=()=>{
 const [phase,section]=scene.path(stage);
 const match=scene.stages.findIndex((s,i)=>scene.path(i)[0]===phase && scene.path(i)[1]===section && allowed(i));
 const target=allowed(stage)?stage:section==="Attention" && allowed(stage+1)?stage+1:match;
 if(traceMode() && scene.path(target)[2]==="MoE"){stage=target;startTrace(traceStep);}else enter(target);
};
$("engine").onchange=selectEngine;
$("model").onchange=()=>loadModel($("model").value).catch(error=>{$("title").textContent="场景加载失败："+error.message;});

$("layout").onchange=()=>{renderParallelConfig();enter(allowed(stage)?stage:0);};
$("prev").onclick=()=>{stop();move(-1);};$("next").onclick=()=>{stop();move(1);};
$("play").onclick=()=>traceMode()?startTrace(traceStep):enter(stage);
$("moePath").onchange=()=>{tocStage=-1;if(traceMode())startTrace($("map").querySelector(".traceDeck")?traceStep:0);else closeTrace();};
$("map").addEventListener("click",e=>{
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
loadModel(catalog[0].id).then(()=>{
 if(location.hash==="#moe"){
   $("toc").querySelector("[data-trace-entry]")?.click();
   $("moeControl").scrollIntoView({block:"start"});
 }
}).catch(()=>{$("title").textContent="模型场景加载失败";});
