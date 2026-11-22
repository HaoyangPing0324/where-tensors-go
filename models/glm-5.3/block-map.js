// The two node panels share coordinates in every frame. Each wire joins two block ports.
const blockMap = (() => {
  const palette = ["#8fb7ff","#eeba85","#9ed6ad","#c4a5eb","#ef9da4","#80d5db","#f2cd78","#caaebd"];
  let before=false;
  const blank = () => ({name:"空槽", value:"—", info:"本拍尚无所跟踪 token 的数据。", idle:true});
  const box = (name,value,info,tags=[]) => ({name,value,info,tags});
  const ranks=Array.from({length:8},(_,i)=>i);
  const x = r => 29 + r*139;
  const y = (side,row) => 54 + (side==="D"?344:0) + [0,109,191][row];
  const text = (x,y,s,cls="")=>'<text x="'+x+'" y="'+y+'" class="'+cls+'">'+s+'</text>';
  function blocks(side,r,f){
    const req=requestName(r), expert=expertsOn(r), n=expert.length, ids=expert.map(i=>"E"+i).join("·");
    const localRows=requests*2, t=requests*16;
    if(f!==null){
      if(side==="D")return [box("TP 权重","分片 "+(r+1)+"/8","每个专家的中间维 ["+256*r+","+(r+1)*256+")"),blank(),blank()];
      const own=box("本地 token 分块","G"+r+" ["+localRows+",6144]","交错位置 "+r+", "+(r+8)+" …；每行保留全部 6144 通道。",[r]);
      const gathered=box("收齐 G0–G7","["+t+",6144]","各色小块对应源 rank 的 token 行。金色 G6 含 A 的第 7 个 token。",ranks);
      if(f===1)return [own,before?blank():gathered,blank()];
      const weights=box("专家权重","E"+32*r+"–"+(32*r+31),"本卡常驻完整专家；对当前 token 命中 "+(ids||"0 个专家"));
      if(f===0)return [gathered,box("路由评分","256 → Top 8","选择 8 个路由专家，同时保留路由系数。"),box("本地命中",ids||"无命中","此 token 在 P"+r+" 命中 "+(ids||"0 个专家"))];
      const out=box(ids||"无命中",n+" × [1,6144]","Gate/Up 各 6144→2048；SiLU(g)⊙u 后 Down 2048→6144。");
      const sum=box("局部贡献 s"+r,n?"[1,6144]":"零 [1,6144]","本卡对 A 第 7 个 token 的 "+(n?n+" 个专家输出加权求和":"零贡献"),[r]);
      if(f===2)return [weights,box("输入 h","[1,6144]","本卡读取 G6 的目标行，使用完整隐藏维。",[6]),out];
      if(f===3)return [weights,out,sum];
      if(f===4)return [sum,!before && r===6?box("归还给 P6","Σ sᵣ [1,6144]","仅跟踪 A 第 7 行的归并终点；其余 token 归各自 rank。",ranks):blank(),blank()];
      return r===6?[box("路由分支","Σ sᵣ [1,6144]","路由专家归并结果"),box("共享分支","shared(h)","共享专家对同一输入的输出 [1,6144]"),box("MoE 输出","y [1,6144]","y = Σ sᵣ + shared(h)")]:[weights,blank(),blank()];
    }
    const weight=box(side==="P"?"路由专家权重":"专家中间维",side==="P"?"E"+32*r+"–"+(32*r+31):"["+256*r+","+(r+1)*256+")","模型权重驻留在显存中。");
    if(step===1 && side==="P")return [box("输入位置","G"+r,"交错分片中的 token 位置。",[r]),box("隐藏状态","["+localRows+",6144]","本 rank 的逻辑行分片。"),box("KV 缓存","写入本地槽","提示词阶段产生的缓存。")];
    if(step===4 && (side==="P" || r<requests))return [weight,box(side==="P"?"源 KV 缓存":"目标 KV 缓存",side==="P"?"Prefill KV":"请求 "+req,"逻辑缓存端点；实际跨机分片需运行记录确认。"),blank()];
    if(step===5 && side==="D" && r<requests)return [box("请求 "+req,"h [1,6144]","当前 token 的隐藏状态。"),box("Q / KV","Q 64×256","低秩 Q 与压缩 KV 投影。"),box("注意力输出","[1,6144]","缓存位于 D"+r+"。")];
    if(step===6 && side==="D")return [weight,box("SwiGLU","[1,256]","本卡的专家中间维分片。"),box("Down 部分积","[1,6144]","各 rank 的部分积需要求和。")];
    return [weight,blank(),blank()];
  }
  let pick=null;
  function showBlock(side,r,row,f){
    const b=blocks(side,r,f)[row];
    return '<div class="linkdetail"><b>'+side+r+' · '+b.name+'</b><strong>'+b.value+'</strong><span>'+b.info+'</span></div>';
  }
  function render(f,focus){
    let panels="",tiles="",wires="";
    for(const side of ["P","D"]){
      const top=side==="P"?0:344;
      panels+='<rect class="machine" x="5" y="'+(top+2)+'" width="1146" height="332" rx="14"/>'+text(26,top+29,side==="P"?"节点 01   /   Prefill":"节点 02   /   Decode","machTitle")+text(310,top+29,side==="P"?"8 × H200     TP 8 · EP 8":"8 × H200     TP 8 · DP 8 · EP 1","machineMeta");
      for(const r of ranks){
        const left=x(r), bs=blocks(side,r,f);
        panels+='<rect class="column" x="'+(left-5)+'" y="'+(top+36)+'" width="134" height="286" rx="7"/>'+text(left+4,top+51,side+r,"rankId");
        bs.forEach((b,j)=>{
          const cy=y(side,j), selected=pick && pick.side===side && pick.r===r && pick.row===j;
          tiles+='<g role="button" tabindex="0" data-slot="'+side+','+r+','+j+'" class="slot '+(b.idle?'idle ':'')+(selected?'selected ':'')+(side==="P" && f!==null && ((r===6 && j===0)||(r===focus && j===1))?'traceSlot':'')+'"><title>'+side+r+' · '+b.name+' '+b.value+'</title><rect x="'+left+'" y="'+cy+'" width="124" height="52" rx="5"/>'+text(left+7,cy+18,b.name,"slotName")+text(left+7,cy+37,b.value,"slotValue")+((b.tags||[]).map((t,i)=>'<rect x="'+(left+6+i*14)+'" y="'+(cy+43)+'" width="12" height="5" fill="'+palette[t]+'" rx="1"/>').join(""))+'</g>';
        });
      }
    }
    function wire(a,ar,b,br,family,peer,chosen){
      const sx=x(a)+62, tx=x(b)+62, sy=y(family,ar)+53, ty=y(family,br)-2;
      const lane=sy+9+peer*4, d=sx===tx?'M'+sx+' '+sy+' V'+ty:'M'+sx+' '+sy+' V'+lane+' H'+tx+' V'+ty;
      const color=f===4?palette[a]:palette[f===1?6:peer];
      return '<g data-side="'+family+'" class="edge blockEdge '+(chosen?'chosen':'')+'" tabindex="0" role="button" data-wire="'+peer+'" aria-label="'+family+a+' 到 '+family+b+'"><title>'+family+a+' → '+family+b+'</title><path d="'+d+'" class="hit"/><path d="'+d+'" class="line" style="stroke:'+color+'"/><path d="M'+(tx-4)+' '+(ty-7)+' l4 7 4 -7" fill="none" stroke="'+color+'" stroke-width="2"/>'+(chosen?'<circle class="packet" r="4" fill="'+color+'"><animateMotion path="'+d+'" dur="2.2s" repeatCount="indefinite"/></circle>':'')+'</g>';
    }
    if(f!==null){
      for(const r of [...ranks.filter(r=>r!==focus),focus]){
        if(f===1 && !before)wires+=wire(6,0,r,1,"P",r,r===focus);
        if(f===4 && !before)wires+=wire(r,0,6,1,"P",r,r===focus);
        if(f===0 || (f===2 && expertsOn(r).length) || f===3)wires+=wire(r,1,r,2,"P",r,r===focus);
        if(f===0)wires+=wire(r,0,r,1,"P",r,r===focus);
        if(f===5 && r===6)wires+=wire(6,1,6,2,"P",6,true);
      }
    }
    if(f===null){
      const side=step===1?"P":"D";
      if([1,5,6].includes(step)){
        for(const r of ranks.filter(r=>step!==5||r<requests)){
          wires+=wire(r,0,r,1,side,r,false)+wire(r,1,r,2,side,r,false);
        }
      }
      if(step===4){
        // Only logical cache endpoints are known. Avoid claiming a physical peer mapping.
        const bus=340;
        let endpoints="";
        for(const r of ranks){
          const px=x(r)+118,py=y("P",1)+52;
          endpoints+='<path d="M'+px+' '+py+' h10 V'+bus+'"/>';
          if(r<requests){
            const target=y("D",1)+25;
            endpoints+='<path d="M'+(px+10)+' '+bus+' V'+target+' h-7"/>';
          }
        }
        wires+='<g class="kvBus"><path d="M'+(x(0)+128)+' '+bus+' H'+(x(7)+128)+'"/>'+endpoints+'</g>';
      }
    }
    return '<div class="blockScroll"><svg class="blockMap" viewBox="0 0 1156 682" aria-label="两台机器，每台八列 GPU，每列三个数据块">'+panels+tiles+wires+'</svg></div>';
  }
  return {render,showBlock,get before(){return before;},set before(value){before=value;},select(side,r,row){pick={side,r,row};},clear(){pick=null;},get pick(){return pick;}};
})();
