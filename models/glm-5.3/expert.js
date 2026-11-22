// Logical CP8 walkthrough. Rank-to-rank edges describe contributions, not NCCL hops.
const expertTrace = (() => {
  const source = 6; // illustrative interleave: position 6 (the seventh token)
  const frames = ["路由评分", "分送激活", "专家计算", "卡内加权", "归还结果", "合入共享专家"];
  let frame = 1, focusRank = 2;
  const sequence = [1, 0, 2, 3, 4, 5];


  const expertLabel = r => expertsOn(r).map(e => "E" + e).join(" + ") || "无命中";
  const titles = ["路由器选出 8 个专家", "AllGather · 分发完整激活", "SwiGLU · 各卡计算本地专家", "本地求和 · 每卡一个部分向量", "ReduceScatter · 求和并归还 token 行", "合并路由分支与共享分支"];
  function detail(){
    const r=focusRank, local=r===source, ids=expertsOn(r), n=ids.length;
    const name="P"+r, slice="h[0:6144]", ys=ids.map(e=>'w'+e+'·y'+e).join(" + ") || "0";
    const title = frame===1||frame===4 ? (local?'P6 本卡':frame===1?'P6 → '+name:name+' → P6') : name+' · '+expertLabel(r);
    let cells;
    switch(frame){
      case 0: cells=[["路由输入","h [1,6144]"],["门控投影","h × Wᵀ → [1,256]"],["Top-8","ID [1,8] · 权重 [1,8]"],["本卡命中",expertLabel(r)]];break;
      case 1: cells=[["发送侧",local?"P6 保留本地行":"P6 · A 的第 7 行"],["负载",slice+" → [1,6144]"],["行布局","P"+r+" 原有位置 "+r+", "+(r+8)+"；收齐全批行"],["接收侧",name+" · 完整隐藏维"],["用途",n?expertLabel(r)+" 共用该行":"保留该行；路由后无专家命中"]];break;
      case 2: cells=n?[["Gate / Up","[1,6144] × [6144,2048] 各一次"],["激活","SiLU(gate) ⊙ up → [1,2048]"],["Down","[1,2048] × [2048,6144]"],["本卡输出",n+" 个专家，各 [1,6144]"]]:[["本卡计算","跳过未命中的专家"],["部分输出","0 [1,6144]"]];break;
      case 3: cells=[["专家输出",n+" × [1,6144]"],["本地加权",ys],["本地结果","s"+r+" [1,6144]"]];break;
      case 4: cells=[["贡献端",name+" · "+ys],["逻辑负载","s"+r+"[0:6144]"],["归并","Σᵣ sᵣ → [1,6144]"],["结果归属","P6 · 第 7 行，保留全部通道"]];break;
      case 5: cells=[["路由输出","Σᵣ sᵣ [1,6144]"],["共享专家","shared(h) [1,6144]"],["本 token 输出","P6 · y = routed + shared"]];break;
    }
    return '<div class="linkdetail"><b>'+title+'</b>'+cells.map(([k,v])=>'<div><small>'+k+'</small><strong>'+v+'</strong></div>').join("")+'</div>';
  }
  function render(){
    const t=requests*16;
    const sizes=["每行 256 个评分 → 8 个 ID 与系数", "每卡 ["+t/8+",6144] → 每卡 ["+t+",6144]","专家按编号分卡 · 隐藏维完整保留","每卡每行 [1,6144] · 未命中为零","每卡 ["+t+",6144] → 每卡 ["+t/8+",6144]","A 的位置 6 → P6 · 输出 [1,6144]"];
    return '<div class="trace"><div class="tracehead"><div><small>节点 01　/　PREFILL · EP8</small><h2>'+titles[frame]+'</h2></div><span>请求 A · token 7 / 16</span></div><div class="beats">'+sequence.map((i,position)=>'<button data-beat="'+i+'" aria-pressed="'+(i===frame)+'">'+(position+1)+' '+frames[i]+'</button>').join("")+'</div><div class="tracebar"><b>'+sizes[frame]+'</b><span>每列一张 GPU · 点块或线</span></div>' + ((frame===1||frame===4)?'<div class="snapshot"><span>缓冲区状态</span><button data-snapshot="before" aria-pressed="'+blockMap.before+'">通信前</button><button data-snapshot="after" aria-pressed="'+(!blockMap.before)+'">通信后</button><span>色块 G0–G7 对应源卡，金色 G6 含当前 token</span></div>':'')+''+blockMap.render(frame,focusRank)+(blockMap.pick?blockMap.showBlock(blockMap.pick.side,blockMap.pick.r,blockMap.pick.row,frame):detail())+'<footer><button data-beat="'+(sequence[sequence.indexOf(frame)-1] ?? -1)+'" '+(frame===1?'disabled':'')+'>上一拍</button><span>'+(sequence.indexOf(frame)+1)+' / 6</span><button data-beat="'+(sequence[sequence.indexOf(frame)+1] ?? 6)+'" '+(frame===5?'disabled':'')+'>下一拍</button></footer><p class="scope">CP8 交错布局教学示例 · 线表示一个 token 在集合通信中的逻辑贡献。实际 NCCL 跳数、融合方式与运行时分块需由 trace 确认。归并系数包含路由缩放。</p></div>';
  }
  return {render, select(r){focusRank=r;}, set(i){frame=i;}, get frame(){return frame;}};
})();
