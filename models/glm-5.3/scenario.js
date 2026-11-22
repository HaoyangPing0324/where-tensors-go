// Matrices use the mathematical [input, output] convention throughout.
const glm53 = {
 id:"glm-5.3", name:"GLM-5.3", ranks:8,
 parallel:{
  hardware:"2 台机器 · 每台 8 张 GPU",
  schemeLabel:"Prefill 注意力分片",
  schemes:[{id:"tp",label:"按头切分（TP8）· 每卡 8 头"},{id:"cp",label:"按位置交错切分（CP8）· 每卡 2 个位置"}],
  columns:["节点","TP","DP","PP","CP","注意力 TP","EP","单专家 TP"],
  rows(layout){
   const cp=layout==="cp";
   return [
    ["Prefill",8,1,1,cp?8:1,cp?1:8,8,1],
    ["Decode",8,8,1,1,1,1,8]
   ];
  }
 },
 stages:[
 ["Embedding","词表分片查表","查表","Tokenizer 已生成整数 ID。八卡读取相同 ID；每卡只查自己持有的词表行，未命中位置填零。"],
 ["Embedding","合并查表结果","AllReduce · SUM","同一位置只有一个分片提供非零向量。逐元素求和后，每卡获得相同的 X。"],
 ["Attention","残差与输入归一化","RMSNorm","保存残差。按最后一维归一化，每个 token 都保留 6144 个通道。"],
 ["Attention","Q / KV 低秩投影","GEMM","复制的 A 投影一次产生 Q latent、KV latent 和旋转位置编码使用的 key。"],
 ["Attention","展开 Q · 拆分 KV latent","列并行投影","Q、KV latent 分别归一化。Q 经过 B 投影，KV 内容以 512 维 latent 表示，位置通道单独保留。"],
 ["Attention","恢复各头的 value 输出","MLA · V up","候选位置的 value 在 512 维潜空间聚合，再经每头的 value 投影恢复输出。展示吸收式 MLA 的等价矩阵运算。"],
 ["Attention","输出投影 Wₒ","GEMM","把各头的注意力结果串接，乘 Wₒ 得到 6144 维表示。"],
 ["Attention","合并投影并加残差","层边界","按头切分时，各卡产生同一输出的部分和；按位置切分时，各卡直接持有本地位置的完整输出。"],
 ["前馈","经过前三个 Dense 层","SwiGLU","第 0–2 层使用 Dense FFN。从第 3 层起使用 MoE。此处折叠重复层，下一步展示一个 MoE 层的 Attention 后边界。"],
 ["MoE","准备专家输入","Dispatch","此部署的 moe-a2a-backend=none。演示按需要汇集 token 行，让各卡能读取本地专家需要的完整隐藏向量。"],
 ["MoE","Top-8 路由","Router","每个 token 对 256 个路由专家评分，选 8 个；共享专家独立计算。编号按连续 32 个专家分卡示意。"],
 ["MoE","按专家打包命中行","Permute","按本地专家 ID 打包命中行，保留 token 位置与路由权重，供后续还原顺序。"],
 ["MoE","合并专家输出","Combine · SUM","各卡先对命中的专家加权求和，再跨卡求和，合入共享专家和残差。"],
 ["PD 交接","传输请求 KV","NIXL / UCX","折叠其余层与末层采样。Prefill 产生首个 token，跨机搬运请求缓存到 Decode 负责该请求的 rank。"],
 ["Decode","为活跃请求计算注意力","DP Attention","普通自回归示例每请求推进一个位置。每个 DP rank 对本地请求使用完整注意力头与本地 KV。"],
 ["Decode","八卡协作计算专家","专家 TP8","EP=1。每个命中专家的中间维切成八段；同一个专家的权重分片和激活乘积同时分布在八卡上。"],
 ["Decode","求和与采样返回","SUM → 采样","对专家部分积做逻辑 SUM 并合路由权重，经过余下层、末层归一化和词表投影后采样，输出 token ID，解码为文本。"]
 ],
 describe(i,r,side,n,layout){
  const T=16*n, cp=layout==="cp", t=cp?T/8:T, heads=cp?64:8, h0=cp?0:8*r;
  const b=(name,value,detail,colors=[])=>({name,value,detail,colors});
  const all=[0,1,2,3,4,5,6,7], own=[r], slice=(a,z)=>"["+a+","+z+")";
  const range=slice(19360*r,19360*(r+1)), es="E"+32*r+"–"+(32*r+31);
  const X="["+t+",6144]", global="["+T+",6144]", sh="["+T/8+",6144]";
  const head="头 "+h0+"–"+(h0+heads-1);
  const positions=cp?"位置 "+r+", "+(r+8)+" …":"全部 "+T+" 行";
  const inactive=[b("常驻权重",side==="D"?"专家分片 "+(r+1)+"/8":es,"本阶段模型已驻留在 GPU 显存中。"),b("请求数据","—","此节点未承担本步运算。"),b("本步输出","—","本阶段无新输出。")];
  if(i===13){
   if(side==="D" && r>=n)return inactive;
   return [b("请求归属",side==="P"?"Prefill 缓存":"请求 "+String.fromCharCode(65+r),"示例将活跃请求按序分配给 Decode rank。"),b("传输对象","KV / 索引缓存","具体页表与传输分片由后端决定。"),b(side==="P"?"源缓存":"目的缓存",side==="P"?"等待读取":"本地缓存槽","跨节点走 NIXL / UCX；图示逻辑端点。")];
  }
  if((i<13 && side==="D")||(i>13 && side==="P"))return inactive;
  const ids=Array.from({length:T},(_,j)=>19360*(j%8)+7+Math.floor(j/8));
  const idSample=ids.filter(id=>Math.floor(id/19360)===r);
  const match=idSample.slice(0,2).join(", ");
  switch(i){
   case 0:return [b("输入 ID","ids ["+T+"]","本批示例 ID："+ids.slice(0,16).join(", ")+"。每卡都有同一批 ID。",all),b("词表权重 Eᵣ",range,"Eᵣ [19360,6144]，全表 [154880,6144]。查本地行 id − "+19360*r+"。",own),b("局部 Xᵣ",global,"命中 "+match+" 等 "+idSample.length+" 行；其余位置填零。",own)];
   case 1:return [b("局部 Xᵣ",global,"非零 token 行数 "+T/8,own),b("贡献行",match,"各卡对齐同一批 token 位置。",own),b("完整 X",global,"X = Σ Xᵣ，每个位置对应一个完整 embedding。",all)];
   case 2:return [b("完整 X",global,"保留残差。",all),b("位置布局",positions,"CP 按序列位置交错；按头 TP 则每卡复制输入。",cp?own:all),b("归一化结果 X̂",X,"除以均方根后乘学习得到的通道尺度。")];
   case 3:return [b("归一化输入 X̂",X,"沿用上一算子的 RMSNorm(X)。"+positions),b("Wₐ · 复制","[6144,2624]","按输出拼接 2048 + 512 + 64 个通道。"),b("qₐ | cₐ | kᴿ","2048 | 512 | 64","每行 Q latent [2048]、KV latent [512]、k_rope [64]。")];
   case 4:return [b("Q latent","["+t+",2048]","RMSNorm 后输入 Q B 投影。"),b("Wqᵦ",head,"本地权重 [2048,"+heads*256+"]；KV 内容保留为压缩表示。"),b("本地 Q","["+t+","+heads+",256]","每头有 192 维内容通道和 64 维 RoPE 通道。")];
   case 5:return [b("潜空间结果","["+heads+",512] / 行","注意力概率对候选 KV latent 加权求和。"),b("Wᵥᵦ / 头","[512,256]","每头从 512 维 latent 投影到 256 维 value 输出。"),b("Attention 输出","["+t+","+heads+",256]","各头恢复为 256 通道后进入 Wₒ。")];
   case 6:return [b("Attention 输出","["+t+","+heads*256+"]",head),b("Wₒ 输入维切片",cp?"[16384,6144]":slice(r*2048,(r+1)*2048),"本地 Wₒ ["+heads*256+",6144]；数学布局 [输入维,输出维]。"),b(cp?"本地完整行":"本卡部分和",X,cp?"已包含全 64 个头的贡献。":"仅含本卡 8 个头的投影贡献。",own)];
   case 7:return [b("投影结果",X,cp?"按位置分片。":"按头计算的部分积。",own),b(cp?"本地相加":"AllReduce SUM",cp?positions:"Σ 八份部分和","与之前保存的残差相加。"),b("FFN 输入",X,"再经 RMSNorm 进入前馈子层。",cp?own:all)];
   case 8:return [b("Dense 输入","["+T+",6144]","逻辑全批，CP 行在前馈边界汇集。"),b("中间维分片",slice(1536*r,1536*(r+1)),"Gate/Up 各 [6144,1536]，Down [1536,6144]。"),b("加权与残差",X,"归并八个 Down 部分积，恢复层边界的位置布局。")];
   case 9:return [b("归一化行",X,positions,cp?own:all),b(cp?"AllGather":"复制态输入",cp?"合并八组行":"各卡已有全批","此处的 dispatch 为输入准备。"),b("专家可读 X",global,"每个 expert rank 读取全批隐藏行。",all)];
   case 10:return [b("X",global,"每个 token 独立路由。"),b("Wrouter","[6144,256]","X × Wrouter → ["+T+",256]，sigmoid 评分后选取 Top-8。"),b("专家与权重","["+T+",8]","ID 与系数各一份；本卡负责 "+es+"。")];
   case 11:{const assignments=[];for(let j=0;j<T;j++){for(let k=0;k<8;k++){const e=(j*17+k*29)%256;if(Math.floor(e/32)===r)assignments.push([j,e]);}}const a=assignments[0];return [b("本地专家",es,"本卡持有 32 个完整专家。"),b("命中表",assignments.length+" 条","确定性示例，token "+a[0]+" → E"+a[1]+"；按专家聚集输入行。"),b("打包的输入","["+assignments.length+",6144]","同一 token 命中多个专家时可出现多行。")];}
   case 12:return [b("本卡局部和",global,"对各 token 计算 Σ wₑ yₑ；无命中行贡献零。",own),b(cp?"ReduceScatter":"AllReduce","Σ 专家贡献","归并八个 rank 的部分和。"),b("层输出",X,"路由贡献 + 共享专家输出 + 残差。",cp?own:all)];
   case 14:return r>=n?inactive:[b("当前行","请求 "+String.fromCharCode(65+r),"每轮一个普通解码位置。"),b("Q 与本地 KV","64 头 × 256","KV latent 512 + RoPE 64，读取历史缓存并更新。"),b("输出投影","[1,6144]","本 rank 计算其请求的注意力输出。")];
   case 15:return [b("专家输入","h [1,6144]","每个命中 token 的完整隐藏维。"),b("中间维",slice(256*r,256*(r+1)),"Gate / Up 各 [6144,256]；SiLU(gate) ⊙ up。",own),b("局部 Down","[1,6144]","本卡中间分片乘 Wdown [256,6144]。",own)];
   case 16:return [b("各卡贡献","[1,6144]","对同一专家的部分积归并。",own),b("逻辑归并","Σ 局部 Down","框架可融合求和与布局转换。"),b("返回请求端",r<n?"请求 "+String.fromCharCode(65+r):"—","在所属 rank 推进请求，前端解码并流式返回。")];
  }
 },
 op(i,layout){if(i===1)return "AllReduce · SUM";if(i===7)return layout==="cp"?"本地残差相加":"AllReduce · SUM → 残差";if(i===9)return layout==="cp"?"AllGather · 按 token 行汇集":"全批输入已复制";if(i===12)return layout==="cp"?"ReduceScatter · SUM":"AllReduce · SUM";if(i===13)return "NIXL / UCX · InfiniBand";if(i===16)return "专家部分积 · SUM";return ""; }
};
window.inferenceModels[glm53.id]=glm53;

// Expand the attention and expert operators into individually selectable matrix operations.
{
 const base=glm53.stages, describe=glm53.describe.bind(glm53), op=glm53.op.bind(glm53);
 const plan=[], stages=[];
 function add(parent,title,formula,explain,fn){
   stages.push([base[parent][0],title,formula,explain]);
   plan.push({parent,fn});
 }
 function matrix(parent,title,formula,explain,fn){add(parent,title,formula,explain,(r,t,H,start,n,cp)=>fn(r,t,H,start,n,cp).map(([name,value,detail])=>({name,value,detail:detail||explain})));}
 for(let p=0;p<base.length;p++){
   stages.push(base[p]);plan.push({parent:p});
   if(p===4){
     matrix(p,"拆出 Q 的内容与位置通道","Split / RoPE","每头前 192 维为内容分量，另 64 维通过 RoPE 旋转。",(_,t,H)=>[["Q",t+"×"+H+"×256"],["切分","192 | 64"],["Qᶜ 与 Qʳ",H+" 头"]]);
     matrix(p,"写入 MLA 压缩缓存","Cache write","对 KV latent 做 RMSNorm，为 64 个位置通道应用 RoPE。每个位置保存一份 512+64 维逻辑缓存，DSA 索引键另存。",(_,t)=>[["KV latent",t+"×512"],["RoPE key",t+"×64"],["缓存行","512 + 64"]]);
     matrix(p,"生成 DSA 索引查询","Indexer","full 索引层使用 32 个索引头，每头 128 维。shared 层复用之前的候选索引；配置中层 0/1/2 为 full，层 3/4/5 为 shared，层 6 再次 full。",(_,t)=>[["索引输入","Q latent / X"],["索引投影","32 头 × 128"],["索引查询","["+t+",32,128]"]]);
     matrix(p,"选择或复用稀疏地址","DSA · Top-k","对可见位置计算索引分数并聚合索引头，最多选 2048 个位置。本例每请求只有 16 个提示 token，因此 L 不超过 16；共享层沿用索引。",()=>[["索引键","每位置 128 维"],["索引评分","causal · top-k"],["候选位置","L ≤ 16"]]);
     matrix(p,"吸收 K 的上投影","MLA · Q absorption","按头将 192 维内容查询投到 512 维潜空间，随后直接与压缩缓存计算点积。64 维 RoPE 查询保持独立。",(_,t,H)=>[["内容查询",H+"×192 / 行"],["Wkc / 头","[192,512]"],["潜空间 Q",H+"×512 / 行"]]);
     matrix(p,"计算稀疏注意力分数","DSA × MLA","只在选中的 L 个地址上计算 Qlatent·Clatentᵀ + Qrope·Kropeᵀ。",(_,t,H)=>[["查询 / 头","512 + 64"],["缓存 / 候选","L × (512+64)"],["分数","["+H+",L]"]]);
     matrix(p,"缩放、因果掩码与归一化","Softmax","沿候选位置维做 softmax，每个头的概率和为 1。",(_,t,H)=>[["分数 S",H+"×L"],["scale 与 mask","softmaxₗ(S)"],["概率 A",H+"×L"]]);
     matrix(p,"聚合候选 latent","A × C","每头用自身的注意力概率对相同候选位置的压缩 latent 求加权和。",(_,t,H)=>[["概率 A",H+"×L"],["所选 C","L×512"],["加权 latent",H+"×512"]]);
   }
   if(p===12){matrix(p,"合入共享专家与残差","Add","共享专家也执行 SwiGLU，路由贡献乘配置中的缩放系数 2.5。缩放可融合进路由权重。",(_,t)=>[["路由输出",t+"×6144"],["共享与残差","shared(h) + x"],["层输出",t+"×6144"]]);}
   if(p===11){
     matrix(p,"计算专家 Gate 与 Up","两次矩阵乘","mₑ 是本批命中该专家的行数。路由专家独立保存两组投影，每个 rank 上有 32 个专家。",r=>[["专家输入 Xₑ","mₑ×6144"],["Wg / Wu","各 6144×2048"],["g 与 u","各 mₑ×2048"]]);
     matrix(p,"专家门控激活","SiLU(g) ⊙ u","SiLU 逐元素作用在 gate 投影上，与 up 投影逐元素相乘。",()=>[["g 与 u","mₑ×2048"],["门控","SiLU(g) ⊙ u"],["中间激活","mₑ×2048"]]);
     matrix(p,"专家输出投影","中间激活 × Wdown","本卡每个被选专家将中间维投回隐藏维，然后按 token 的路由系数加权。",()=>[["中间激活","mₑ×2048"],["Wdown","2048×6144"],["专家输出","mₑ×6144"]]);
   }
 }
 glm53.stages=stages;
 glm53.phase=i=>plan[i].parent;
 glm53.describe=(i,r,side,n,layout)=>{
   const {parent,fn}=plan[i];
   if(!fn || side==="D")return describe(parent,r,side,n,layout);
   const cp=layout==="cp";return fn(r,cp?2*n:16*n,cp?64:8,cp?0:r*8,n,cp);
 };
 glm53.op=(i,layout)=>plan[i].fn?"":op(plan[i].parent,layout);
}

// Ordinary decode traverses the same stack, with one query position per active request.
{
 const first=glm53.stages.findIndex(s=>s[0]==="Decode");
 const previousDescribe=glm53.describe, previousPhase=glm53.phase, previousOp=glm53.op;
 const local=14, team=15;
 const steps=[];
 function add(phase,title,operation,detail,blocks,protocol=""){
   steps.push({phase,title,operation,detail,blocks,protocol});
 }
 add(local,"读取新 token 与历史 KV","Embedding","普通解码的第一轮使用 Prefill 采样的 token ID。历史 16 个位置的逐层 KV 已在请求所属 rank。",(r,q)=>[["当前 token","请求 "+q],["查表","词表 → 6144"],["新位置 X","[1,6144]"]]);
 add(local,"输入归一化与低秩投影","RMSNorm / GEMM","保留残差，投影出 Q latent、KV latent 和旋转位置通道。",()=>[["新位置","[1,6144]"],["Wₐ","[6144,2624]"],["Q / KV / RoPE","2048 | 512 | 64"]]);
 add(local,"展开查询与追加缓存","MLA / RoPE","DP Attention 的活跃 rank 处理完整 64 个头。只计算新位置，并追加这一层的压缩 KV。",()=>[["Q latent","[1,2048]"],["Q B 投影","64 × 256"],["KV 追加","512 + 64"]]);
 add(local,"生成或复用稀疏地址","DSA · Top-k","full 层计算索引，shared 层复用候选。第一轮可见 16 个历史位置和 1 个新位置。",()=>[["索引查询","32 头 × 128"],["历史与新 KV","17 个位置"],["候选索引","L ≤ 17"]]);
 add(local,"吸收 K 投影，计算候选分数","MLA × DSA","每头将 192 维内容查询投到 512 维，与候选 latent 点积，再加 64 维 RoPE 点积。",()=>[["Q 内容 / 位置","192 | 64"],["每头 Wkc","[192,512]"],["候选分数","[64,L]"]]);
 add(local,"归一化并聚合候选 latent","Softmax / AV","每头沿候选位置归一化，用注意力概率对候选压缩 latent 加权。",()=>[["候选分数","[64,L]"],["A × C","[64,L]×[L,512]"],["加权 latent","[64,512]"]]);
 add(local,"恢复 value、输出投影与残差","V up / Wₒ","64 个头均在本请求的 DP rank。恢复每头 value 后经输出投影，再加残差。",()=>[["每头 Wᵥ","[512,256]"],["Wₒ","[16384,6144]"],["层内隐藏状态","[1,6144]"]]);
 add(team,"前三层的 Dense 前馈","SwiGLU · TP8","层 0–2 的前馈为 Dense。每张卡负责 12288 维中间表示的 1/8。输入行在前馈边界对齐，Down 部分积归并后加残差。",r=>[["前馈输入","RMSNorm(X)"],["中间维切片",r*1536+"…"+((r+1)*1536-1)],["Down 部分积","每行 6144"]]);
 add(team,"MoE 路由与专家输入","Router · Top-8","从第 3 层起，每个 token 选择 256 个路由专家中的 8 个。EP=1，八卡按 TP 分担被选专家的计算；另有共享专家。",()=>[["归一化行","每行 6144"],["专家评分","256 个"],["所选专家","8 个 ID + 权重"]]);
 add(team,"八卡计算专家中间分片","Gate / Up / SiLU","每卡保留每个专家 2048 个中间通道中的 256 个。",r=>[["完整输入 h","[1,6144]"],["Wg / Wu","各 [6144,256]"],["本卡通道","["+r*256+","+(r+1)*256+")"]]);
 add(team,"Down 投影与专家部分和","GEMM","本卡中间维乘本卡的 Down 权重分片。按路由系数加权，形成待归并的部分积。",()=>[["中间激活","[1,256]"],["Wdown","[256,6144]"],["专家部分积","[1,6144]"]]);
 add(team,"归并 TP 部分积","AllReduce · SUM","按 SUM 的等价环形实现演示。此处每个 B 块覆盖 768 个隐藏通道，含本批所有请求行。实际归并可与布局转换融合。",()=>[],"AllReduce · SUM");
 add(local,"合并分支并进入下一层","Residual","归并后将本请求的路由分支、共享分支与残差合成。上述层结构重复到第 78 层。",()=>[["路由结果","[1,6144]"],["共享分支","shared(h)"],["下一层输入","[1,6144]"]]);
 add(local,"末层归一化与词表投影","LM head","完成全部层后得到当前行的逻辑词表 logits；这里折叠最后投影的物理布局。",()=>[["末层状态","[1,6144]"],["RMSNorm","6144 个通道"],["逻辑 logits","[1,154880]"]]);
 add(local,"采样与流式返回","Sample","采样一个 token ID，经服务端解码为文本。尚未遇到停止条件时，下一轮复用已增长的 KV，再走一遍模型。",()=>[["词表分数","154880"],["采样结果","1 个 token ID"],["请求端","文本片段"]]);
 glm53.stages.splice(first,Infinity,...steps.map(s=>["Decode",s.title,s.operation,s.detail]));
 glm53.phase=i=>i<first?previousPhase(i):steps[i-first].phase;
 glm53.op=(i,l)=>i<first?previousOp(i,l):steps[i-first].protocol;
 glm53.describe=(i,r,side,n,l)=>{
   if(i<first)return previousDescribe(i,r,side,n,l);
   const s=steps[i-first], q=String.fromCharCode(65+r);
   if(side==="P" || (s.phase===local && r>=n))return previousDescribe(first,r,side,n,l);
   return s.blocks(r,q).map(([name,value])=>({name,value,detail:s.detail}));
 };
}

// Present the execution phase first, then the transformer sublayer and its implementation.
{
 const handoff=glm53.stages.findIndex(s=>s[0]==="PD 交接");
 const decode=glm53.stages.findIndex(s=>s[0]==="Decode");
 const section=[
   ["Embedding"],["Attention"],["Attention"],["Attention"],["Attention"],["Attention"],["Attention"],
   ["FFN","Dense"],["FFN","MoE"],["FFN","MoE"],["FFN","MoE"],["FFN","MoE"],["FFN","MoE"],["输出"],["输出"]
 ];
 const path=glm53.stages.map(([group],i)=>{
   if(i>=decode)return ["Decode",...section[i-decode]];
   if(group==="PD 交接")return ["Prefill","KV 交接"];
   if(group==="前馈"||group==="MoE")return ["Prefill","FFN",group==="前馈"?"Dense":"MoE"];
   return ["Prefill",group];
 });
 const prevDescribe=glm53.describe,prevOp=glm53.op,prevPhase=glm53.phase;
 glm53.stages.splice(handoff,0,["输出","首个输出 token","LM head / Sample","末层归一化后，将每个请求最后一个提示位置投影到词表并采样。这个首个生成 token 将作为 Decode 第一次前向的输入。此处展示逻辑张量。"]);
 path.splice(handoff,0,["Prefill","输出"]);
 glm53.path=i=>path[i];
 const old=i=>i>handoff?i-1:i;
 glm53.phase=i=>i===handoff?12:prevPhase(old(i));
 glm53.op=(i,l)=>i===handoff?"":prevOp(old(i),l);
 glm53.describe=(i,r,side,n,l)=>{
   if(i!==handoff||side==="D")return prevDescribe(old(i),r,side,n,l);
   return [["末位置隐藏态","["+n+",6144]"],["逻辑 logits","["+n+",154880]"],["采样","每请求 1 个 ID"]].map(([name,value])=>({name,value,detail:"全组逻辑形状；此步未展开输出头分片。"}));
 };
}

// Exact, zero-based layer classes from config.json: mlp_layer_types × indexer_types.
glm53.layerClasses=[
 {ffn:"Dense",index:"full",label:"0–2 层 · Dense FFN · 计算索引"},
 {ffn:"MoE",index:"shared",label:"3–5、7–9、…、71–73、75–77 层 · MoE · 复用索引"},
 {ffn:"MoE",index:"full",label:"6、10、14、…、74 层 · MoE · 计算索引"}
];
glm53.layerAllows=(i,group)=>{
 const path=glm53.path(i);
 if(path[1]==="FFN")return path[2]===group.ffn;
 return !(group.index==="shared" && glm53.stages[i][1]==="生成 DSA 索引查询");
};
glm53.layerPresentation=(i,group)=>{
 const title=glm53.stages[i][1];
 if(!["生成或复用稀疏地址","选择或复用稀疏地址"].includes(title))return null;
 const decode=glm53.path(i)[0]==="Decode",len=decode?17:16;
 const shared=group.index==="shared";
 return {
  title:shared?"读取前层的 DSA 候选索引":"计算本层的 DSA 候选索引",
  desc:shared?"本层沿用最近一个 full 索引层的候选位置。仍然用本层的查询、KV 与这些地址计算注意力。":"本层使用索引查询与历史索引键打分，按可见位置选取至多 2048 个候选，供本层及后续 shared 层使用。",
  blocks:(shared?[["已有地址","前层 Top-k"],["读取地址","同一请求"],["复用候选","L ≤ "+len]]:[["索引查询","32 头 × 128"],["索引评分","因果 Top-k"],["本层候选","L ≤ "+len]]).map(([name,value])=>({name,value,detail:shared?"沿用地址，不复用注意力结果。":"新计算的 token 位置索引。"}))
 };
};
