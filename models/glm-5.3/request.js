// Service events surround the existing GPU walkthrough. Keep the model's layer indices intact.
{
 const model=window.inferenceModels["glm-5.3"];
 const events=[];
 function service(phase,section,title,from,via,to,detail){
   events.push({path:[phase,section],stage:[section,title,via,detail],from,via,to});
 }
 service("Prefill","请求接入","用户发送流式聊天请求","客户端","POST /v1/chat/completions","Dynamo Frontend","示例使用 stream=true、max_tokens=2，演示提示未命中前缀缓存的正常路径。");
 service("Prefill","请求接入","校验请求并应用聊天模板","messages 与采样参数","校验模型名、套用 chat template","带角色标记的提示文本","前端检查输入，将角色、消息和生成提示组装成模型可用的输入。");
 service("Prefill","请求接入","分词并建立生成上下文","提示文本","Tokenizer","prompt token IDs","GPU 接收整数 ID。页面统一采用每请求 16 个教学 ID，未对某段真实输入做分词。");
 service("Prefill","路由与组批","选择 Prefill worker","Token IDs 与请求信息","KV-aware Router","Prefill worker","路由综合缓存重合和负载。本拓扑只有一个 Prefill worker；worker 内八张卡共同工作。");
 service("Prefill","路由与组批","排队、检查前缀与缓存预算","SGLang 等待队列","Scheduler / 前缀缓存","可执行请求与 KV 槽","检查可复用前缀和本轮 token、KV 预算。示例无缓存命中，有足够空间接纳当前批次。");
 service("Prefill","路由与组批","组批并提交模型执行","选中的请求","TP worker / ModelRunner","input_ids、位置与 KV 映射","本轮无切块。模型执行器准备设备张量和注意力元数据，下一步才进入 Embedding。");
 for(let i=0;i<model.stages.length;i++){
   if(model.phase(i)===13) service("Prefill","KV 交接","匹配两端请求并准备传输","Prefill / Decode worker","bootstrap 信息与请求标识","源缓存、目标槽与传输上下文","这是逻辑依赖视图；bootstrap 和目的端准备可以与 Prefill 重叠。缓存内容经 NIXL 搬运。");
   events.push({i,path:model.path(i),stage:model.stages[i]});
   if(model.phase(i)===13) service("Decode","流式输出","首个生成结果进入输出流","Prefill 产生的 token ID","增量解码与输出缓冲","可发送的 SSE delta","首个生成 ID 来自 Prefill。图将其放在交接边界说明，网络实际可见时间还取决于缓冲与调度。");
 }
 service("Decode","返回与清理","检查停止条件","已生成的 ID 序列","EOS / stop / 长度检查","本例达到 max_tokens=2","若继续生成，新 ID 成为下一轮 Decode 的输入，沿用各层 KV；本例在第二个输出 ID 后结束。");
 service("Decode","返回与清理","增量解码并流式回传","采样 ID 与输出缓冲","Detokenize / SSE","客户端收到文本 delta","把可解码的文本分片送到流式接口，一个网络分片可以对应多个 token。");
 service("Decode","返回与清理","用户收到结束事件，回收请求状态","最后的文本与结束原因","finish_reason=length · [DONE]","客户端结束读取","服务端移出运行队列并释放请求持有的缓存引用；可复用前缀按缓存策略保留。清理和网络收尾可能重叠。");
 const at=i=>events[i], gpu=i=>at(i).i!==undefined;
 const adapted={...model,stages:events.map(e=>e.stage),path:i=>at(i)?.path,
   tocPath(i){
     const [phase,section,ffn]=at(i).path;
     if(section==="请求接入" || section==="路由与组批")return ["请求准备",section];
     if(section==="KV 交接")return ["PD 交接","KV 缓存传输"];
     if(section==="流式输出")return ["PD 交接","首 token 回传"];
     if(section==="返回与清理")return ["请求结束",section];
     return [phase,ffn?"FFN · "+ffn:section];
   },
   service:i=>gpu(i)?null:at(i),phase:i=>gpu(i)?model.phase(at(i).i):at(i).path[0]==="Prefill"?-1:17,
   op:(i,l)=>gpu(i)?model.op(at(i).i,l):"",
   layerAllows:(i,g)=>!gpu(i)||model.layerAllows(at(i).i,g),
   layerPresentation:(i,g)=>gpu(i)?model.layerPresentation(at(i).i,g):null,
   describe:(i,r,side,n,l)=>gpu(i)?model.describe(at(i).i,r,side,n,l):[]
 };
 window.inferenceModels[model.id]=adapted;
}

// Equations use [input, output] weight layout, matching the matrix walkthrough.
{
 const scene=window.inferenceModels["glm-5.3"];
 scene.equation=(i,side,n,layout,group)=>{
   if(scene.service(i))return null;
   const title=scene.stages[i][1], p=scene.phase(i), d=side==="D";
   const t=d?1:layout==="cp"?2*n:16*n, heads=d||layout==="cp"?64:8;
   const h="["+t+",6144]", latent="["+t+",2048]";
   if(title==="首个输出 token" || title==="末层归一化与词表投影")return [
     String.raw`z=\operatorname{RMSNorm}(x_{\rm last})\,W_{\rm head}`,
     d?"当前行 [1,6144] × 逻辑权重 [6144,154880] → 全词表分数 [1,154880]。":"取提示的最后一行，6144 个通道投影为 154880 个词表分数，再采样首个 ID。"
   ];
   if(title==="采样与流式返回")return tokenSamplingEquation;
   if(title==="读取新 token 与历史 KV")return [String.raw`x_{\rm new}=E[t_{\rm prev},:]`,"用上一轮生成的 ID 查出 6144 维输入，历史 KV 继续留在缓存中。"];
   if(title==="词表分片查表")return [String.raw`\mathbf{x}^{(r)}_j=\begin{cases}E^{(r)}[t_j-a_r,:],& a_r\le t_j<a_{r+1},\\ \mathbf{0},&\text{otherwise}.\end{cases}`,"tⱼ 为 token ID；aᵣ 为本卡词表起始行，命中时取整行 6144 个通道。"];
   if(p===1)return [String.raw`X=\sum_{r=0}^{7}X^{(r)}`,"每个 token 只有一张卡命中；合并后每卡都有 X ["+16*n+",6144]。"];
   if(title==="残差与输入归一化")return [String.raw`\widehat{x}_{j,k}=\frac{\gamma_k\,x_{j,k}}{\sqrt{\dfrac{1}{H}\sum_{\ell=1}^{H}x_{j,\ell}^{\,2}+\varepsilon}},\qquad H=6144`,h+" → "+h+"　每个 token 单独归一化，保留 X 供残差相加。"];
   if(title==="Q / KV 低秩投影" || title==="输入归一化与低秩投影")return [String.raw`\bigl[\,Q_a\mid C_a\mid K_{\mathrm{pos}}\,\bigr]=\widehat X W_a`,h+" × [6144,2624] → ["+t+",2624]；按列拆成 2048 | 512 | 64。qₐ 即 Q latent。"];
   if(title.startsWith("展开 Q") || title==="展开查询与追加缓存")return [String.raw`Q^{(r)}=\operatorname{reshape}\!\left(\operatorname{RMSNorm}(Q_a)\,W_{q,b}^{(r)}\right)`,latent+" × [2048,"+heads*256+"] → ["+t+","+heads+",256]；每头 192 个内容通道、64 个位置通道。"];
   if(title.includes("拆出 Q"))return [String.raw`\mathbf q_j=[\,\mathbf q_j^{\mathrm c}\mid\mathbf q_j^{\mathrm p}\,],\qquad \mathbf q_j^{\mathrm R}=\operatorname{RoPE}(\mathbf q_j^{\mathrm p},j)`,"仅旋转每头的 64 维位置分量；192 维内容分量另行投影。"];
   if(title.includes("压缩缓存"))return [String.raw`\mathcal C[j]\gets\bigl[\,\operatorname{RMSNorm}(\mathbf c_{a,j})\mid\operatorname{RoPE}(\mathbf k_{\mathrm{pos},j},j)\,\bigr]`,"每个位置 512 个压缩内容通道 + 64 个位置通道，DSA 的索引缓存另存。"];
   if(title.includes("稀疏地址"))return group.index==="shared"?[String.raw`\mathcal I_j^{(\ell)}=\mathcal I_j^{(\ell-1)}`,"复用候选位置地址，本层仍重新计算注意力。"]:[String.raw`\mathcal I_j=\underset{0\le i\le j}{\operatorname{TopKIndices}}\,s_{j,i},\qquad K=\min(2048,j+1)`,"最多 2048 个位置，受当前请求的因果可见长度限制。"];
   if(title==="吸收 K 投影，计算候选分数")return [String.raw`S_h=(q_h^{\rm c}W_{\rm KC,h})\,C_{\mathcal I}^{\mathsf T}+q_h^{\rm R}(K_{\mathcal I}^{\rm R})^{\mathsf T}`,"每个候选位置的分数等于内容点积加位置点积。"];
   if(title.includes("吸收 K"))return [String.raw`\widetilde{\mathbf q}_h=\mathbf q_h^{\mathrm c}W_{\mathrm{KC},h}`,"每头 [1,192] × [192,512] → [1,512]，直接与缓存 latent 点积。"];
   if(title.includes("稀疏注意力分数"))return [String.raw`S_h=\widetilde{\mathbf q}_h C_{\mathcal I}^{\mathsf T}+\mathbf q_h^{\mathrm R}\bigl(K_{\mathcal I}^{\mathrm R}\bigr)^{\mathsf T}`,"每个候选位置贡献一个分数；两项分别沿 512 和 64 个通道做点积。"];
   if(title.includes("归一化并聚合"))return [String.raw`Z_h=\operatorname{softmax}_{\mathcal I}(\alpha S_h+m)\,C_{\mathcal I}`,"沿候选位置归一化，再加权求和，每头得到一个 512 维向量。"];
   if(title.includes("Softmax") || title.includes("因果掩码"))return [String.raw`A_{h,i}=\frac{\exp(\alpha S_{h,i}+m_i)}{\displaystyle\sum_{u\in\mathcal I}\exp(\alpha S_{h,u}+m_u)},\qquad i\in\mathcal I`,"每个查询头在候选位置上的权重之和为 1。"];
   if(title.includes("聚合候选 latent"))return [String.raw`Z=A\,C_{\mathcal I}`,"每头 [1,L] × [L,512] → [1,512]，L 为所选位置数。"];
   if(title.includes("恢复各头"))return [String.raw`\mathbf o_h=\mathbf z_h W_{v,b,h}`,"每头 [1,512] × [512,256] → [1,256]。"];
   if(title.includes("恢复 value"))return [String.raw`x_{\rm out}=x_{\rm res}+\operatorname{concat}_{h}(Z_h W_{v,b,h})\,W_o`,"64 个头各恢复 256 维，拼接投影到 6144 维后加回残差。"];
   if(title.includes("输出投影 W"))return [String.raw`Y^{(r)}=\operatorname{Concat}_{h\in\mathcal H_r}(O_h)\,W_o^{(r)}`,"["+t+","+heads*256+"] × ["+heads*256+",6144] → "+h+"。"];
   if(p===7)return [layout==="tp"?String.raw`X_{\mathrm{out}}=X_{\mathrm{res}}+\sum_{r=0}^{7}Y^{(r)}`:String.raw`X_{\mathrm{out}}^{(r)}=X_{\mathrm{res}}^{(r)}+Y^{(r)}`,"TP 的每份 Yᵣ 只包含本卡头的贡献。CP 配置在本卡已得到该行的完整输出。"];
   if(title==="经过前三个 Dense 层" || title==="前三层的 Dense 前馈")return [String.raw`Y^{(r)}=\bigl[\operatorname{SiLU}(HW_g^{(r)})\odot(HW_u^{(r)})\bigr]W_d^{(r)}`,"H 是归一化输入。每卡计算 1536 个中间通道，再归并八份 Down 部分积。"];
   if(title.includes("Gate 与 Up") || title.includes("专家中间分片"))return [String.raw`G=H W_{\mathrm{gate}},\qquad U=H W_{\mathrm{up}}`,"同一份专家输入分别乘两组权重，生成门控分量与数值分量。"];
   if(title.includes("门控激活"))return [String.raw`M=\operatorname{SiLU}(G)\odot U`,"形状不变，逐元素门控。"];
   if(title.includes("Down 投影") || title.includes("专家输出投影"))return [String.raw`Y_e=M_e W_{\mathrm{down},e}`,"从专家中间维投回 6144 维。按中间维切分时，本卡仅贡献部分积。"];
   if(title==="归并 TP 部分积")return [String.raw`Y=\sum_{r=0}^{7}Y^{(r)}`,"每卡的 [1,6144] 是专家 TP 部分积。ReduceScatter 后每卡持有已求和的 [1,768]，AllGather 后每卡都有完整的 [1,6144]。"];
   if(title==="合并专家输出")return [String.raw`R_j=\sum_{r=0}^7\sum_{e\in E_r\cap I_j}w_{j,e} f_e(h_j)`,layout==="cp"?"ReduceScatter 沿 token 行求和，Pᵣ 留下位置 r 和 r+8。":"AllReduce 求和，每卡都得到 16 行路由专家输出。"];
   if(title.includes("合入共享"))return [String.raw`\mathbf y_j=\mathbf x_{\mathrm{res},j}+f_{\mathrm{shared}}(\mathbf h_j)+\sum_{e\in\mathcal E_j} w_{j,e}\, f_e(\mathbf h_j)`,"对选中的 8 个路由专家加权，wₑ 包含路由缩放。"];
   return null;
 };
 scene.weightView=(i,side,r,layout,active)=>{
   const p=scene.phase(i), d=side==="D",title=scene.stages[i][1],heads=d||layout==="cp"?64:8;
   if(active){
    if(p===0 && !d)return ["Embedding Eᵣ","19,360 × 6,144"];
    if(title==="残差与输入归一化")return ["RMSNorm γ","尺度 [6144]"];
    if(title.includes("低秩投影"))return ["Wₐ · 卡间复制","6144 × 2624"];
    if(title.includes("展开 Q")||title.includes("展开查询"))return ["Wqᵦ · "+heads+" 头","2048 × "+heads*256];
    if(title.includes("吸收 K"))return ["每头 Wkc","192 × 512"];
    if(title.includes("恢复各头"))return ["每头 Wᵥᵦ","512 × 256"];
    if(title.includes("输出投影 W")||title.includes("恢复 value"))return ["Wₒ · "+heads+" 头",heads*256+" × 6144"];
   }
   return d?["专家 Wg/Wu/Wd","中间维 "+r*256+"…"+((r+1)*256-1)]:["专家 E"+32*r+"–E"+(32*r+31),"各有 Wg / Wu / Wd"];
 };
}
