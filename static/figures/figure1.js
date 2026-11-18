/* EchoMoE Figure 1. Timings communicate order, not benchmark measurements. */
(() => {
 'use strict';
 const root = document.querySelector('.fig1');
 if (!root) return;
 const svg = root.querySelector('.f1-diagram');
 const ns = 'http://www.w3.org/2000/svg';
 const duration = 10200;
 const timelines = {
  standard: [
   {start:0,end:900,key:'attention',label:'Attention computes the activations'},
   {start:1050,end:3750,key:'dispatch',label:'Dispatching full activations'},
   {start:3900,end:4700,key:'experts',label:'Experts compute'},
   {start:4850,end:7550,key:'combine',label:'Combining full expert outputs'},
   {start:7700,end:8350,key:'output',label:'Output is ready'},
   {start:8350,end:10200,key:'done',label:'Transfer complete'}
  ],
  echo: [
   {start:0,end:900,key:'attention',label:'Attention computes the activations'},
   {start:1050,end:1500,key:'compress-dispatch',label:'Compressing the change'},
   {start:1590,end:2590,key:'dispatch',label:'Dispatching a compact change'},
   {start:2680,end:3130,key:'decompress-dispatch',label:'Reconstructing the expert input'},
   {start:3220,end:3960,key:'experts',label:'Experts compute'},
   {start:4050,end:4500,key:'compress-combine',label:'Compressing the expert outputs'},
   {start:4590,end:5590,key:'combine',label:'Combining compact outputs'},
   {start:5680,end:6130,key:'decompress-combine',label:'Decompressing the combined output'},
   {start:6220,end:6770,key:'output',label:'Output is ready'},
   {start:6770,end:10200,key:'done',label:'Transfer complete'}
  ]
 };
 // The compact payload completes each collective in 1.0s, versus 2.7s for
 // the full payload. Its blocks also traverse their paths visibly faster.
 const transferTimings={};
 for(const [lane,stages] of Object.entries(timelines)){
  for(const stage of stages){
   if(stage.key==='dispatch'||stage.key==='combine'){
    transferTimings[`${lane}-${stage.key}`]=Object.freeze({start:stage.start,end:stage.end,duration:stage.end-stage.start});
   }
  }
 }
 // ---- Portrait layout for narrow screens: the two lanes side by side, flow top to bottom.
 // Only GPU 0 (node 0) and GPU 2 (node 1) are drawn: token A's local and remote expert.
 const P={W:340,H:486,laneW:151,lanes:{standard:30,echo:189},col:[38,113],
  bands:[[60,142,'#f0f4fa','Attention','DP','#365c91'],[146,212,'#f7f8fa','Dispatch','all-to-all','#7d838b'],
   [216,330,'#f2f7ed','Experts','EP','#557936'],[334,400,'#f7f8fa','Combine','all-to-all','#7d838b'],
   [404,482,'#f0f4fa','Output','DP','#365c91']]};
 const mk=(tag,attrs,parent,text)=>{const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text!=null)n.textContent=text;if(parent)parent.appendChild(n);return n;};
 function portraitRoutes(lane){
  const x0=P.lanes[lane],c0=x0+P.col[0],c2=x0+P.col[1];
  return lane==='standard'
   ?{'standard-dispatch':[[c0-4,104,c0-4,255],[c0+6,104,c2,255]],'standard-combine':[[c0-4,294,c0-4,443],[c2,294,c0+6,443]]}
   :{'echo-dispatch':[[c0-4,140,c0-4,219],[c0+6,140,c2,219]],'echo-combine':[[c0-4,330,c0-4,407],[c2,330,c0+6,407]]};
 }
 function buildPortrait(){
  const m=mk('svg',{class:'f1-diagram f1-portrait',viewBox:`0 0 ${P.W} ${P.H}`,role:'img',
   'aria-label':'Standard expert parallelism and EchoMoE side by side. Token A on GPU 0 is sent to local expert E zero and remote expert E two on GPU 2, and their outputs are combined back on GPU 0. Standard EP moves nine-block payloads; EchoMoE compresses before dispatch and combine and sends two-block payloads, using a cached reference z for dispatch. Red arrows cross nodes; dark arrows stay within a node.'});
  const defs=mk('defs',{},m);
  for(const [id,c] of [['dark','#424950'],['red','#d43b35']]){
   const mark=mk('marker',{id:`f1m-arrow-${id}`,markerWidth:7,markerHeight:6,refX:6,refY:3,orient:'auto',markerUnits:'userSpaceOnUse'},defs);
   mk('path',{d:'M0 0 7 3 0 6Z',fill:c},mark);
  }
  // Stage names run down the left gutter and apply to both lanes.
  for(const [y0,y1,,name,sub,color] of P.bands){
   const cy=(y0+y1)/2;
   mk('text',{class:'f1-m-stage',transform:`translate(13 ${cy}) rotate(-90)`,'text-anchor':'middle'},m,name);
   mk('text',{class:'f1-m-stage-sub',transform:`translate(26.5 ${cy}) rotate(-90)`,'text-anchor':'middle',style:`fill:${color}`},m,sub);
  }
  for(const lane of ['standard','echo']){
   const x0=P.lanes[lane],c0=x0+P.col[0],c2=x0+P.col[1],echo=lane==='echo';
   const g=mk('g',{class:`f1-lane f1-${lane}`},m);
   mk('rect',{x:x0,y:0,width:P.laneW,height:24,rx:8,fill:echo?'#f1eaf7':'#eff1f4'},g);
   mk('text',{class:'f1-m-lane',x:x0+P.laneW/2,y:16.5,'text-anchor':'middle',style:`fill:${echo?'#75469a':'#37414d'}`},g,echo?'EchoMoE':'Standard EP');
   for(const [cx,gpu,node] of [[c0,'GPU 0','Node 0'],[c2,'GPU 2','Node 1']]){
    mk('text',{class:'f1-m-gpu',x:cx,y:40,'text-anchor':'middle'},g,gpu);
    mk('text',{class:'f1-m-node',x:cx,y:53,'text-anchor':'middle'},g,node);
   }
   for(const [y0,y1,fill] of P.bands)mk('rect',{x:x0,y:y0,width:P.laneW,height:y1-y0,rx:9,fill},g);
   mk('path',{d:`M${x0+75.5} 30V${P.H-2}`,class:'f1-m-divider'},g);
   const R=portraitRoutes(lane);
   for(const [x1,y1,x2,y2] of [...R[`${lane}-dispatch`],...R[`${lane}-combine`]]){
    const red=x1!==x2;
    mk('path',{d:`M${x1} ${y1}L${x2} ${y2}`,class:'f1-m-halo'},g);
    mk('path',{class:'f1-route',d:`M${x1} ${y1}L${x2} ${y2}`,stroke:red?'#d43b35':'#424950','marker-end':`url(#f1m-arrow-${red?'red':'dark'})`},g);
   }
   const box=(cls,part,cx,y,w,h,muted)=>{
    const n=mk('g',{class:`f1-node ${cls}${muted?' f1-muted':''}`},g);
    if(part)n.setAttribute('data-f1-part',part);
    mk('rect',{class:'f1-surface',x:cx-w/2,y,width:w,height:h,rx:7},n);
    return n;
   };
   if(echo)for(const cx of [c0,c2])mk('path',{d:`M${cx} 100V108M${cx} 250.6V258M${cx} 290V298M${cx} 438.6V446`,class:'f1-m-stub'},g);
   for(const [cx,tok,muted] of [[c0,'A',false],[c2,'C',true]]){
    const a=box('f1-attn',muted?null:`${lane}-attention`,cx,68,60,32,muted);
    mk('rect',{x:cx-26,y:73,width:22,height:22,rx:5,fill:muted?'#7d98bf':'#365c91'},a);
    mk('text',{class:'f1-m-token',x:cx-15,y:88.5,'text-anchor':'middle'},a,tok);
    mk('text',{class:'f1-m-attn',x:cx+14,y:88,'text-anchor':'middle'},a,'Attn');
    const o=box('f1-output',muted?null:`${lane}-output`,cx,446,48,30,muted);
    mk('text',{class:'f1-m-output',x:cx,y:466,'text-anchor':'middle'},o,tok+'′');
   }
   for(const [cx,i] of [[c0,0],[c2,2]]){
    const e=box('f1-expert',`${lane}-experts`,cx,258,52,32,false);
    const t=mk('text',{class:'f1-m-expert',x:cx-2,y:279,'text-anchor':'middle'},e,'E');
    mk('tspan',{'font-size':12,dy:4},t,String(i));
   }
   if(echo){
    // [codec, column x, top y, muted, uses the cached reference]
    for(const [key,cx,y,muted,ref] of [['compress-dispatch',c0,108,false,true],['compress-dispatch',c2,108,true,true],
      ['decompress-dispatch',c0,222,false,true],['decompress-dispatch',c2,222,false,true],
      ['compress-combine',c0,298,false,false],['compress-combine',c2,298,false,false],
      ['decompress-combine',c0,410,false,false],['decompress-combine',c2,410,true,false]]){
     const compress=key.startsWith('compress');
     // Same trapezoid as the wide layout, turned to point down the flow.
     const n=mk('g',{class:`f1-node f1-codec${muted?' f1-muted':''}`,transform:`translate(${cx} ${y}) rotate(90) scale(.65)`},g);
     if(!muted)n.setAttribute('data-f1-part',`echo-${key}`);
     mk('title',{},n,compress?'Compress':'Decompress');
     mk('polygon',{class:'f1-surface','vector-effect':'non-scaling-stroke',points:compress?'0,-25 44,-11 44,11 0,25':'0,-11 44,-25 44,25 0,11'},n);
     if(ref){
      const my=y+14.3,r=mk('g',{class:muted?'f1-muted':''},g);
      mk('path',{class:'f1-reference',d:`M${cx+12} ${my}H${cx+20}`},r);
      mk('rect',{class:'f1-cache',x:cx+20,y:my-7,width:14,height:14,rx:3},r);
      mk('text',{class:'f1-m-cache',x:cx+27,y:my+4,'text-anchor':'middle'},r,'z');
     }
    }
   }
   mk('g',{'data-f1-packets':`${lane}-dispatch`,'aria-hidden':'true'},g);
   mk('g',{'data-f1-packets':`${lane}-combine`,'aria-hidden':'true'},g);
  }
  return m;
 }
 const portrait=buildPortrait();
 svg.after(portrait);

 // ---- One timeline drives both layouts.
 const views=[];
 function addView(view,routes,codecLocations,block){
  const packets={},codecs=[];
  for (const [key,segments] of Object.entries(routes)) {
   const layer = view.querySelector(`[data-f1-packets="${key}"]`);
   layer.setAttribute('data-transfer-start-ms',String(transferTimings[key].start));
   layer.setAttribute('data-transfer-end-ms',String(transferTimings[key].end));
   layer.setAttribute('data-transfer-duration-ms',String(transferTimings[key].duration));
   layer.innerHTML='';
   packets[key] = segments.map((line,routeIndex) => {
    const count=key.startsWith('standard') ? 9 : 2;
    const blocks=[];
    for(let i=0;i<count;i++) {
     const rect=document.createElementNS(ns,'rect');
     rect.setAttribute('x',String(-block.w/2));rect.setAttribute('y',String(-block.h/2));
     rect.setAttribute('width',String(block.w));rect.setAttribute('height',String(block.h));
     rect.setAttribute('rx',String(block.rx));rect.setAttribute('class','f1-payload');
     rect.setAttribute('data-route-index',String(routeIndex));
     rect.setAttribute('data-block-index',String(i));
     rect.setAttribute('visibility','hidden');layer.appendChild(rect);blocks.push(rect);
    }
    const [x1,y1,x2,y2]=line;
    return {line,blocks,length:Math.hypot(x2-x1,y2-y1),angle:Math.atan2(y2-y1,x2-x1)*180/Math.PI};
   });
  }
  // Inside each codec, the full payload contracts to two blocks (or expands back).
  for(const [key,locations] of Object.entries(codecLocations)) {
   const parents=view.querySelectorAll(`[data-f1-part="echo-${key}"]`);
   locations.forEach(([x,y],index)=>{
    const group=document.createElementNS(ns,'g');
    group.setAttribute('aria-hidden','true');
    const bits=[];
    for(let i=0;i<9;i++){
     const bit=document.createElementNS(ns,'rect');
     bit.setAttribute('width','3.8');bit.setAttribute('height','5.8');
     bit.setAttribute('rx','1');bit.setAttribute('fill','#5f6b76');
     group.appendChild(bit);bits.push(bit);
    }
    parents[index].appendChild(group);
    codecs.push({key,x,y,bits,group,decompress:key.startsWith('decompress')});
   });
  }
  views.push({packets,codecs,block});
 }
 addView(svg,{
  'standard-dispatch':[[339,125,604,125],[339,135,604,273]],
  'standard-combine':[[777,125,1062,125],[777,273,1062,135]],
  'echo-dispatch':[[394,561,588,561],[394,571,588,709]],
  'echo-combine':[[820,561,1000,561],[820,709,1000,571]]
 },{
  'compress-dispatch':[[350,561]],
  'decompress-dispatch':[[588,561],[588,719]],
  'compress-combine':[[776,561],[776,719]],
  'decompress-combine':[[1000,561]]
 },{w:11,h:18,rx:2.8,gap:14,lead:9});
 // Portrait codecs live in their own rotated frame, so their bits sit at the local origin.
 addView(portrait,{...portraitRoutes('standard'),...portraitRoutes('echo')},
  {'compress-dispatch':[[0,0]],'decompress-dispatch':[[0,0],[0,0]],'compress-combine':[[0,0],[0,0]],'decompress-combine':[[0,0]]},
  {w:6,h:11,rx:1.6,gap:8.5,lead:6});
 const nodes=Array.from(root.querySelectorAll('[data-f1-part]'));
 const status={standard:root.querySelector('[data-f1-status="standard"]'),echo:root.querySelector('[data-f1-status="echo"]')};
 const progress=root.querySelector('.f1-progress-fill');
 const scrubber=root.querySelector('.f1-scrubber');
 const toggle=root.querySelector('[data-f1-action="toggle"]');
 const toggleText=toggle.querySelector('span');
 const toggleIcon=toggle.querySelector('svg');
 const replay=root.querySelector('[data-f1-action="replay"]');
 const speedButtons=Array.from(root.querySelectorAll('[data-f1-speed] [data-speed]'));
 const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
 let elapsed=reduced.matches?2200:0, speed=1, playing=!reduced.matches, inView=false, frame=0, previous=null;
 let lastStages='';
 let announcedProgress=-1;
 const findStage=(lane,t) => timelines[lane].find(s=>t>=s.start&&t<s.end);
 const stageLabel=(lane,t) => {
  const active=findStage(lane,t);
  if(active)return active.label;
  const prev=timelines[lane].filter(s=>s.end<=t).pop();
  return prev ? prev.label : 'Ready';
 };
 function paintPackets(route,p,block) {
  const tail=(route.blocks.length-1)*block.gap;
  const lead=p*(route.length+tail+2*block.lead)-block.lead;
  const [x1,y1,x2,y2]=route.line;
  route.blocks.forEach((el,i)=>{
   const d=lead-i*block.gap;
   if(p<0||d<0||d>route.length){el.setAttribute('visibility','hidden');return;}
   const q=d/route.length;
   el.setAttribute('transform',`translate(${(x1+(x2-x1)*q).toFixed(2)} ${(y1+(y2-y1)*q).toFixed(2)}) rotate(${route.angle.toFixed(2)})`);
   el.setAttribute('visibility','visible');
  });
 }
 function render(t) {
  const a=findStage('standard',t),b=findStage('echo',t);
  const keys=[a?`standard-${a.key}`:'',b?`echo-${b.key}`:''];
  const stageId=keys.join('|');
  if(stageId!==lastStages){
   nodes.forEach(n=>n.classList.toggle('is-active',keys.includes(n.dataset.f1Part)));
   status.standard.textContent=stageLabel('standard',t);
   status.echo.textContent=stageLabel('echo',t);
   root.dataset.standardStage=a?a.key:'transition';root.dataset.echoStage=b?b.key:'transition';
   lastStages=stageId;
  }
  for(const view of views){
   for(const [key,routes] of Object.entries(view.packets)) {
    const active=key.startsWith('standard')?a:b;
    const type=key.split('-')[1];
    const p=active&&active.key===type?(t-active.start)/(active.end-active.start):-1;
    for(const route of routes)paintPackets(route,p,view.block);
   }
   for(const codec of view.codecs){
    if(!b||b.key!==codec.key){codec.group.setAttribute('visibility','hidden');continue;}
    codec.group.setAttribute('visibility','visible');
    const p=(t-b.start)/(b.end-b.start);
    const q=p*p*(3-2*p);
    codec.bits.forEach((bit,i)=>{
     const row=Math.floor(i/3),col=i%3;
     let x0,y0,x1,y1,opacity;
     if(codec.decompress){
      x0=codec.x+7;y0=codec.y+(i%2?1.5:-7.5);
      x1=codec.x+25+col*5;y1=codec.y-10+row*8;
      opacity=i<2?1:Math.max(0,(q-.2)/.8);
     }else{
      x0=codec.x+7+col*5;y0=codec.y-10+row*8;
      x1=codec.x+34;y1=codec.y+(i%2?1.5:-7.5);
      opacity=i<2?1:Math.max(0,1-q*1.5);
     }
     bit.setAttribute('x',(x0+(x1-x0)*q).toFixed(2));
     bit.setAttribute('y',(y0+(y1-y0)*q).toFixed(2));
     bit.setAttribute('opacity',opacity.toFixed(3));
    });
   }
  }
  progress.style.width=`${t/duration*100}%`;
  scrubber.value=String(t);
  const tenths=Math.floor(t/100);
  if(tenths!==announcedProgress){scrubber.setAttribute('aria-valuetext',`${(t/1000).toFixed(1)} seconds of ${(duration/1000).toFixed(1)} seconds`);announcedProgress=tenths;}
 }
 function updateControl(){
  toggleText.textContent=playing?'Pause':'Play';
  toggle.setAttribute('aria-label',playing?'Pause Figure 1 animation':'Play Figure 1 animation');
  toggle.setAttribute('aria-pressed',String(playing));
  toggleIcon.innerHTML=playing?'<rect x="2" y="1" width="3" height="10" rx=".8"></rect><rect x="7" y="1" width="3" height="10" rx=".8"></rect>':'<path d="M3 1.4 11 6 3 10.6Z"></path>';
 }
 function tick(now){
  frame=0;
  if(!playing||!inView||document.hidden){previous=null;return;}
  if(previous!==null)elapsed=(elapsed+Math.min(now-previous,100)*speed)%duration;
  previous=now;render(elapsed);frame=requestAnimationFrame(tick);
 }
 function sync(){
  updateControl();
  if(playing&&inView&&!document.hidden&&!frame){previous=null;frame=requestAnimationFrame(tick);}
  if((!playing||!inView||document.hidden)&&frame){cancelAnimationFrame(frame);frame=0;previous=null;}
 }
 function play(){playing=true;sync();}
 function pause(){playing=false;sync();}
 toggle.addEventListener('click',()=>playing?pause():play());
 replay.addEventListener('click',()=>{elapsed=0;lastStages='';render(elapsed);play();});
 speedButtons.forEach(btn=>btn.addEventListener('click',()=>{
  speed=Number(btn.dataset.speed)||1;
  speedButtons.forEach(b=>b.setAttribute('aria-pressed',String(b===btn)));
 }));
 scrubber.addEventListener('input',()=>{pause();elapsed=Math.max(0,Math.min(duration-1,Number(scrubber.value)||0));lastStages='';render(elapsed);});
 if('IntersectionObserver'in window){
  const observer=new IntersectionObserver(entries=>{inView=entries[0].isIntersecting;sync();},{threshold:.08});
  observer.observe(root);
 }else{inView=true;}
 document.addEventListener('visibilitychange',()=>{
  if(document.hidden){if(frame)cancelAnimationFrame(frame);frame=0;previous=null;}
  else sync();
 });
 const onMotionChange=()=>{if(reduced.matches)pause();};
 if(reduced.addEventListener)reduced.addEventListener('change',onMotionChange);
 root.classList.add('has-animation');render(elapsed);sync();
 window.EchoFigure1={
  seek(ms){pause();elapsed=Math.max(0,Math.min(duration-1,Number(ms)||0));lastStages='';render(elapsed);return this.getState();},
  pause,play,
  getState(){return{elapsed:Math.round(elapsed),duration,speed,playing,inView,standard:root.dataset.standardStage,echo:root.dataset.echoStage,transfers:Object.fromEntries(Object.entries(transferTimings).map(([key,value])=>[key,{...value}]))};}
 };
})();
