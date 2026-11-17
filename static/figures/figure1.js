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
 const routes = {
  'standard-dispatch':[[339,125,604,125],[339,135,604,273]],
  'standard-combine':[[777,125,1062,125],[777,273,1062,135]],
  'echo-dispatch':[[394,561,588,561],[394,571,588,709]],
  'echo-combine':[[820,561,1000,561],[820,709,1000,571]]
 };
 const packets = {};
 for (const [key,segments] of Object.entries(routes)) {
  const layer = svg.querySelector(`[data-f1-packets="${key}"]`);
  layer.setAttribute('data-transfer-start-ms',String(transferTimings[key].start));
  layer.setAttribute('data-transfer-end-ms',String(transferTimings[key].end));
  layer.setAttribute('data-transfer-duration-ms',String(transferTimings[key].duration));
  layer.innerHTML='';
  packets[key] = segments.map((line,routeIndex) => {
   const count=key.startsWith('standard') ? 9 : 2;
   const blocks=[];
   for(let i=0;i<count;i++) {
    const rect=document.createElementNS(ns,'rect');
    rect.setAttribute('x','-5.5');rect.setAttribute('y','-9');
    rect.setAttribute('width','11');rect.setAttribute('height','18');
    rect.setAttribute('rx','2.8');rect.setAttribute('class','f1-payload');
    rect.setAttribute('data-route-index',String(routeIndex));
    rect.setAttribute('data-block-index',String(i));
    rect.setAttribute('visibility','hidden');layer.appendChild(rect);blocks.push(rect);
   }
   const [x1,y1,x2,y2]=line;
   return {line,blocks,length:Math.hypot(x2-x1,y2-y1),angle:Math.atan2(y2-y1,x2-x1)*180/Math.PI};
  });
 }
 // Inside each codec, the full payload contracts to two blocks (or expands back).
 const codecs=[];
 const codecLocations={
  'compress-dispatch':[[350,561]],
  'decompress-dispatch':[[588,561],[588,719]],
  'compress-combine':[[776,561],[776,719]],
  'decompress-combine':[[1000,561]]
 };
 for(const [key,locations] of Object.entries(codecLocations)) {
  const parents=svg.querySelectorAll(`[data-f1-part="echo-${key}"]`);
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
 const nodes=Array.from(svg.querySelectorAll('[data-f1-part]'));
 const status={standard:root.querySelector('[data-f1-status="standard"]'),echo:root.querySelector('[data-f1-status="echo"]')};
 const progress=root.querySelector('.f1-progress-fill');
 const scrubber=root.querySelector('.f1-scrubber');
 const toggle=root.querySelector('[data-f1-action="toggle"]');
 const toggleText=toggle.querySelector('span');
 const toggleIcon=toggle.querySelector('svg');
 const replay=root.querySelector('[data-f1-action="replay"]');
 const speedControl=root.querySelector('[data-f1-speed]');
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
 function paintPackets(key,p) {
  for(const route of packets[key]) {
   const count=route.blocks.length;
   const tail=(count-1)*14;
   const lead=p*(route.length+tail+18)-9;
   route.blocks.forEach((block,i)=>{
    const d=lead-i*14;
    if(p<0||d<0||d>route.length){block.setAttribute('visibility','hidden');return;}
    const q=d/route.length;
    const [x1,y1,x2,y2]=route.line;
    block.setAttribute('transform',`translate(${(x1+(x2-x1)*q).toFixed(2)} ${(y1+(y2-y1)*q).toFixed(2)}) rotate(${route.angle.toFixed(2)})`);
    block.setAttribute('visibility','visible');
   });
  }
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
  for(const key of Object.keys(packets)) {
   const lane=key.startsWith('standard')?'standard':'echo';
   const active=lane==='standard'?a:b;
   const type=key.split('-')[1];
   const p=active&&active.key===type?(t-active.start)/(active.end-active.start):-1;
   paintPackets(key,p);
  }
  for(const codec of codecs){
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
 speedControl.addEventListener('change',()=>{speed=Number(speedControl.value)||1;});
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
