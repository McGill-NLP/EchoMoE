/* Figure 2: deterministic, independently controllable execution timeline. */
(function () {
  'use strict';
  function initFigure2() {
    const root = document.querySelector('.fig2');
    if (!root || root.dataset.initialized) return;
    root.dataset.initialized = 'true';
    const NS = 'http://www.w3.org/2000/svg';
    const duration = 20300;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const controls = {
      play: root.querySelector('.f2-play'),
      replay: root.querySelector('.f2-replay'),
      speed: Array.from(root.querySelectorAll('[data-f2-speed] [data-speed]')),
      scrubber: root.querySelector('.f2-scrubber'),
      caption: root.querySelector('.f2-status'),
      tag: root.querySelector('.f2-phase-tag')
    };
    const elements = new Map(Array.from(root.querySelectorAll('[id]')).map(el => [el.id, el]));
    const nodes = Array.from(root.querySelectorAll('.f2-compute'));
    const packetLayers = {
      l: root.querySelector('[data-panel="left"]'),
      r: root.querySelector('[data-panel="right"]'),
      m: root.querySelector('[data-panel="mobile"]')
    };
    // Narrow screens show one panel at a time and a single-column version of
    // the layer detail ("f2-m-" ids mirror the desktop "f2-r-" ids).
    const narrow = window.matchMedia('(max-width: 799px)');
    const viewButtons = Array.from(root.querySelectorAll('[data-f2-view]'));
    const detailWindow = [8250, 17550];
    let view = '';
    const colors = { ink:'#4a5356', purple:'#7b4f9b', teal:'#27879d', green:'#6a8c4c', orange:'#ae641a' };
    let time = reducedMotion ? 12100 : 0;
    let paused = reducedMotion;
    let visible = true;
    let speed = 1;
    let raf = 0;
    let lastStamp = null;
    let lastPhase = -1;

    // Full reference transfer: 1700 ms. Compact dispatch/combine: 1000 ms.
    // The timings illustrate the sequence and payload-size difference.
    const transfers = {
      reference: {start:3150, end:4850, duration:1700, payload:'full'},
      dispatch: {start:11100, end:12100, duration:1000, payload:'compact'},
      combine: {start:15300, end:16300, duration:1000, payload:'compact'}
    };
    const phases = [
      [0, 'REFERENCE LAYER', 'The token enters the reference layer. Follow the black residual stream.'],
      [1200, 'ATTENTION', 'Attention updates the token while the original signal follows the residual path.'],
      [3150, 'STORE A REFERENCE', 'Copy the full state into the reference cache while the model keeps running.'],
      [4850, 'KEEP THE REFERENCE', 'The next layer starts. The reference stays ready in the cache.'],
      [7850, 'REUSE THE REFERENCE', 'The current input x_ℓ and cached reference z travel to the EchoMoE block and arrive together.'],
      [8250, 'FIND THE CHANGE', 'Subtract what the reference predicts; the router still reads the full current state.'],
      [10500, 'COMPRESS', 'Encode the change as a compact code, c_d.'],
      [11100, 'DISPATCH', 'Send the small code. Each expert receives z and c_d at the same time.'],
      [12100, 'BOTH INPUTS READY', 'The cached reference z and the compressed change c_d arrive together.'],
      [12750, 'EXPERT COMPUTE', 'The experts compute directly from the reference and the compressed change.'],
      [14700, 'WEIGHT THE OUTPUTS', 'Apply the routing weight to each compact expert output.'],
      [15300, 'COMBINE', 'Each weighted expert output travels directly into all-to-all combine.'],
      [16300, 'DECOMPRESS ONCE', 'Restore the full-width output once, on the token’s owner GPU.'],
      [16850, 'RESIDUAL UPDATE', 'Add the expert result back into the residual stream.'],
      [18850, 'READY TO REUSE', 'The reference stays cached for the following EchoMoE layers.']
    ];

    const highlights = [];
    function highlight(id, start, end) {
      const el = elements.get(id);
      if (el) highlights.push({el, start, end});
      if (id.startsWith('f2-r-')) highlight(id.replace('f2-r-', 'f2-m-'), start, end);
    }
    [
      ['f2-l-n0',450,900], ['f2-l-a0',1200,1750], ['f2-l-add0',2050,2350],
      ['f2-l-n1',2700,3150], ['f2-l-moe',3450,4050], ['f2-l-add1',4350,4650],
      ['f2-l-n2',5150,5600], ['f2-l-a1',5900,6450], ['f2-l-add2',6750,7050],
      ['f2-l-n3',7400,7850], ['f2-l-echo',8250,17550], ['f2-l-add3',17850,18200],
      ['f2-r-scale',8900,9350], ['f2-r-router',8900,9850], ['f2-r-subtract',9850,10150],
      ['f2-r-compress',10500,11100], ['f2-r-dispatch',11100,12100],
      ['f2-r-combine',15300,16300], ['f2-r-decompress',16300,16850]
    ].forEach(args => highlight(...args));
    for (let i=0; i<3; i++) {
      highlight('f2-r-pair'+i,12100,12450);
      highlight('f2-r-win'+i,12750,13250);
      highlight('f2-r-sigma'+i,13500,13950);
      highlight('f2-r-wout'+i,14200,14700);
      highlight('f2-r-weight'+i,14950,15300);
    }

    const packets = [];
    function packet(pathId, start, end, color='ink', count=4, options={}) {
      if (pathId.startsWith('f2-r-')) packet(pathId.replace('f2-r-', 'f2-m-'), start, end, color, count, options);
      const path = elements.get(pathId);
      if (!path) return;  // the single-column view draws one expert, not three
      const layer = pathId.slice(3, 4);
      const group = document.createElementNS(NS,'g');
      group.classList.add('f2-packet');
      group.dataset.path = pathId;
      group.dataset.startMs = String(start);
      group.dataset.arrivalMs = String(end);
      group.dataset.durationMs = String(end-start);
      group.dataset.payloadBlocks = String(count);
      group.setAttribute('fill',colors[color]);
      group.setAttribute('opacity','0');
      const w = count <= 2 ? 5.4 : 4.2;
      const gap = 1.6;
      const h = count <= 2 ? 7 : 7.8;
      const totalWidth = count*w+(count-1)*gap;
      const endPadding = options.arriveAtBoundary ? totalWidth/2 : 0;
      group.dataset.arrivalPadding = String(endPadding);
      for (let i=0; i<count; i++) {
        const block=document.createElementNS(NS,'rect');
        block.setAttribute('x',String(i*(w+gap)-totalWidth/2));
        block.setAttribute('y',String(-h/2));
        block.setAttribute('width',String(w));
        block.setAttribute('height',String(h));
        block.setAttribute('rx','1.1');
        group.appendChild(block);
      }
      packetLayers[layer].appendChild(group);
      packets.push({path, group, layer, start, end, length:null, endPadding,
        hold:options.hold||0, offset:options.offset||0});
    }
    // Left: attention and MoE sublayers plus the actual residual bypasses.
    [
      ['f2-l-input',0,450], ['f2-l-n0-a0',900,1200], ['f2-l-a0-add0',1750,2050],
      ['f2-l-add0-n1',2350,2700], ['f2-l-n1-moe',3150,3450], ['f2-l-moe-add1',4050,4350],
      ['f2-l-add1-n2',4650,5150], ['f2-l-n2-a1',5600,5900], ['f2-l-a1-add2',6450,6750],
      ['f2-l-add2-n3',7050,7400],
      ['f2-l-echo-add3',17550,17850], ['f2-l-output',18200,18850],
      ['f2-l-skip0',0,2050], ['f2-l-skip1',2350,4350], ['f2-l-skip2',4650,6750]
    ].forEach(args=>packet(...args));
    // The residual bypass waits at its addition. It is not an EchoMoE input.
    packet('f2-l-skip3',7050,8150,'ink',4,{hold:17850});
    // The current input and cached reference take 400 ms and arrive together.
    // Local reuse is quicker than the initial full-state transfer into the cache.
    packet('f2-l-n3-echo',7850,8250,'ink',4,{arriveAtBoundary:true});
    packet('f2-l-store',3150,4850,'purple',4);
    packet('f2-l-reuse',7850,8250,'purple',4,{arriveAtBoundary:true});

    // Right: full input, prediction, routing, then a small dispatched code.
    packet('f2-r-x',8250,8900,'ink',4,{hold:9850});
    packet('f2-r-to-router',8250,8900,'ink',4);
    packet('f2-r-z-scale',8250,8900,'purple',4);
    packet('f2-r-z-sub',9350,9850,'purple',4);
    packet('f2-r-sub-cd',10150,10500,'ink',4);
    packet('f2-r-router-meta',9850,10400,'orange',1);
    packet('f2-r-code-stem',11100,11400,'teal',2);
    for(let i=0;i<3;i++) {
      // Separate entry/exit paths keep dispatch's label clear. The full cached
      // state begins earlier; both inputs finish at the exact same timestamp.
      packet('f2-r-dwire'+i,11650,12100,'teal',2,{arriveAtBoundary:true});
      packet('f2-r-z'+i,10600,12100,'purple',4,{arriveAtBoundary:true});
      packet('f2-r-input-win'+i,12450,12750,'green',3);
      packet('f2-r-win-sigma'+i,13250,13500,'green',3);
      packet('f2-r-sigma-wout'+i,13950,14200,'green',3);
      packet('f2-r-wout-weight'+i,14700,14950,'teal',2);
      packet('f2-r-wmeta'+i,14700,14950,'orange',1);
      packet('f2-r-weight-combine'+i,15300,15950,'teal',2);
    }
    packet('f2-r-combine-decomp',16050,16300,'teal',2);
    packet('f2-r-output',16850,17550,'ink',4);

    function render(ms) {
      const t=Math.max(0,Math.min(duration,ms));
      nodes.forEach(el=>el.classList.remove('is-active','is-stored','is-waiting'));
      highlights.forEach(h=>{if(t>=h.start && t<h.end)h.el.classList.add('is-active');});
      if(t>=4850){
        ['f2-l-cache','f2-r-z','f2-r-local-cache','f2-m-z','f2-m-local-cache'].forEach(id=>elements.get(id)?.classList.add('is-stored'));
      }
      if(t>=8150&&t<17850)elements.get('f2-l-add3').classList.add('is-waiting');
      root.querySelector('.f2-detail-panel').classList.toggle('is-muted',t<7850);
      const nextView=t>=detailWindow[0]&&t<detailWindow[1]?'detail':'model';
      if(nextView!==view){
        view=nextView;
        root.dataset.view=view;
        viewButtons.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.f2View===view)));
      }
      const hiddenLayer=narrow.matches?'r':'m';
      packets.forEach(p=>{
        if(p.layer===hiddenLayer){p.group.setAttribute('opacity','0');return;}
        if(p.length===null)p.length=p.path.getTotalLength();
        const inTravel=t>=p.start&&t<p.end;
        const inHold=p.hold&&t>=p.end&&t<p.hold;
        if(!inTravel&&!inHold){p.group.setAttribute('opacity','0');return;}
        const raw=inHold ? 0.96 : (t-p.start)/(p.end-p.start);
        const progress=Math.min(1,Math.max(0,raw));
        const dist=progress*Math.max(0,p.length-p.endPadding);
        const pt=p.path.getPointAtLength(dist);
        const before=p.path.getPointAtLength(Math.max(0,dist-1));
        const after=p.path.getPointAtLength(Math.min(p.length,dist+1));
        const angle=Math.atan2(after.y-before.y,after.x-before.x)*180/Math.PI;
        p.group.setAttribute('transform','translate('+pt.x.toFixed(2)+' '+pt.y.toFixed(2)+') rotate('+angle.toFixed(2)+')');
        p.group.setAttribute('opacity',inHold?'.62':'1');
      });
      let phase=0;
      for(let i=1;i<phases.length;i++){if(t>=phases[i][0])phase=i;else break;}
      if(phase!==lastPhase){
        controls.tag.textContent=phases[phase][1];
        controls.caption.textContent=phases[phase][2];
        lastPhase=phase;
      }
      controls.scrubber.value=String(Math.round(t));
      controls.scrubber.style.setProperty('--f2-progress',(t/duration*100).toFixed(2)+'%');
      controls.scrubber.setAttribute('aria-valuetext',(t/1000).toFixed(1)+' of '+(duration/1000).toFixed(1)+' seconds. '+phases[phase][1].toLowerCase());
    }
    function updateControl(){
      root.classList.toggle('is-paused',paused);
      controls.play.querySelector('.f2-play-symbol').innerHTML=paused
        ?'<path d="M3 1.4 11 6 3 10.6Z"></path>'
        :'<rect x="2" y="1" width="3" height="10" rx=".8"></rect><rect x="7" y="1" width="3" height="10" rx=".8"></rect>';
      controls.play.querySelector('.f2-play-label').textContent=paused?'Play':'Pause';
      controls.play.setAttribute('aria-label',(paused?'Play':'Pause')+' architecture animation');
    }
    function schedule(){
      if(!raf&&!paused&&visible&&!document.hidden)raf=requestAnimationFrame(frame);
    }
    function frame(stamp){
      raf=0;
      if(paused||!visible||document.hidden){lastStamp=null;return;}
      if(lastStamp!==null)time=(time+Math.min(100,stamp-lastStamp)*speed)%duration;
      lastStamp=stamp;
      render(time);
      schedule();
    }
    function pause(){paused=true;lastStamp=null;if(raf)cancelAnimationFrame(raf);raf=0;updateControl();}
    function play(){paused=false;lastStamp=null;if(time>=duration)time=0;updateControl();schedule();}
    function seek(ms){time=Math.max(0,Math.min(duration,Number(ms)||0));lastStamp=null;render(time);}
    controls.play.addEventListener('click',()=>paused?play():pause());
    controls.replay.addEventListener('click',()=>{seek(0);play();});
    controls.speed.forEach(btn=>btn.addEventListener('click',()=>{
      speed=Number(btn.dataset.speed)||1;
      lastStamp=null;
      controls.speed.forEach(b=>b.setAttribute('aria-pressed',String(b===btn)));
    }));
    controls.scrubber.addEventListener('input',()=>{pause();seek(controls.scrubber.value);});
    viewButtons.forEach(btn=>btn.addEventListener('click',()=>{
      if(btn.dataset.f2View!==view)seek(btn.dataset.f2View==='detail'?detailWindow[0]:0);
    }));
    narrow.addEventListener('change',()=>render(time));
    if('IntersectionObserver' in window){
      const observer=new IntersectionObserver(entries=>{
        visible=entries[0].isIntersecting;
        lastStamp=null;
        if(!visible&&raf){cancelAnimationFrame(raf);raf=0;}
        if(visible)schedule();
      },{threshold:0.06});
      observer.observe(root);
    }
    document.addEventListener('visibilitychange',()=>{
      lastStamp=null;
      if(document.hidden&&raf){cancelAnimationFrame(raf);raf=0;}
      else schedule();
    });
    window.EchoFigure2={
      seek,pause,play,
      getState:()=>({time,duration,paused,speed,visible,phase:phases[lastPhase][1],cacheFilled:time>=4850,
        active:Array.from(root.querySelectorAll('.f2-compute.is-active')).map(el=>el.id),
        synchronizedEchoArrival:8250,
        synchronizedExpertArrival:12100,
        transfers:Object.fromEntries(Object.entries(transfers).map(([key,value])=>[key,{...value}])),
        visiblePackets:packets.filter(p=>p.group.getAttribute('opacity')!=='0').length})
    };
    updateControl();render(time);schedule();
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',initFigure2,{once:true});
  else initFigure2();
})();
