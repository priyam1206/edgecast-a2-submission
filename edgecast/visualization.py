from __future__ import annotations

import json
from typing import Dict, Iterable

MODEL_PALETTE = [
    "#8b5cf6",
    "#22d3ee",
    "#f59e0b",
    "#34d399",
    "#60a5fa",
    "#f472b6",
    "#fb7185",
    "#a3e635",
]


def model_colors(models: Iterable[str]) -> Dict[str, str]:
    return {model: MODEL_PALETTE[i % len(MODEL_PALETTE)] for i, model in enumerate(models)}


def _result_frames(result):
    models = list(result.queues.keys())
    frames = []
    for i, slot in enumerate(result.slots):
        frames.append(
            {
                "slot": int(slot),
                "queues": {m: int(result.queues[m][i]) for m in models},
                "arrivals": {m: int(result.arrivals[m][i]) for m in models},
                "allocations": result.allocations[i],
                "nodes": result.node_states[i] if i < len(result.node_states) else [],
                "decision": result.decisions[i] if i < len(result.decisions) else {},
            }
        )
    return models, frames


def build_comparison_animation(
    base_result,
    enhanced_result,
    *,
    height: int = 1380,
    layout: str = "stacked",
    run_id: str | None = None,
) -> str:
    base_models, base_frames = _result_frames(base_result)
    enhanced_models, enhanced_frames = _result_frames(enhanced_result)
    models = list(dict.fromkeys(base_models + enhanced_models))
    colors = model_colors(models)
    side = layout == "side"

    payload = json.dumps(
        {
            "models": models,
            "colors": colors,
            "layout": "side" if side else "stacked",
            "run_id": run_id or "",
            "base": {
                "name": base_result.policy_name,
                "frames": base_frames,
                "summary": base_result.summary(),
            },
            "enhanced": {
                "name": enhanced_result.policy_name,
                "frames": enhanced_frames,
                "summary": enhanced_result.summary(),
            },
        }
    )

    stack_css = (
        "display:grid;grid-template-columns:1fr 1fr;gap:14px;"
        if side
        else "display:flex;flex-direction:column;gap:18px;"
    )
    canvas_height = 590 if side else 540
    viewbox = "0 0 820 590" if side else "0 0 1200 560"

    return f'''<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
:root {{
  --bg:#050505;
  --panel:#090909;
  --text:#f5f5f5;
  --muted:#8f8f8f;
  --line:rgba(255,255,255,.10);
  --line2:rgba(255,255,255,.18);
  --cyan:#22d3ee;
  --green:#34d399;
  --amber:#f59e0b;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0;
  background:#050505;
  color:var(--text);
  font-family:"Segoe UI",Arial,sans-serif;
  overflow-x:hidden;
  overflow-y:auto;
}}
.wrap {{
  min-height:{height}px;
  height:auto;
  padding:8px;
}}
.topbar {{
  display:flex;
  align-items:center;
  gap:12px;
  padding:8px 11px;
  margin-bottom:8px;
  border:1px solid var(--line);
  border-radius:14px;
  background:rgba(7,7,7,.97);
  box-shadow:0 18px 48px rgba(0,0,0,.30);
}}
.play {{
  min-width:86px;
  height:40px;
  border:1px solid var(--line2);
  background:#111;
  color:#fff;
  border-radius:10px;
  font-size:13px;
  font-weight:700;
  cursor:pointer;
}}
.timeline {{ flex:1; accent-color:#f3f3f3; }}
.speed {{
  height:40px;
  background:#0d0d0d;
  color:#ddd;
  border:1px solid var(--line);
  border-radius:10px;
  padding:0 10px;
}}
.slot {{
  min-width:145px;
  color:#b7b7b7;
  font-size:12px;
  text-align:right;
  font-variant-numeric:tabular-nums;
}}
.sync-pill {{
  white-space:nowrap;
  color:#c7c7c7;
  font-size:10px;
  font-weight:700;
  letter-spacing:.08em;
  border:1px solid var(--line);
  padding:7px 9px;
  border-radius:999px;
  background:#0a0a0a;
}}
.delta-strip {{
  display:grid;
  grid-template-columns:repeat(4,1fr);
  gap:10px;
  margin:0 0 8px;
}}
.delta-card {{
  border:1px solid var(--line);
  border-radius:12px;
  padding:8px 12px;
  background:#080808;
}}
.delta-card small {{
  color:#777;
  font-size:9.5px;
  text-transform:uppercase;
  letter-spacing:.10em;
}}
.delta-card strong {{
  display:block;
  margin-top:3px;
  font-size:16px;
  color:#f4f4f4;
}}
.stack {{ {stack_css} }}
.panel {{
  position:relative;
  border:1px solid var(--line);
  border-radius:20px;
  background:
    radial-gradient(circle at 50% 42%,rgba(255,255,255,.035),transparent 36%),
    linear-gradient(180deg,#0a0a0a,#060606);
  overflow:hidden;
  min-width:0;
  box-shadow:0 24px 70px rgba(0,0,0,.35);
  animation:panelIn .5s ease both;
}}
.panel.edgecast {{ animation-delay:.08s; }}
.panel-head {{
  min-height:70px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:14px;
  padding:12px 16px;
  border-bottom:1px solid var(--line);
  background:linear-gradient(180deg,rgba(255,255,255,.025),rgba(255,255,255,.008));
}}
.panel-title {{
  display:flex;
  flex-direction:column;
  gap:4px;
}}
.panel-title b {{
  font-size:{'17px' if side else '16px'};
  letter-spacing:.06em;
}}
.panel-title span {{
  font-size:{'11.5px' if side else '11px'};
  color:#a8a8a8;
  line-height:1.35;
}}
.method-badge {{
  font-size:{'9px' if side else '10px'};
  color:#d0d0d0;
  padding:6px 8px;
  border:1px solid var(--line2);
  border-radius:999px;
  background:#0a0a0a;
  white-space:nowrap;
}}
.canvas {{
  position:relative;
  height:{canvas_height}px;
}}
svg {{ width:100%; height:100%; }}
.edge-line {{
  stroke:rgba(255,255,255,.11);
  stroke-width:1.35;
  transition:stroke .25s ease, stroke-width .25s ease;
}}
.edge-line.prefetch {{
  stroke:var(--cyan)!important;
  stroke-width:2.1;
  stroke-dasharray:4 8;
  animation:dash .58s linear infinite;
  filter:drop-shadow(0 0 5px rgba(34,211,238,.45));
}}
.node-shell {{
  fill:#090909;
  stroke:rgba(255,255,255,.11);
  stroke-width:1.2;
}}
.node-ring {{
  fill:none;
  stroke-width:3.2;
  opacity:.98;
  transition:stroke .25s ease, filter .25s ease;
}}
.node-ring.switching {{
  stroke-dasharray:7 6;
  animation:dash .72s linear infinite, switchPulse 1.15s ease-in-out infinite;
}}
.node-ring.staging {{
  stroke:#22d3ee !important;
  stroke-dasharray:4 6;
  filter:drop-shadow(0 0 9px rgba(34,211,238,.55));
  animation:dash .65s linear infinite;
}}
.node-title {{
  fill:#f7f7f7;
  font-size:{'14.5px' if side else '14px'};
  font-weight:800;
  text-anchor:middle;
}}
.node-sub {{
  fill:#b2b2b2;
  font-size:{'11.5px' if side else '11px'};
  text-anchor:middle;
}}
.node-cache {{
  fill:#969696;
  font-size:{'10px' if side else '9.5px'};
  text-anchor:middle;
}}
.center-shell {{
  fill:#070707;
  stroke:rgba(255,255,255,.34);
  stroke-width:1.7;
  filter:drop-shadow(0 0 24px rgba(255,255,255,.11));
  animation:repoPulse 3s ease-in-out infinite;
  transform-origin:center;
}}
.center-title {{
  fill:white;
  font-size:{'15.5px' if side else '16px'};
  font-weight:850;
  text-anchor:middle;
}}
.center-sub {{
  fill:#a0a0a0;
  font-size:{'10px' if side else '10px'};
  text-anchor:middle;
}}
.qbar-bg {{ fill:rgba(255,255,255,.06); }}
.qbar {{ transition:width .35s cubic-bezier(.2,.8,.2,1); }}
.qtext {{
  fill:#b8b8b8;
  font-size:{'10.5px' if side else '10px'};
}}
.decision {{
  position:absolute;
  left:16px;
  bottom:16px;
  width:{'58%' if side else 'min(520px,52%)'};
  min-height:90px;
  padding:13px 15px;
  border:1px solid var(--line2);
  border-radius:14px;
  background:rgba(4,4,4,.93);
  backdrop-filter:blur(14px);
  box-shadow:0 12px 36px rgba(0,0,0,.36);
}}
.decision.flash {{ animation:decisionFlash .55s ease; }}
.decision .k {{
  color:#717171;
  font-size:10px;
  letter-spacing:.15em;
  text-transform:uppercase;
  margin-bottom:6px;
}}
.decision .v {{
  color:#dedede;
  font-size:{'12.5px' if side else '13px'};
  line-height:1.45;
}}
.legend {{
  position:absolute;
  right:16px;
  bottom:16px;
  display:flex;
  gap:6px;
  flex-wrap:wrap;
  justify-content:flex-end;
  max-width:{'205px' if side else '390px'};
}}
.legend span {{
  font-size:{'9.5px' if side else '10px'};
  color:#b8b8b8;
  padding:5px 7px;
  border:1px solid var(--line);
  border-radius:999px;
  background:rgba(0,0,0,.48);
}}
.backhaul {{
  position:absolute;
  top:12px;
  left:50%;
  transform:translateX(-50%);
  display:flex;
  align-items:center;
  gap:6px;
  font-size:{'9px' if side else '10px'};
  color:#969696;
  padding:5px 8px;
  border:1px solid var(--line);
  border-radius:999px;
  background:rgba(0,0,0,.58);
  z-index:4;
}}
.backhaul .lamp {{
  width:6px;
  height:6px;
  border-radius:50%;
  background:#555;
}}
.backhaul.busy {{
  color:#e1e1e1;
  border-color:rgba(34,211,238,.45);
  box-shadow:0 0 22px rgba(34,211,238,.10);
}}
.backhaul.busy .lamp {{
  background:var(--cyan);
  box-shadow:0 0 12px rgba(34,211,238,.85);
  animation:blink .85s ease-in-out infinite;
}}
.packet {{ filter:drop-shadow(0 0 5px rgba(34,211,238,.8)); }}
@keyframes dash {{ to {{ stroke-dashoffset:-12; }} }}
@keyframes repoPulse {{
  0%,100% {{ opacity:.82; transform:scale(.985); }}
  50% {{ opacity:1; transform:scale(1.025); }}
}}
@keyframes switchPulse {{
  0%,100% {{ filter:drop-shadow(0 0 0 rgba(245,158,11,0)); }}
  50% {{ filter:drop-shadow(0 0 9px rgba(245,158,11,.6)); }}
}}
@keyframes blink {{ 0%,100% {{ opacity:.35; }} 50% {{ opacity:1; }} }}
@keyframes panelIn {{
  from {{ opacity:0; transform:translateY(10px); }}
  to {{ opacity:1; transform:translateY(0); }}
}}
@keyframes decisionFlash {{
  0% {{ transform:translateY(4px); border-color:rgba(255,255,255,.12); }}
  45% {{ transform:translateY(0); border-color:rgba(255,255,255,.40); }}
  100% {{ border-color:rgba(255,255,255,.18); }}
}}
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <button class="play" id="play">Pause</button>
    <input class="timeline" id="timeline" type="range" min="0" max="1" value="0"/>
    <select class="speed" id="speed">
      <option value="1000">0.5x</option>
      <option value="600" selected>1x</option>
      <option value="300">2x</option>
      <option value="140">4x</option>
    </select>
    <div class="slot" id="slotLabel"></div>
    <div class="sync-pill">SAME DATA / SHARED CLOCK</div>
  </div>

  <div class="delta-strip">
    <div class="delta-card"><small>Live backlog · paper → EdgeCast</small><strong><span id="baseBacklog">0</span> → <span id="enhBacklog">0</span></strong></div>
    <div class="delta-card"><small>Live control action</small><strong><span id="baseAction">HOLD</span> / <span id="enhAction">HOLD</span></strong></div>
    <div class="delta-card"><small>Final avg latency</small><strong id="latencyOutcome">—</strong></div>
    <div class="delta-card"><small>Final total switches</small><strong id="switchOutcome">—</strong></div>
  </div>

  <div class="stack">
    <section class="panel base">
      <div class="panel-head">
        <div class="panel-title">
          <b>1. ASSIGNED PAPER - STAGE 2 MODEL</b>
          <span>P2b aggregate queue service + exact Tp - Tc - sigma reallocation objective in the demo abstraction</span>
        </div>
        <div class="method-badge">NO PERSISTENT INACTIVE CACHE / NO BACKGROUND STAGING</div>
      </div>
      <div class="canvas">
        <div class="backhaul"><span class="lamp"></span><span>cold model deployment when allocation changes</span></div>
        <svg id="baseSvg" viewBox="{viewbox}"></svg>
        <div class="decision" id="baseDecisionBox">
          <div class="k">Paper-model decision</div>
          <div class="v" id="baseDecision">Waiting for trace...</div>
        </div>
        <div class="legend" id="baseLegend"></div>
      </div>
    </section>

    <section class="panel edgecast">
      <div class="panel-head">
        <div class="panel-title">
          <b>2. EDGECAST IMPROVEMENT</b>
          <span>Persistent cache + background staging + warm/cold reallocation + minimum-persistence anti-thrashing</span>
        </div>
        <div class="method-badge">CACHE + STAGING + STABILITY CONTROL</div>
      </div>
      <div class="canvas">
        <div class="backhaul" id="enhBackhaul"><span class="lamp"></span><span id="enhBackhaulText">shared backhaul idle</span></div>
        <svg id="enhSvg" viewBox="{viewbox}"></svg>
        <div class="decision" id="enhDecisionBox">
          <div class="k">EdgeCast decision</div>
          <div class="v" id="enhDecision">Waiting for trace...</div>
        </div>
        <div class="legend" id="enhLegend"></div>
      </div>
    </section>
  </div>
</div>

<script>
const DATA={payload};
const SIDE=DATA.layout==='side';
const NS='http://www.w3.org/2000/svg';
const timeline=document.getElementById('timeline');
const playBtn=document.getElementById('play');
const speedSel=document.getElementById('speed');
const slotLabel=document.getElementById('slotLabel');
const baseDecision=document.getElementById('baseDecision');
const enhDecision=document.getElementById('enhDecision');
const baseDecisionBox=document.getElementById('baseDecisionBox');
const enhDecisionBox=document.getElementById('enhDecisionBox');
const enhBackhaul=document.getElementById('enhBackhaul');
const enhBackhaulText=document.getElementById('enhBackhaulText');
const baseBacklog=document.getElementById('baseBacklog');
const enhBacklog=document.getElementById('enhBacklog');
const baseAction=document.getElementById('baseAction');
const enhAction=document.getElementById('enhAction');
const latencyOutcome=document.getElementById('latencyOutcome');
const switchOutcome=document.getElementById('switchOutcome');

function lowerText(baseValue,newValue,digits=1) {{
  const b=Number(baseValue), n=Number(newValue);
  if(!Number.isFinite(b) || !Number.isFinite(n) || b===0) return '';
  const pct=100*(b-n)/b;
  return (pct>=0?pct.toFixed(digits)+'% lower':Math.abs(pct).toFixed(digits)+'% higher');
}}
latencyOutcome.textContent=
  Number(DATA.base.summary.avg_latency).toFixed(2)+' → '+
  Number(DATA.enhanced.summary.avg_latency).toFixed(2)+' · '+
  lowerText(DATA.base.summary.avg_latency,DATA.enhanced.summary.avg_latency);
switchOutcome.textContent=
  DATA.base.summary.switches+' → '+DATA.enhanced.summary.switches+' · '+
  lowerText(DATA.base.summary.switches,DATA.enhanced.summary.switches);

const maxFrames=Math.max(DATA.base.frames.length,DATA.enhanced.frames.length);
timeline.max=Math.max(0,maxFrames-1);

function el(tag,attrs={{}},text='') {{
  const e=document.createElementNS(NS,tag);
  Object.entries(attrs).forEach(function(kv) {{ e.setAttribute(kv[0],kv[1]); }});
  if(text)e.textContent=text;
  return e;
}}

const positions=SIDE
  ? [[130,155],[130,295],[130,435],[690,155],[690,295],[690,435],[300,105],[520,105]]
  : [[185,135],[185,280],[185,425],[1015,135],[1015,280],[1015,425],[390,105],[810,105],[390,455],[810,455],[600,85],[600,475]];
const center=SIDE?[410,295]:[600,280];
const NODE_W=SIDE?150:170;
const NODE_H=SIDE?90:100;
const REPO_R=SIDE?60:70;
const BAR_W=SIDE?170:185;
const BAR_STEP=SIDE?185:205;
const BAR_X=SIDE?18:24;

function setupPanel(svgId,legendId,showCache) {{
  const svg=document.getElementById(svgId);
  const maxNodes=Math.max(
    1,
    ...DATA.base.frames.map(function(f) {{ return f.allocations.length; }}),
    ...DATA.enhanced.frames.map(function(f) {{ return f.allocations.length; }})
  );

  const lines=[];
  for(let i=0;i<maxNodes;i++) {{
    const pos=positions[i%positions.length];
    const line=el('line',{{x1:center[0],y1:center[1],x2:pos[0],y2:pos[1],class:'edge-line'}});
    svg.appendChild(line);
    lines.push(line);
  }}

  const repo=el('g');
  repo.appendChild(el('circle',{{cx:center[0],cy:center[1],r:REPO_R,class:'center-shell'}}));
  repo.appendChild(el('text',{{x:center[0],y:center[1]-10,class:'center-title'}},'MODEL REPOSITORY'));
  repo.appendChild(el('text',{{x:center[0],y:center[1]+11,class:'center-sub'}},showCache?'shared staging backhaul':'deployment source'));
  repo.appendChild(el('text',{{x:center[0],y:center[1]+29,class:'center-sub'}},showCache?'stage | cache | warm cutover':'cold deployment on reallocation'));
  svg.appendChild(repo);

  const nodes=[];
  for(let i=0;i<maxNodes;i++) {{
    const pos=positions[i%positions.length];
    const x=pos[0], y=pos[1];
    const g=el('g');
    const shell=el('rect',{{x:x-NODE_W/2,y:y-NODE_H/2,width:NODE_W,height:NODE_H,rx:19,class:'node-shell'}});
    const ring=el('rect',{{x:x-NODE_W/2,y:y-NODE_H/2,width:NODE_W,height:NODE_H,rx:19,class:'node-ring'}});
    const title=el('text',{{x:x,y:y-19,class:'node-title'}},'EDGE '+(i+1));
    const sub=el('text',{{x:x,y:y+3,class:'node-sub'}},'');
    const cache=el('text',{{x:x,y:y+27,class:'node-cache'}},'');
    g.append(shell,ring,title,sub,cache);
    svg.appendChild(g);
    nodes.push({{ring:ring,sub:sub,cache:cache,x:x,y:y}});
  }}

  DATA.models.forEach(function(m,i) {{
    const x=BAR_X+i*BAR_STEP;
    svg.append(
      el('rect',{{x:x,y:20,width:BAR_W,height:7,rx:3.5,class:'qbar-bg'}}),
      el('rect',{{x:x,y:20,width:0,height:7,rx:3.5,fill:DATA.colors[m],class:'qbar',id:svgId+'q'+i}}),
      el('text',{{x:x,y:43,class:'qtext',id:svgId+'qt'+i}},m)
    );
  }});

  document.getElementById(legendId).innerHTML=DATA.models
    .map(function(m) {{
      return '<span><b style="color:'+DATA.colors[m]+'">&#9679;</b>&nbsp;'+m+'</span>';
    }})
    .join('');

  return {{svg:svg,nodes:nodes,lines:lines,showCache:showCache}};
}}

const basePanel=setupPanel('baseSvg','baseLegend',false);
const enhPanel=setupPanel('enhSvg','enhLegend',true);

function frameAt(frames,index) {{
  if(!frames.length)return null;
  return frames[Math.min(index,frames.length-1)];
}}

function decisionHtml(d) {{
  d=d||{{}};
  if(d.stability_gate==='minimum-persistence') {{
    return '<b style="color:#34d399">STABILITY HOLD</b> | positive P3 candidate suppressed'
      +'<br><span style="color:#888">minimum allocation persistence prevents warm-switch oscillation'
      +(d.remaining_persistence_slots!==undefined?' · '+d.remaining_persistence_slots+' slot(s) remaining':'')
      +' · candidate benefit '+fmt(d.net_benefit)+'</span>';
  }}
  if(d.action==='prefetch') {{
    return '<b style="color:#22d3ee">BACKGROUND STAGE</b> | '
      +(d.target||'')+' -> '+(d.edge||'edge')
      +'<br><span style="color:#888">current inference continues while the shared backhaul transfers the checkpoint</span>';
  }}
  if(d.action==='multi_switch') {{
    const count=d.changes||((d.change_details||[]).length);
    const benefit=(d.net_benefit===null || d.net_benefit===undefined)?'INF':Number(d.net_benefit).toFixed(2);
    const isEdgeCast=String(d.method||'').toLowerCase().includes('edgecast');
    const mode=String(d.mode||'').toLowerCase();
    const color=isEdgeCast
      ? (mode==='warm'?'#34d399':(mode==='cold'?'#f59e0b':'#22d3ee'))
      : '#f59e0b';
    const label=isEdgeCast?'CACHE-AWARE P3 REALLOCATION':'P3 REALLOCATION';
    return '<b style="color:'+color+'">'+label+'</b> | '+count+' edge change(s)'
      +'<br><span style="color:#888">Tp '+fmt(d.Tp)+' | Tc '+fmt(d.Tc)+' | sigma '+fmt(d.sigma)
      +' | benefit '+benefit+(mode?' | '+mode.toUpperCase():'')+'</span>';
  }}
  if(d.action==='switch') {{
    const mode=String(d.mode||'cold').toLowerCase();
    const color=mode==='warm'?'#34d399':'#f59e0b';
    return '<b style="color:'+color+'">'+mode.toUpperCase()+' REALLOCATION</b> | '
      +(d.source||'')+' -> <b>'+(d.target||'')+'</b>'
      +'<br><span style="color:#888">service interruption '+(d.switch_slots||'?')
      +' slot(s) | net benefit '+fmt(d.net_benefit)+'</span>';
  }}
  return '<b>HOLD CURRENT ALLOCATION</b><br><span style="color:#9a9a9a">'
    +(d.reason||'No active-model change is justified at this slot.')+'</span>';
}}

function fmt(v) {{
  if(v===null || v===undefined)return '-';
  const n=Number(v);
  return Number.isFinite(n)?n.toFixed(2):String(v);
}}

function backlog(frame) {{
  if(!frame)return 0;
  return Object.values(frame.queues).reduce(function(a,b) {{ return a+b; }},0);
}}

function actionLabel(frame) {{
  if(!frame || !frame.decision)return 'HOLD';
  const d=frame.decision;
  if(d.stability_gate==='minimum-persistence')return 'STABLE HOLD';
  if(d.action==='prefetch')return 'STAGE';
  if(d.action==='multi_switch') {{
    const isEdgeCast=String(d.method||'').toLowerCase().includes('edgecast');
    const mode=String(d.mode||'').toUpperCase();
    return isEdgeCast?(mode?mode+' P3':'CACHE P3'):'P3 REALLOC';
  }}
  if(d.action==='switch')return String(d.mode||'cold').toUpperCase();
  return 'HOLD';
}}

function flash(box) {{
  box.classList.remove('flash');
  void box.offsetWidth;
  box.classList.add('flash');
}}

function spawnPacket(panel,nodeIndex) {{
  const n=panel.nodes[nodeIndex];
  if(!n)return;
  const p=el('circle',{{cx:center[0],cy:center[1],r:4,fill:'#22d3ee',class:'packet'}});
  panel.svg.appendChild(p);
  const dx=n.x-center[0], dy=n.y-center[1];
  const anim=p.animate(
    [
      {{transform:'translate(0px,0px)',opacity:0}},
      {{transform:'translate('+(dx*.45)+'px,'+(dy*.45)+'px)',opacity:1,offset:.45}},
      {{transform:'translate('+dx+'px,'+dy+'px)',opacity:0}}
    ],
    {{duration:820,easing:'cubic-bezier(.2,.8,.2,1)'}}
  );
  anim.onfinish=function() {{ p.remove(); }};
}}

function renderPanel(panel,frame,enhanced) {{
  if(!frame)return null;
  const maxQ=Math.max(1,...Object.values(frame.queues));

  DATA.models.forEach(function(m,j) {{
    const bar=document.getElementById(panel.svg.id+'q'+j);
    const label=document.getElementById(panel.svg.id+'qt'+j);
    bar.setAttribute('width',Math.min(BAR_W,BAR_W*(frame.queues[m]/maxQ)));
    label.textContent=m+' | Q '+frame.queues[m]+' | +'+(frame.arrivals[m]||0);
  }});

  let staging=null;
  frame.nodes.forEach(function(s,i) {{
    const n=panel.nodes[i];
    if(!n)return;
    const shown=s.target||s.active;
    const color=DATA.colors[shown]||'#aaa';

    n.ring.setAttribute('stroke',color);
    let cls='node-ring';
    if(s.switch_remaining>0)cls+=' switching';
    else if(enhanced && s.prefetch)cls+=' staging';
    n.ring.setAttribute('class',cls);

    if(s.switch_remaining>0) {{
      n.sub.textContent=s.active+' -> '+s.target+' | '+String(s.switch_mode||'cold').toUpperCase();
    }} else if(enhanced && s.prefetch) {{
      n.sub.textContent=s.active+' | staging '+s.prefetch+' ('+s.prefetch_remaining+')';
      staging={{index:i,target:s.prefetch}};
      if(Math.random()>.45)spawnPacket(panel,i);
    }} else {{
      n.sub.textContent=s.active;
    }}

    n.cache.textContent=enhanced
      ? 'cache ['+(s.cache||[]).join(', ')+']'
      : 'active model only';

    panel.lines[i].setAttribute('class','edge-line'+(enhanced && s.prefetch?' prefetch':''));
  }});

  return staging;
}}

let idx=0;
let playing=true;
let timer=null;
let prevBaseAction='';
let prevEnhAction='';

function render(index) {{
  idx=Math.max(0,Math.min(maxFrames-1,index));
  timeline.value=idx;

  const bf=frameAt(DATA.base.frames,idx);
  const ef=frameAt(DATA.enhanced.frames,idx);

  renderPanel(basePanel,bf,false);
  const staging=renderPanel(enhPanel,ef,true);

  baseDecision.innerHTML=decisionHtml(bf?bf.decision:{{}});
  enhDecision.innerHTML=decisionHtml(ef?ef.decision:{{}});

  const ba=actionLabel(bf);
  const ea=actionLabel(ef);
  if(ba!==prevBaseAction)flash(baseDecisionBox);
  if(ea!==prevEnhAction)flash(enhDecisionBox);
  prevBaseAction=ba;
  prevEnhAction=ea;

  baseBacklog.textContent=String(backlog(bf));
  enhBacklog.textContent=String(backlog(ef));
  baseAction.textContent=ba;
  enhAction.textContent=ea;

  if(staging) {{
    enhBackhaul.className='backhaul busy';
    enhBackhaulText.textContent='staging '+staging.target+' | shared backhaul active';
  }} else {{
    enhBackhaul.className='backhaul';
    enhBackhaulText.textContent='shared backhaul idle';
  }}

  const slot=(bf&&bf.slot!==undefined)?bf.slot:((ef&&ef.slot!==undefined)?ef.slot:idx);
  slotLabel.textContent='SLOT '+slot+' | FRAME '+(idx+1)+' OF '+maxFrames;
}}

function schedule() {{
  clearTimeout(timer);
  if(!playing)return;
  timer=setTimeout(function() {{
    render((idx+1)%maxFrames);
    schedule();
  }},Number(speedSel.value));
}}

playBtn.onclick=function() {{
  playing=!playing;
  playBtn.textContent=playing?'Pause':'Play';
  schedule();
}};
timeline.oninput=function(e) {{ render(Number(e.target.value)); }};
speedSel.onchange=schedule;

render(0);
schedule();
</script>
</body>
</html>'''


def build_network_animation(result, *, height: int = 610) -> str:
    return build_comparison_animation(result, result, height=height)


def compact_decision_log(result, limit: int = 30):
    return [
        d
        for d in result.decisions
        if d and d.get("action") in {"prefetch", "switch", "multi_switch"}
    ][:limit]
