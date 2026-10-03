// Minimal DOM harness: runs the actual inline page script without a browser/server.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const path=require('node:path');const html=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
class Element{
 constructor(tag='div'){this.tagName=tag;this.children=[];this.dataset={};this.style={};this.attributes={};this.listeners={};this._text='';this.className='';this.hidden=false;this.classList={toggle:(name,on)=>{const s=new Set(this.className.split(' ').filter(Boolean));on?s.add(name):s.delete(name);this.className=[...s].join(' ');}};}
 set textContent(v){this._text=String(v);this.children=[];}get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
 append(...children){this.children.push(...children);}prepend(c){this.children.unshift(c);}replaceChildren(...c){this._text='';this.children=c;}
 setAttribute(k,v){this.attributes[k]=v;}addEventListener(k,fn){this.listeners[k]=fn;}click(){this.listeners.click?.();}
}
const elements={};for(const m of html.matchAll(/id="([^"]+)"/g))elements[m[1]]=new Element();
elements['model-weights'].textContent=html.match(/<script id="model-weights" type="application\/json">([\s\S]*?)<\/script>/)[1];
function all(){const result=[];function visit(e){result.push(e);e.children.forEach(visit);}Object.values(elements).forEach(visit);return result;}
const document={getElementById:id=>{assert(elements[id],`Missing #${id}`);return elements[id];},createElement:tag=>new Element(tag),querySelectorAll:selector=>all().filter(e=>e.className.split(' ').includes(selector.slice(1))),querySelector:selector=>document.querySelectorAll(selector)[0]};
let ticks=0;const intervals=[];const context={document,performance,console,Float64Array,Math,window:{},setInterval:(fn,ms)=>intervals.push({fn,ms}),fetch:()=>{throw Error('Forbidden fetch');},XMLHttpRequest:()=>{throw Error('Forbidden XHR');}};
vm.createContext(context);const script=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];vm.runInContext(script,context);
const generated=elements['generated-text'];assert(generated.textContent.length>20);assert.equal(elements['log-body'].children.length,24);
const initial=generated.textContent;const presets=document.querySelectorAll('.preset-prompt');assert(presets.length>=18);assert(presets.every(e=>e.dataset.category));
function checkBars(){const bars=document.querySelectorAll('.prob-bar');assert(bars.length>=5);const sum=bars.reduce((s,e)=>s+Number(e.dataset.prob),0);assert(Math.abs(sum-1)<1e-12);assert(bars.every(e=>Number.isFinite(Number(e.dataset.prob))));}checkBars();
for(const b of presets){b.click();assert(generated.textContent.startsWith(b.dataset.prompt));checkBars();}
function advance(ms){for(const timer of intervals)for(let i=0;i<Math.floor(ms/timer.ms);i++){timer.fn();ticks++;}}
elements['pause-autoplay'].click();const paused=generated.textContent;advance(560);assert.equal(generated.textContent,paused);elements['pause-autoplay'].click();advance(980);assert(generated.textContent.length>paused.length);advance(5000);assert(elements['log-body'].children.length>=15);for(const row of elements['log-body'].children){assert.equal(row.children.length,3);assert(row.children[1].textContent.length);assert(/\d+\.\d+%/.test(row.children[2].textContent));}checkBars();
for(const m of html.matchAll(/(?:src|href)\s*=\s*["']([^"']+)["']/gi)){const ref=m[1];assert(!/^https?:|^\/\//.test(ref),'External asset');if(!/^(?:data:|#)/.test(ref))assert(fs.existsSync(path.resolve(__dirname,'..',ref)));}
const forward=context.window.pocketTransformer.forward;for(const prompt of ['Alice was beginning ','café','こんにちは','1234567890','']){const p=Array.from(forward(prompt));assert(p.every(x=>Number.isFinite(x)&&x>=0));assert(Math.abs(p.reduce((a,b)=>a+b,0)-1)<1e-12);}
fs.writeFileSync('/private/tmp/pocket-js-probs.json',JSON.stringify(Array.from(forward('Alice was beginning '))));
console.log(JSON.stringify({result:'PASS',initialText:initial,presets:presets.length,logRows:elements['log-body'].children.length,timerTicks:ticks,parameters:Object.values(context.window.pocketTransformer.model.weights).reduce((s,w)=>s+w.data.length,0)},null,2));
