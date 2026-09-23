// DOM unit checks with a minimal test double; not a visual browser test.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Element {
 constructor(tag){this.tagName=tag;this.children=[];this.dataset={};this.style={};this.hidden=false;this.value='';this.listeners={};this._text='';this.classList={toggle(){}};}
 set textContent(t){this._text=String(t);this.children=[];}
 get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
 setAttribute(k,v){this[k]=v;}
 append(...children){this.children.push(...children);}
 replaceChildren(...children){this.children=children;this._text='';}
 addEventListener(name,fn){this.listeners[name]=fn;}
}
const ids=['diagram','diagram-note','explanation','technical','project-data','legend','counts','top','gid','card','links','status','node-gid','role','metrics','evidence','coverage','incoming','outgoing','in-caption','out-caption','search'];
const elements=Object.fromEntries(ids.map(id=>[id,new Element('div')]));
const html=fs.readFileSync(require('path').join(__dirname,'out/analyst.html'),'utf8');
const payload=JSON.parse(html.match(/<script id="project-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
elements['project-data'].textContent=JSON.stringify(payload);
const context={document:{getElementById:id=>elements[id],createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag),createTextNode:text=>{const e=new Element('#text');e.textContent=text;return e;}},Intl,Map,Number,String,Object,console};
vm.createContext(context);vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
assert.equal(elements.top.children.length,25);
for(const gid of ['100000000661912100','100000008346837100','100000002224132100']){
 elements.gid.value=gid;elements.search.listeners.submit({preventDefault(){}});
 assert.equal(elements['node-gid'].textContent,gid);assert.equal(elements.card.hidden,false);
 const node=payload.nodes.find(n=>n.gid===gid);assert.equal(elements.evidence.textContent,node.evidence);
 for(const [table,column] of [['incoming','dst'],['outgoing','src']]){
 const expected=payload.edges.filter(e=>e[column]===gid).sort((a,b)=>b.sum_kzt-a.sum_kzt);
 assert.equal(elements[table].children.length,expected.length);
 expected.forEach((e,i)=>{
 const row=elements[table].children[i];assert.equal(row.children[0].textContent,e.src+' → '+e.dst);assert.equal(row.children[2].textContent,String(e.n_tx));
 });
 }
 console.log('PASS form search and rendered directions:',gid);
}
elements.top.children[0].children[0].children[1].listeners.click();assert.equal(elements['node-gid'].textContent,payload.top[0].gid);
elements.outgoing.children[0].children[0].children[2].listeners.click();assert.equal(elements.card.hidden,false);
elements.gid.value='100000008346837101';elements.search.listeners.submit({preventDefault(){}});assert.equal(elements.card.hidden,true);assert.match(elements.status.textContent,/не найден/);
const isolated=payload.nodes.find(n=>n.in_deg==='0'&&n.out_deg==='0');context.selectNode(isolated.gid);assert.match(elements.incoming.textContent,/Нет связей/);assert.match(elements.outgoing.textContent,/Нет связей/);
console.log('PASS top navigation, edge navigation, unknown exact gid, isolated node. DOM unit checks (not a browser).');

context.selectNode('100000000661912100');
assert.match(elements.metrics.children[0].children[1].textContent,/^0,\d{3}$/);
assert.equal(elements.technical.open,false);
assert.match(elements.explanation.textContent,/0,00136/);
assert.match(elements.metrics.textContent,/не вероятность нарушения/);
assert.match(elements.metrics.textContent,/393.036 ₸/);
const cut=payload.nodes.find(n=>n.depth==='4'&&n.out_deg==='0');context.selectNode(cut.gid);
assert.match(elements.explanation.textContent,/наблюдение обрывается/);
assert.match(elements.role.textContent,/Нет наблюдаемого выхода/);
console.log('PASS Russian explanations, 3 decimals, money units, collapsed evidence, depth boundary');

context.selectNode('100000000661912100');
const svg=elements.diagram.children[0];
assert.equal(svg.tagName,'svg');
assert.equal(svg.children.filter(e=>e.tagName==='line').length,7);
assert.equal(svg.children.filter(e=>e.tagName==='g').length,8);
const firstNeighbor=svg.children.find(e=>e.tagName==='g');
const clicked=firstNeighbor.children[1].textContent;firstNeighbor.listeners.click();
assert.equal(elements['node-gid'].textContent,clicked);
console.log('PASS SVG edge count, neighbor labels and click navigation');
