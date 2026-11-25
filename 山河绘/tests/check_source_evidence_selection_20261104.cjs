const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/source_evidence_selection_20261104';
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
(async()=>{
 const ref=read('reference_regression.json'),summary=read('results.json');
 assert.equal(ref.preservation_regression_passed,true);assert.equal(ref.user_positive_curves,ref.user_positive_kept);
 assert.equal(summary.strong_only_variant_rejected,true);assert(summary.real_reference_curves_in_uncertain_layer>0);
 const ids=['G03_01','G05_03','T02'];
 for(const id of ids){
  const d=read(id+'/candidates.json'),a=d.selection,stats=d.statistics;
  assert.deepEqual([...a.selected,...a.removed].sort((a,b)=>a-b),d.input_curves.map((_,i)=>i));
  assert.equal(d.physical_pressure_mm,null);assert.equal(d.ready_for_robot,false);
  assert.equal(stats.geometry_of_selected_curves_unchanged,true);assert.equal(stats.guarded_candidates_rejected_as_noise,0);
  assert(a.decisions.filter(x=>x.disposition==='uncertain').every(x=>a.selected.includes(x.curve)));
  assert(a.duplicates.every(x=>a.selected.includes(x.representative)));
  assert.equal(sha(d.source),d.source_sha256);for(const p of d.parents)assert.equal(sha(p.path),p.sha256);
  const original=JSON.parse(fs.readFileSync(d.parents[0].path,'utf8'));
  if(id!=='T02')assert.deepEqual(d.input_curves.map(x=>x.points),original.curves.map(x=>x.points));
  assert(d.input_curves.every(x=>x.points.flat().every(Number.isFinite)));
 }
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{
  headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  viewport:{width:1450,height:1000},args:['--disable-background-networking','--no-first-run']});
 try{
  const p=await ctx.newPage(),errors=[];p.on('pageerror',e=>errors.push(String(e)));
  await p.goto('file:///'+root+'/index.html');
  let checks=0;
  for(const id of ids)for(const mode of ['comparison.jpg','before.png','selected.png','removed_overlay.png','source_removed.png','pending_overlay.png']){
   await p.locator('#scene').selectOption(id);await p.locator('#mode').selectOption(mode);
   await p.waitForFunction(()=>{const a=document.getElementById('board');return a.complete&&a.naturalWidth>0});
   assert((await p.locator('#board').getAttribute('src')).includes(id+'/full/'+mode));checks++;
  }
  assert(!(await p.locator('#mode').innerText()).includes('strong_only'));
  await p.locator('#scene').selectOption('G03_01');await p.locator('#mode').selectOption('removed_overlay.png');
  await p.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  await p.screenshot({path:path.join(root,'browser_report.png'),fullPage:true});
  const links=await p.locator('a').evaluateAll(a=>a.map(x=>x.getAttribute('href')));
  for(const href of links)assert(fs.existsSync(path.join(root,href)));
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(root,'verification.json'),JSON.stringify({passed:true,scenarios:ids,viewer_checks:checks,
   reviewed_positive_curves:ref.user_positive_curves,reviewed_positive_kept:ref.user_positive_kept,
   uncertain_real_reference_lines_preserved:summary.real_reference_curves_in_uncertain_layer,
   strong_only_variant_not_offered:true,source_archives_unchanged:true,partition_complete:true,
   semantic_cun_accuracy_established:false,physical_improvement_verified:false,errors},null,2));
  console.log('PASS: source archives, uncertainty retention, reference preservation and 18 viewer states.');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
