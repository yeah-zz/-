const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/module_stroke_planner_20261104';
(async()=>{
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{
  headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  viewport:{width:1700,height:1250},args:['--disable-background-networking','--no-first-run']});
 try{
  const page=await ctx.newPage(),errors=[],network=[];
  page.on('pageerror',e=>errors.push(String(e)));
  page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
  const results=[];
  for(const id of ['G03_01','G05_03','T02']){
   await page.goto('file:///'+path.join(root,'outputs',id,'index.html').replaceAll('\\','/'));
   await page.waitForFunction(()=>document.getElementById('status').textContent.includes('新规划'));
   assert.equal(await page.locator('canvas').count(),2);
   await page.waitForFunction(()=>document.getElementById('reference').complete);
   assert(await page.locator('#reference').evaluate(e=>e.naturalWidth>0));
   await page.locator('#toggle').click();
   await page.waitForFunction(()=>document.getElementById('reference').complete&&document.getElementById('reference').src.includes('module_candidates'));
   await page.locator('#ghost').uncheck();
   await page.locator('#progress').evaluate(e=>{e.value=200;e.dispatchEvent(new Event('input'))});
   const before=await page.locator('#progress').inputValue();
   await page.locator('#play').click();
   await page.waitForFunction(v=>+document.getElementById('progress').value>+v,before);
   await page.locator('#play').click();
   await page.locator('#progress').evaluate(e=>{e.value=200;e.dispatchEvent(new Event('input'))});
   await page.screenshot({path:path.join(root,'outputs',id,'viewer_preview.png'),fullPage:true});
   await page.locator('#progress').evaluate(e=>{e.value=1000;e.dispatchEvent(new Event('input'))});
   const status=await page.locator('#status').textContent();
   assert(status.includes('100%')===false); // status uses exact curve counts, percentage has own element
   assert.equal(await page.locator('#percent').textContent(),'100%');
   const geometry=JSON.parse(fs.readFileSync(path.join(root,'outputs',id,'offline_plan.json'),'utf8'));
   assert.equal(new Set(geometry.actions.map(r=>r.curve)).size,geometry.curves.length);
   assert.equal(geometry.ready_for_robot,false);
   let seen=new Set(),last=null;
   for(const action of geometry.actions){if(action.module!==last){assert(!seen.has(action.module));seen.add(action.module);last=action.module}}
   results.push({id,passed:true,all_curves_once:true,module_reentry:false,status});
  }
  assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
  fs.writeFileSync(path.join(root,'browser_verification.json'),JSON.stringify({passed:true,results,page_errors:errors,
   external_network_requests:network,hardware_access:false},null,2));
  console.log('PASS: all three offline viewers, image toggle/play/seek/full completion; no external requests.');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
