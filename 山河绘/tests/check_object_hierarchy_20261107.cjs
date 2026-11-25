const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/object_hierarchy_20261107';
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
(async()=>{
 const p=read('G03_01/plan.json'),old=JSON.parse(fs.readFileSync(p.parent,'utf8')),url=read('server.json').url;
 assert.equal(hash(p.parent),p.parent_sha256);assert.equal(p.curves.length,489);assert.equal(p.curves.length,old.curves.length);
 for(let i=0;i<p.curves.length;i++){
  assert.deepEqual(p.curves[i].points,old.curves[i].points);assert.deepEqual(p.curves[i].source_curve_ids,old.curves[i].source_curve_ids);
 }
 assert.equal(new Set(p.actions.map(a=>a.curve)).size,489);assert.equal(p.actions.length,489);
 const completed=new Set();let current=null,phase=-1,painted=new Set();
 for(const a of p.actions){
  if(a.module!==current){
   if(current!==null){assert(p.curves.every((r,i)=>r.module!==current||painted.has(i)));completed.add(current)}
   assert(!completed.has(a.module));current=a.module;phase=-1;
  }
  assert(a.phase>=phase);phase=a.phase;painted.add(a.curve);
 }
 const tree=p.modules.find(m=>m.name.startsWith('松树整体')),house=p.modules.find(m=>m.name==='亭子整体');
 assert.equal(tree.planned_strokes,155);assert.equal(house.planned_strokes,44);
 assert.equal(p.actions.filter(a=>a.module===tree.id).length,155);
 assert.equal(p.stats.hierarchy_order.module_reentries,0);assert(p.stats.baseline_order.module_reentries>0);
 assert.equal(p.automatic_semantic_generalization,false);assert.equal(p.ready_for_robot,false);assert.equal(read('execution_binding.json').journal,null);
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{headless:true,
  executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',viewport:{width:1540,height:1120},
  args:['--disable-background-networking','--no-first-run']});
 try{
  const page=await ctx.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(url);await page.waitForFunction(()=>plan&&sourceImage.naturalWidth>0);
  assert((await page.locator('#certainty').textContent()).includes('本幅归属由助手校正'));
  await page.locator('#play').click();await page.waitForFunction(()=>cursor>1);await page.locator('#play').click();
  const next=p.actions.findIndex(a=>a.module!==p.actions[0].module);
  await page.locator('#next').click();assert.equal(Math.floor(await page.evaluate(()=>cursor)),next);
  const treeStart=p.actions.findIndex(a=>a.module===tree.id),treePhase=p.actions.findIndex(a=>a.module===tree.id&&a.phase===1);
  await page.evaluate(value=>{cursor=value;playing=false;render()},treeStart+.5);
  assert((await page.locator('#currentModule').textContent()).includes('松树整体'));
  await page.screenshot({path:path.join(root,'tree_start_preview.png'),fullPage:true});
  await page.evaluate(value=>{cursor=value;render()},treePhase+5.5);
  await page.screenshot({path:path.join(root,'platform_preview.png'),fullPage:true});
  await page.locator('#next').click();assert.equal(Math.floor(await page.evaluate(()=>cursor)),treeStart+155);
  await page.locator('#live').click();await page.waitForFunction(()=>document.getElementById('feed').textContent.includes('尚未绑定'));
  assert.equal(await page.locator('#percent').textContent(),'0%');assert(await page.locator('#play').isDisabled());
  await page.goto(url+'report.html');await page.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(root,'verification.json'),JSON.stringify({passed:true,curves:489,exact_coordinates_preserved:true,
   whole_object_completion_barrier:true,phases_monotonic_within_object:true,tree_strokes:155,pavilion_strokes:44,
   previous_route_reentries_under_same_objects:p.stats.baseline_order.module_reentries,new_route_reentries:0,
   scene_specific_assistant_review:true,semantic_generalization_established:false,
   uncertain_boundary_strokes:p.stats.unresolved_regions,browser_errors:errors,robot_bound:false},null,2));
  console.log('PASS: all 489 coordinates unchanged; complete-object barrier; tree/pavilion grouping; phase order; browser playback');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
