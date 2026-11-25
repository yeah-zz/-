const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/structural_roles_20261107';
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
(async()=>{
 const p=read('G03_01/plan.json'),old=JSON.parse(fs.readFileSync(p.parent,'utf8')),url=read('server.json').url;
 assert.equal(p.curves.length,489);
 for(let i=0;i<489;i++){assert.deepEqual(p.curves[i].points,old.curves[i].points);assert.equal(p.curves[i].module,old.curves[i].module)}
 let checked=0;
 for(const m of p.modules){
  const actions=p.actions.filter(a=>a.module===m.id);let phase=-1;
  for(const a of actions){assert(a.phase>=phase);phase=a.phase;assert.equal(a.phase,p.curves[a.curve].phase)}
  for(let layer=0;layer<2;layer++){
   const firstLater=actions.findIndex(a=>a.phase>layer);
   if(firstLater>=0)assert(actions.slice(firstLater).every(a=>a.phase>layer));
  }
  checked++;
 }
 assert.equal(new Set(p.actions.map(a=>a.curve)).size,489);
 assert.equal(p.ready_for_robot,false);assert.equal(p.new_training,false);
 assert.equal(read('execution_binding.json').journal,null);
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{headless:true,
   executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',viewport:{width:1540,height:1120},args:['--disable-background-networking','--no-first-run']});
 try{
  const page=await ctx.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(url);await page.waitForFunction(()=>plan&&sourceImage.naturalWidth>0);
  assert(await page.locator('#byrole').isChecked());
  const rock=p.actions.findIndex(a=>a.module===2&&a.phase===1),tree=p.actions.findIndex(a=>a.module===8&&a.phase===2);
  await page.evaluate(c=>{cursor=c;render()},rock+.5);
  assert((await page.locator('#phases').textContent()).includes('皴线'));
  assert((await page.locator('#currentSource').textContent()).includes('皴线'));
  await page.screenshot({path:path.join(root,'rock_stage_preview.png'),fullPage:true});
  await page.evaluate(c=>{cursor=c;render()},tree+.5);
  assert((await page.locator('#phases').textContent()).includes('主枝干'));
  assert((await page.locator('#currentSource').textContent()).includes('枝叶'));
  await page.screenshot({path:path.join(root,'tree_stage_preview.png'),fullPage:true});
  await page.locator('#byrole').uncheck();await page.locator('#byrole').check();
  await page.locator('#play').click();const before=await page.evaluate(()=>cursor);await page.waitForFunction(c=>cursor>c,before);await page.locator('#play').click();
  await page.goto(url+'report.html');await page.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  assert.deepEqual(errors,[]);assert.equal((await page.request.post(url+'api/move',{data:{}})).status(),405);
  fs.writeFileSync(path.join(root,'verification.json'),JSON.stringify({passed:true,coordinates_and_objects_unchanged:true,
   object_stage_barriers_checked:checked,curves:489,browser_errors:errors,semantic_accuracy_verified:false,
   role_inference_is_rule_based:true,robot_connected:false},null,2));
  console.log('PASS: 489 unchanged paths, 13 object-stage barriers, stage labels and color mode, no robot binding');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
