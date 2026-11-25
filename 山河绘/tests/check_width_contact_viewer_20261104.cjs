const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/width_aware_contact_20261104';
(async()=>{
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{
  headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  viewport:{width:1550,height:1100},args:['--disable-background-networking','--no-first-run']});
 try{
  const page=await ctx.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto('file:///'+root+'/index.html');
  const rows=[];
  for(const id of ['G03_01','G05_03','T02'])for(const w of ['0.6','1','2.04']){
   await page.locator('#scene').selectOption(id);await page.locator('#width').selectOption(w);
   await page.waitForFunction(()=>{const a=document.getElementById('board');return a.complete&&a.naturalWidth>0});
   const key='w'+(Number.isInteger(+w)?(+w).toFixed(1):w).replace('.','p');
   const source=await page.locator('#board').getAttribute('src');assert(source.includes(id+'/'+key+'/'));
   const archive=JSON.parse(fs.readFileSync(path.join(root,id,'complete_paths.json'),'utf8'));
   const candidate=JSON.parse(fs.readFileSync(path.join(root,id,key,'drawing_candidate.json'),'utf8'));
   const stats=JSON.parse(fs.readFileSync(path.join(root,id,key,'statistics.json'),'utf8'));
   const all=candidate.selected_curve_ids.concat(candidate.deferred_curve_ids).sort((a,b)=>a-b);
   assert.deepEqual(all,archive.paths.map((_,i)=>i));assert.equal(candidate.ready_for_robot,false);
   assert(stats.measured_on_geometry_max_deviation_mm<=.0800001);assert.equal(stats.endpoints_preserved,true);
   assert(candidate.actual_compression_mm===null);assert.equal(candidate.contact_envelope_is_unmeasured_design,true);
   if(id==='G03_01')await page.waitForFunction(()=>['pavilion','pine','rock'].every(i=>{const a=document.getElementById(i);return a.complete&&a.naturalWidth>0}));
   await page.locator('#deferred').click();await page.waitForFunction(()=>{const a=document.getElementById('board');return a.complete&&a.naturalWidth>0});
   assert((await page.locator('#board').getAttribute('src')).includes('deferred_overlay'));
   await page.locator('#deferred').click();rows.push({id,width:w,passed:true});
  }
  await page.locator('#scene').selectOption('G03_01');await page.locator('#width').selectOption('1');
  await page.waitForFunction(()=>document.getElementById('board').complete&&document.getElementById('board').naturalWidth>0);
  await page.screenshot({path:path.join(root,'browser_preview.png'),fullPage:true});
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(root,'browser_verification.json'),JSON.stringify({passed:true,rows,errors,
   full_archive_partition_verified:true,geometry_deviation_verified:true,robot_commands:false},null,2));
  console.log('PASS: 9 scenario pages, local images, deferred layer, complete archive partition, geometry bounds.');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
