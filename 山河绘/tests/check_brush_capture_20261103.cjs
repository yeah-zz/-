// Offline browser smoke/interaction test. No real measurement is written.
const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/brush_execution_model_20261103';
(async()=>{
 const qa=path.join(base,'qa');fs.mkdirSync(qa,{recursive:true});
 const ctx=await chromium.launchPersistentContext(path.join(qa,'browser_profile'),{
  executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,
  viewport:{width:1280,height:1000},args:['--disable-background-networking','--no-first-run']});
 try {
 const page=await ctx.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto('file:///'+base+'/capture.html');
 await page.locator('#plan').setInputFiles(path.join(base,'capture','capture_plan.json'));
 assert.equal(await page.locator('#sample option').count(),144);
 await page.locator('#actual_press_mm').fill('0.5');
 await page.locator('#actual_speed_mm_s').fill('15');
 await page.locator('#photo').setInputFiles(path.join(base,'outputs','G03_01','footprint_0.6.jpg'));
 await page.waitForFunction(()=>document.getElementById('canvas').width===1000);
 await page.locator('#canvas').click({position:{x:100,y:100}});
 await page.locator('#canvas').click({position:{x:200,y:100}});
 await page.locator('#mode').selectOption('middle');
 await page.locator('#canvas').click({position:{x:100,y:100}});
 await page.locator('#canvas').click({position:{x:100,y:120}});
 await page.locator('#quality').selectOption('uncertain');
 await page.locator('#notes').fill('QA ONLY - NOT A PHYSICAL SAMPLE');
 await page.locator('#commit').click();
 await page.waitForFunction(()=>document.getElementById('output').value.includes('middle_width_mm'));
 const data=JSON.parse(await page.locator('#output').inputValue()),r=data.records.A_01_1;
 assert.equal(r.actual_press_mm,.5);assert.equal(r.actual_speed_mm_s,15);
 console.log('QA measurement',JSON.stringify({width:r.middle_width_mm,scale:r.mm_per_pixel,points:r.measurement_points}));
 assert(Math.abs(r.middle_width_mm-2)<.02);assert.equal(r.ready_for_robot,false);
 assert(r.photo_sha256.length===64);
 await page.locator('#known').fill('25');
 await page.locator('#sample').selectOption('B_01_1');
 await page.locator('#sample').selectOption('A_01_1');
 assert.equal(await page.locator('#known').inputValue(),'10');
 await page.locator('#commit').click();
 assert(Math.abs(JSON.parse(await page.locator('#output').inputValue()).records.A_01_1.middle_width_mm-2)<.02);
 await page.screenshot({path:path.join(qa,'capture_page.png'),fullPage:true});
 await page.goto('file:///'+base+'/index.html');
 assert.equal(await page.locator('table tr').count(),54);
 await page.screenshot({path:path.join(qa,'report_top.png'),fullPage:false});
 assert.deepEqual(errors,[]);
 fs.writeFileSync(path.join(qa,'browser_check.json'),JSON.stringify({passed:true,plan_options:144,case_rows:53,
  measured_sample_records_written:0,tests:['plan load','photo load','scale/width clicks','null parameters',
   'record serialization','photo hash','restore ruler scale','report links/table','no page errors']},null,2));
 console.log('Offline browser checks passed; no physical measurement records saved.');
 } finally {await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
