const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/autonomous_full_image_screened_20261105';
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
(async()=>{
 const url=read('server.json').url,catalog=read('catalog.json');
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{
   executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,
   viewport:{width:1540,height:1120},args:['--disable-background-networking','--no-first-run']});
 try{
  const page=await ctx.newPage(),errors=[],external=[];
  page.on('pageerror',e=>errors.push(String(e)));
  page.on('request',r=>{if(/^https?:/.test(r.url())&&!r.url().startsWith(url))external.push(r.url())});
  await page.goto(url);await page.waitForFunction(()=>plan&&scene==='G03_01_500'&&sourceImage.naturalWidth>0);
  assert((await page.locator('#scene option').count())===18);
  await page.locator('#ghost').uncheck();await page.locator('#play').click();await page.waitForFunction(()=>cursor>1);
  await page.locator('#play').click();const stopped=await page.evaluate(()=>cursor);await page.waitForTimeout(200);
  assert.equal(await page.evaluate(()=>cursor),stopped);
  await page.locator('#next').click();assert(await page.evaluate(()=>cursor)>stopped);
  await page.locator('#seek').evaluate(e=>{e.value=180;e.dispatchEvent(new Event('input'))});
  await page.screenshot({path:path.join(root,'platform_preview.png'),fullPage:true});
  await page.locator('#seek').evaluate(e=>{e.value=1000;e.dispatchEvent(new Event('input'))});
  assert.equal(await page.locator('#percent').textContent(),'100%');
  await page.locator('#live').click();await page.waitForFunction(()=>document.getElementById('feed').textContent.includes('尚未绑定'));
  assert(await page.locator('#play').isDisabled());assert.equal(await page.locator('#percent').textContent(),'0%');
  await page.locator('#offline').click();
  for(const id of ['G03_01_700','G05_03_500','T02_500','G01_01_500','G02_02_500','G04_02_500']){
    await page.selectOption('#scene',id);await page.waitForFunction(id=>scene===id&&sourceImage.naturalWidth>0,id);
    assert.equal(await page.locator('#failure').textContent(),'');
  }
  await page.goto(url+'report.html');await page.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  assert.equal(await page.locator('img').count(),6);
  await page.screenshot({path:path.join(root,'report_preview.png'),fullPage:false});
  const forbidden=await page.request.post(url+'api/move',{data:{}});assert.equal(forbidden.status(),405);
  assert.equal((await page.request.get(url+'source_snapshot/screen_full_image_ownership_20261105.py')).status(),404);
  assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  fs.writeFileSync(path.join(root,'platform_verification.json'),JSON.stringify({passed:true,catalog_plans:catalog.scenes.length,
    report_images_loaded:6,play_pause_seek_module_jump:true,execution_unbound:true,no_motion_endpoint:true,
    browser_errors:errors,external_requests:external,artistic_accuracy_established:false},null,2));
  console.log('PASS: full-image report, 18 catalog plans, playback, module jump, unbound execution and read-only routes');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
