const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/complete_G03_01_trial_20261103/live_viewer';
(async()=>{
 const info=JSON.parse(fs.readFileSync(path.join(root,'server.json'),'utf8'));
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{headless:true,
  executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',viewport:{width:1450,height:1100},
  args:['--disable-background-networking','--no-first-run']});
 try{
  const page=await ctx.newPage(),errors=[],writes=[];page.on('pageerror',e=>errors.push(String(e)));
  page.on('request',r=>{if(r.method()!=='GET')writes.push(r.url())});
  await page.goto(info.url);await page.waitForFunction(()=>document.getElementById('progress').textContent.includes('1184'));
  const initial=await page.locator('#progress').textContent();
  assert((await page.locator('#painting').evaluate(c=>c.width))>0);
  await page.locator('#play').click();await page.waitForFunction(()=>document.getElementById('replayNotice').style.display==='block');
  await page.locator('#stop').click();
  await page.locator('#timeline').evaluate(e=>{e.value=50;e.dispatchEvent(new Event('input',{bubbles:true}))});
  await page.waitForFunction(()=>document.getElementById('stepCounter').textContent.includes('50 / 1184'));
  await page.locator('#showCurrent').click();
  assert((await page.locator('#viewTitle').textContent()).includes('回放'));
  await page.locator('#zoom').evaluate(e=>{e.value=2;e.dispatchEvent(new Event('input',{bubbles:true}))});
  await page.locator('#resetView').click();assert.equal(await page.locator('#zoom').inputValue(),'1');
  await page.locator('#live').click();await page.waitForFunction(()=>document.getElementById('viewTitle').textContent.includes('跟随现场'));
  await page.waitForFunction(()=>document.getElementById('connection').textContent==='现场绘画中');
  await page.screenshot({path:path.join(root,'live_page.png'),fullPage:true});
  assert.deepEqual(errors,[]);assert.deepEqual(writes,[]);
  fs.writeFileSync(path.join(root,'browser_verification.json'),JSON.stringify({passed:true,initial_progress:initial,
   final_progress:await page.locator('#progress').textContent(),read_only_browser_requests:true,
   tests:['live data','canvas render','replay','stop replay','seek','current stroke explanation','zoom/reset','return to live','no page errors'],
   screenshot:'live_page.png'},null,2));
  console.log('Live viewer browser checks passed; all requests GET; no robot commands.');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
