const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/camera_preview_20261104';
(async()=>{
 const session=JSON.parse(fs.readFileSync(path.join(root,'active_preview.json'),'utf8'));
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{
  headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  viewport:{width:1300,height:1050},args:['--disable-background-networking','--no-first-run']});
 try{
  const page=await ctx.newPage(),errors=[];
  page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(session.url);
  await page.waitForFunction(()=>document.getElementById('state').textContent.includes('画面正常'),{timeout:15000});
  const pixels=await page.locator('#frame').evaluate(im=>{
   const c=document.createElement('canvas');c.width=im.naturalWidth;c.height=im.naturalHeight;
   c.getContext('2d').drawImage(im,0,0);const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
   let sum=0;for(let i=0;i<d.length;i+=4)sum+=(d[i]+d[i+1]+d[i+2])/3;
   return {width:c.width,height:c.height,mean:sum/(c.width*c.height)};
  });
  assert(pixels.width>0&&pixels.mean>5,'Browser image is black or unavailable');
  const status=await (await page.request.get(session.url+'status')).json();
  assert.equal(status.black_frame,false);assert(status.frame_age_s<3);
  await page.screenshot({path:path.join(session.folder,'browser_preview.png'),fullPage:true});
  assert.deepEqual(errors,[]);
  const result={passed:true,pixels,state:await page.locator('#state').textContent(),errors,
    actual_browser_pixels_verified:true,no_stop_or_robot_command_sent:true};
  fs.writeFileSync(path.join(session.folder,'browser_verification.json'),JSON.stringify(result,null,2));
  console.log(JSON.stringify(result));
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
