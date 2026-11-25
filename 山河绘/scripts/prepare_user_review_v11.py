# -*- coding: utf-8 -*-
"""Prepare a small target-domain annotation package; never invent user labels."""
import sys,json,hashlib,shutil
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'user_review_v11_20261024'
PACKAGE_ID='qianli_user_review_v11_20261024'

def build():
    ROOT.mkdir(exist_ok=True)
    for name in ('samples','overlays','labels','backups','exports','logs','browser_profile','cache','selftest'):(ROOT/name).mkdir(exist_ok=True)
    specs=[
      ('A01','user_qianli_02',(140,450,640,950),'山峰轮廓与皴线','先描 3 条确定的轮廓，再描 3 条皴线；不要把山顶苔点串成轮廓。'),
      ('A02','user_qianli_05',(20,30,520,530),'青绿山体内部','区分真正的皴线与蓝绿色块交界；粗细不确定可以不填。'),
      ('A03','user_qianli_07',(510,120,1010,620),'组合山石与遮挡','先看清哪些线属于前山、哪些属于后山，避免把两笔接在一起。'),
      ('A04','user_qianli_07',(10,15,510,515),'绢纹与背景杂线','主要圈出确实不应画的绢纹 / 空白区域；避开淡山和真实笔画。'),
      ('A05','user_qianli_07',(260,560,760,1060),'前景组合山石','检查一条皴线应在哪里开始、在哪里结束；暂不追求全图标完。'),
      ('A06','user_qianli_07',(930,550,1430,1050),'山石、屋宇与树木','第二批再做；先区分对象，不要把屋檐、树枝与皴线混在一起。')]
    records=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    board=Image.new('RGB',(1440,1140),'#f5f2e9');d=ImageDraw.Draw(board);d.text((20,10),'第一批人工复核：先做 A01–A03；下排留到下一步',font=font,fill='#243d33')
    d.text((20,43),'每张先确认 3 条轮廓、3 条皴线、2 处明确背景；不确定就留空。以下图片均未经用户标注。',font=small,fill='#53655b')
    for i,(identity,source_id,box,title,focus) in enumerate(specs):
        source=LAB/'data'/'processed'/source_id/'image.png';original=Image.open(source).convert('RGB')
        if not (0<=box[0]<box[2]<=original.width and 0<=box[1]<box[3]<=original.height):raise ValueError('invalid crop '+identity)
        image=original.crop(box);target=ROOT/'samples'/(identity+'.png');image.save(target)
        overlay_source=LAB/'full_landscape_probe_20261024'/source_id/'overlay.png';overlay=overlay_source.exists()
        if overlay:Image.open(overlay_source).convert('RGB').crop(box).save(ROOT/'overlays'/(identity+'.png'))
        records.append(dict(id=identity,title=title,focus=focus,image='samples/'+identity+'.png',image_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
          width=image.width,height=image.height,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_id=source_id,source_group='wang_ximeng_qianli_jiangshan',
          source_box=list(box),split='train_calibration',overlay=overlay,early_reference=True,independent_test=False,annotation_status='awaiting_user'))
        x=(i%3)*480+20;y=(i//3)*525+85;d.text((x,y),identity+'  '+title,font=font,fill='#243d33');preview=image.copy();preview.thumbnail((440,440));board.paste(preview,(x,y+37))
    board.save(ROOT/'first_batch_overview.jpg',quality=94)
    (ROOT/'package_manifest.json').write_text(json.dumps(dict(package_id=PACKAGE_ID,schema='shanhehui.partial_strokes.v11',samples=records,
      training_status='waiting_for_user_labels',user_labels_created=0,all_samples_same_work=True,unlabeled='unknown',dense_gold=False,robot_connected=False),ensure_ascii=False,indent=2),encoding='utf-8')
    source_root=Path(__file__).resolve().parents[1]
    shutil.copy2(source_root/'gongbi_training'/'review_server_v11.py',ROOT/'review_app.py')
    shutil.copy2(source_root/'gongbi_training'/'review_v11.html',ROOT/'review.html')
    shutil.copy2(source_root/'scripts'/'compile_user_review_v11.py',ROOT/'compile_labels.py')
    launcher=r'''$ErrorActionPreference='Stop'
$taskRoot=$PSScriptRoot
$env:TEMP=Join-Path $taskRoot 'temp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR=Join-Path $taskRoot 'cache\pip'
$env:TORCH_HOME=Join-Path $taskRoot 'cache\torch'
$env:HF_HOME=Join-Path $taskRoot 'cache\huggingface'
$env:XDG_CACHE_HOME=Join-Path $taskRoot 'cache'
$env:CUDA_CACHE_PATH=Join-Path $taskRoot 'cache\cuda'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
$python='E:\Anaconda\python.exe'
$address='http://127.0.0.1:8779'
$ready=$false
try { $health=Invoke-RestMethod "$address/api/health" -TimeoutSec 2; if ($health.package_id -ne 'qianli_user_review_v11_20261024') { throw 'Port 8779 is used by another application.' }; $ready=$true } catch { }
if (-not $ready) {
  $listener=Get-NetTCPConnection -LocalPort 8779 -State Listen -ErrorAction SilentlyContinue
  if ($listener) { throw 'Port 8779 is occupied. Do not stop the unrelated process.' }
  $process=Start-Process -FilePath $python -ArgumentList @('-B',('"{0}"' -f (Join-Path $taskRoot 'review_app.py')),'--root',('"{0}"' -f $taskRoot),'--port','8779') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRoot 'logs\server.out.log') -RedirectStandardError (Join-Path $taskRoot 'logs\server.err.log')
  $process.Id | Set-Content -LiteralPath (Join-Path $taskRoot 'logs\server.pid')
  for ($attempt=0;$attempt -lt 40;$attempt++) {
    Start-Sleep -Milliseconds 250
    try { $health=Invoke-RestMethod "$address/api/health" -TimeoutSec 1; if ($health.package_id -eq 'qianli_user_review_v11_20261024') { $ready=$true;break } } catch { }
    if ($process.HasExited) { throw 'Review server exited. See logs\server.err.log.' }
  }
}
if (-not $ready) { throw 'Review server did not become ready.' }
$edge='C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
if (-not (Test-Path -LiteralPath $edge)) { $edge='C:\Program Files\Microsoft\Edge\Application\msedge.exe' }
if (-not (Test-Path -LiteralPath $edge)) { Write-Host "Open $address using a browser profile stored on E:.";exit }
$profile=Join-Path $taskRoot 'browser_profile'
$edgeCache=Join-Path $taskRoot 'cache\edge'
Start-Process -FilePath $edge -ArgumentList @("--user-data-dir=$profile","--disk-cache-dir=$edgeCache",'--no-first-run','--no-default-browser-check',"--app=$address") -WindowStyle Normal
'''
    (ROOT/'start_review.ps1').write_text(launcher,encoding='utf-8-sig')
    (ROOT/'start_review.cmd').write_text('@echo off\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_review.ps1"\r\nif errorlevel 1 pause\r\n',encoding='ascii')
    guide='''# 第一批人工复核怎么做

双击同目录 start_review.cmd。工具离线运行，网页、浏览器独立配置、缓存、标注都保存在 E 盘；不会连接机械臂。

先只做 A01、A02、A03。每张先做 3 条确定的轮廓、3 条皴线、2 个明确背景区域，用来校准标准；不需要描完整张图。

1. 选择“山石轮廓 / 结构线”，沿一条线的中间逐点点击，转弯处补点；Enter 或“完成本笔”结束。粗线只画一条中心线，不描两侧边缘。橙色起点只代表描线起点，除非勾选确认，不作为真实落笔方向。
2. 选择“皴线”，同样描出几条你确定应该保留的内部笔画。能判断皴法才选名字，拿不准就保留“不确定”。粗细、浓淡可选起/中/末段的相对变化，不确定可以全部留空；这不代表机械臂实际下压力或墨液浓度。
3. 选择“圈出明确背景 / 杂纹”，逐点围住纸绢纹、污渍或纯色交界中的杂线，Enter 闭合。避开所有真实墨线和苔点，包括浅淡笔画。拿不准的区域用“不确定”标记。
4. 按住“看原图”检查位置；点击“本张已检查”，标签会直接保存到 E 盘。切换图片也会先保存已完成的修改；未完成的一笔要先完成或取消。
5. 做完前三张后点击“导出备份到 E 盘”。回到聊天告诉我“A01–A03 已保存”，我会读取 labels 文件夹，不用上传文件。关闭前可点“结束复核并关闭服务”。

本批是同一幅《千里江山图》的局部训练校准，不能当成独立测试图。未标注区域始终未知，不会自动作为背景。模型红色叠加仅用于观察问题，不会自动成为你的标注。

确认标签后会先检查描线是否落在正确位置、有无正负冲突，再组建有来源分组的训练/验证集。不会仅凭三张少量标签宣称已训练出可靠模型；新的独立画作会另行准备并复核。
'''
    (ROOT/'HOW_TO_REVIEW.md').write_text(guide,encoding='utf-8')
    print('REVIEW_PACKAGE',ROOT,flush=True)
if __name__=='__main__':build()
