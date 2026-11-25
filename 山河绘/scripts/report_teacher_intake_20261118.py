# -*- coding: utf-8 -*-
import sys,json,hashlib,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from gongbi_training.stage_reference_policy import permitted_supervision
from PIL import Image
ROOT=LAB/'independent_teacher_intake_20261118';PROJECT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')

def main():
    raw=read(ROOT/'su_stages_raw.json');assert len(raw['stages'])==6
    images=[im for s in raw['stages'] for im in s['images']];assert len(images)==7
    for im in images:assert sha(im['path'])==im['sha256']
    # Bounded exact-pixel dedup against the current named evidence and known input crops.
    known=[]
    for base in [LAB/'cunfa_role_reference_20261112/references',LAB/'object_prospective_20261118/raw',LAB/'cunfa_roles_v2_20261112/references']:
        for p in base.iterdir():
            if p.suffix.lower() in {'.png','.jpg','.jpeg'} and not any(k in p.name for k in ['overlay','target','review','comparison']):known.append(p)
    hashes={}
    for p in known:
        a=Image.open(p).convert('RGB');hashes.setdefault((a.size,hashlib.sha256(a.tobytes()).hexdigest()),[]).append(str(p))
    matches=[]
    for im in images:
        for old in hashes.get((tuple(im['size']),im['rgb_sha256']),[]):matches.append(dict(new=im['path'],old=old))
    evidence=dict(source_url=raw['url'],artist='苏春生',artist_attribution_checked=True,ordered_steps_checked=True,
        source_article_attribution='ECNU republication of Xinmin Evening News,2015-07-18',
        same_view_alignment_verified=False,style_evidence='explicit_mixed_named_object',
        style_components=['hemp','axe'],named_object='near foreground rocks',
        style_scope='article-named foreground rocks, not entire image or per-pixel labels',
        unique_work_reference=1,source_specific_order_not_universal_rule=True,
        training_usage='knowledge and coarse stage reference only; no classification or pixel gradients',
        scope_of_novelty_check='bounded exact pixel check; not full historical corpus, crop or repost independence certification',
        exact_pixel_comparison_files=len(known),exact_pixel_matches=matches,
        independent_validation=False,model_recognition_added=False)
    evidence['permitted_supervision']=permitted_supervision(evidence)
    write('source_admission.json',evidence)
    phases=[dict(stage=s['stage'],title=s['title'],preview='views/'+Path(s['images'][0]['path']).name,
        object_focus=o,phase_type=t,source_description=s['description'],
        source_provenance=raw['url'],single_strokes=None,parameters=None)
        for s,o,t in zip(raw['stages'],['foreground_rock','midground_rock','pine_trunk_and_needles','distant_rock_cloud_and_figures','shared_coloring','pine_needles_refinement'],
        ['outline_and_texture_mixed','rubbing_and_wash','trunk_and_foliage','background_and_details','color','foliage_refinement'])]
    summaries=['先处理前景山石；原文指出这里混合披麻、斧劈。','转向中景山体，以皴擦和淡墨表现层次。','描绘松干与松针，区分近景和中景的轻重。','补远山、云层、人物和点苔。','统筹赭石及青绿色，保留前后浓淡差异。','修整松针与整体画面后完成。']
    for step,summary in zip(phases,summaries):
        step['source_description']=[summary];step['description_is_assistant_paraphrase']=True
    graph=dict(steps=phases,edges=[[i,i+1] for i in range(1,6)],
        basis='human organized source teaching sequence; no individual stroke chronology verified',
        initial_outline_only_frame_available=False,automatic_planner=False,robot_connected=False)
    write('stage_reference.json',graph)
    access=read(ROOT/'normal_access_checks.json');assert all(r.get('access',{}).get('d',[None])[0]=='0' for r in access)
    write('intake_summary.json',dict(liu_xuewei=dict(page_status=403,media_acquired=False),wang_xingtang=dict(paid_episodes_checked=[6920,6917,9216],media_acquired=False),
        han_jingwei=dict(preexisting_shortlist=True,official_attribution_rechecked=True,no_new_video_acquired=True),
        su_chunsheng=dict(stages=6,images=7,artist_attribution_checked=True,single_strokes=0),no_access_bypass=True))
    lock=read(LAB/'local_rock_patch_20261117/baseline_lock.json');bc={k:sha(lock[k])==lock[k+'_sha256'] for k in ['approved500','working489']};assert all(bc.values())
    weights=read(LAB/'two_work_bands_20261118/checkpoint_freeze.json');assert all(sha(LAB/f'two_work_bands_20261118/{a}.pt')==h for a,h in weights.items())
    checks=dict(source_image_hashes_match=True,stages=6,source_images=7,artist_source_references=1,
        bounded_dedup_files=len(known),bounded_exact_matches=len(matches),full_work_independence_certified=False,
        mixed_style_not_binary_label=True,pixel_masks_not_inferred=True,different_crops_not_frame_differenced=True,
        source_steps_not_model_planning=True,access_limits_respected=True,policy_tests_passed=5,
        checkpoint_hashes_unchanged=True,baseline_hashes=bc,new_model_trained=False,
        independent_validation=False,adopted=False,robot_connected=False)
    write('verification.json',checks)
    # The reference replay deliberately exposes original stage changes and framing.
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>具名画家阶段参考</title><style>body{font:17px/1.7 system-ui;max-width:1250px;margin:auto;padding:24px;background:#f6f3eb;color:#294136}.notice{padding:18px;background:#e8ddc7}img{max-width:100%;height:auto;display:block;margin:18px 0}button,input{font:inherit}button{padding:8px 18px}#picture{width:500px;image-rendering:auto}a{color:#17694f}.viewer{border:1px solid #b8baae;padding:20px}</style><h1>苏春生黄山图：来源阶段参考回放</h1><p class="notice">人工整理的教学步骤，非模型自主识别、非实机直播。本轮新增6阶段参考和1件署名作品的来源证据；7张原图属于同一示范，分辨率多为约250像素。不能作为7件独立作品或真实分笔数据。</p><p>原文明确近景山石采用披麻与斧劈结合；该标签仅适用于原文所指近景山石。不得把整图强制标为单一皴法，也不能把树干、松针纳入皴笔像素监督。阶段1已经含勾皴，没有可分离的纯轮廓前帧。</p><div class="viewer"><button id="play">播放参考步骤</button> <input id="step" aria-label="来源阶段" type="range" min="1" max="6" value="1"><h2 id="title"></h2><p id="focus"></p><img id="picture" alt="所选原始教学步骤"><p id="description"></p><p>这段示范后期还返回松针修整；来源步骤是本例参考，不是所有山水画都必须遵守的固定顺序。图片取景不同，禁止直接作逐笔差分。</p></div><img src="stages_overview.png" alt="六阶段原图总览"><h2>其余具名课程核查</h2><p>刘学惟课程直接读取返回403，未获取视频；王兴堂披麻、斧劈及山石过程三节课程的正常权限查询标记为付费，未取流。韩敬伟已在旧候选清单，本轮仅复核官方介绍，不计为新作者/新视频。</p><p>来源页面：<a href="https://www.ecnu.edu.cn/info/1095/6915.htm">华东师范大学转载署名教学示范</a>。图片仅作本地参考核查。</p><p><a href="source_admission.json">作者、标签范围及去重边界</a> · <a href="stage_reference.json">来源步骤与依赖</a> · <a href="intake_summary.json">课程获取结果</a> · <a href="verification.json">检查结果</a></p><script>const steps=STEPS;let timer=null;const slider=document.getElementById('step');function show(){const s=steps[+slider.value-1];document.getElementById('title').textContent=s.stage+'. '+s.title;document.getElementById('focus').textContent='本阶段对象：'+s.object_focus;document.getElementById('picture').src=s.preview;document.getElementById('description').textContent=s.source_description.join(' ')}slider.addEventListener('input',show);document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='播放参考步骤';return}document.getElementById('play').textContent='暂停回放';timer=setInterval(()=>{slider.value=(+slider.value%6)+1;show()},2500)};show();</script></html>'''
    (ROOT/'index.html').write_text(page.replace('STEPS',json.dumps(phases,ensure_ascii=False).replace('</','<\\/')),encoding='utf8')
    report=PROJECT/'reports/20261118_teacher_intake.md'
    report.write_text('''# 不同绘者与作品证据补充

## 推进哪项目标
推进G1具名皴法及混合技法证据、G2树石与绘画阶段区分；暂不启动G3树木画法训练。

## 实际新增什么能力
核查苏春生署名黄山图示范，整理6阶段、7张来源图和5条来源步骤依赖，提供人工参考回放；不是自动规划能力。原文明确近景山石混合披麻与斧劈，保留该对象范围，不赋予整图或逐像素类别。新增阶段参考权限接口与5项逻辑测试，防止把步骤图当成逐笔真值。未新增识别模型。

## 还有哪些失败
图片多为约250像素且取景不同，第1阶段已含勾皴，不能恢复纯轮廓→逐笔加皴过程或执行帧差。真实分笔、墨带边界、单笔时序与物理参数仍缺。不同于上一具名示例绘者，但未认证在全部历史作品中未出现；有限像素去重不是作品级独立验收。
刘学惟课程返回403，王兴堂3节课程权限查询为付费，均未获取视频；韩敬伟是既有候选，仅复核官方说明。

## 是否通过独立验证
未进行。来源图哈希、有限图集去重、参考权限5项逻辑测试、冻结模型和500/489笔基线已检查；没有新训练、新模型采用或实机连接。已有窄线粘连问题仍未解决。

## 下一步
以这套不同绘者参考补齐混合技法和对象阶段证据；继续寻找可正常获取且逐笔清晰的单块山石视频。图文参考进入知识与阶段库，不能进入完整笔触或像素角色训练，也不把付费/拒绝访问的候选当已看完资料。
''',encoding='utf8')
    p=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=read(p);assert not any(s['capability_state'].values())
    s.update(latest_round_report=str(report),latest_teacher_evidence=str(ROOT/'source_admission.json'),new_model_trained_this_round=False,production_replaced=False,robot_connected=False);p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
    for name in ['project_plan_20261117.json','guohua_four_priorities_execution_20261116.json']:
        p=PROJECT/'research'/name;s=read(p);s.update(latest_round_report=str(report),latest_teacher_evidence=str(ROOT/'source_admission.json'));p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(checks,ensure_ascii=False))

if __name__=='__main__':main()
