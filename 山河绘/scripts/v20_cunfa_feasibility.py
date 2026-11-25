from structure_train_v20 import *
from collections import Counter
STYLE_BOXES={
'mooc_gu_he_01':[[120,140,340,275]],'mooc_gu_he_02':[[120,140,300,310]],'mooc_gu_he_03':[[22,130,145,205],[145,115,232,220]],'mooc_gu_he_04':[[65,65,165,265]],'mooc_gu_he_05':[[58,28,174,144],[241,164,385,290]],
'book_pima_juran':[[100,60,430,340]],'book_pima_wangmeng':[[55,70,240,360]],'book_pima_huang':[[65,30,470,305]],'book_fupi_litang':[[220,60,525,505]],'book_fupi_mayuan':[[15,10,240,215]]}
LABELS=['pima','fupi']


def images():
    meta=read(ROOT/'source_manifest.json');lookup={r['id']:r for r in meta['records']};out=[];rng=np.random.default_rng(202520)
    for role in ['classification_train_ids','classification_test_ids']:
        for key in meta[role]:
            r=lookup[key];rgb=np.asarray(Image.open(r['image']).convert('RGB'));patches=[]
            for _ in range(96 if role=='classification_train_ids' else 24):
                x0,y0,x1,y1=STYLE_BOXES[key][int(rng.integers(len(STYLE_BOXES[key])))];maxs=min(x1-x0,y1-y0);size=int(rng.integers(max(28,int(maxs*.5)),max(29,maxs+1)));x=int(rng.integers(x0,max(x0+1,x1-size+1)));y=int(rng.integers(y0,max(y0+1,y1-size+1)));p=rgb[y:y+size,x:x+size];p=cv2.resize(p,(96,96));p=cv2.cvtColor(p,cv2.COLOR_RGB2GRAY);patches.append(np.repeat(p[:,:,None],3,axis=2))
            out.append(dict(id=key,label=LABELS.index(r['label']),split='train' if role=='classification_train_ids' else 'source_heldout_test',author=r['author'],source_group=r['source_group'],patches=patches,image_sha256=digest(r['image']),weak_label_basis=r['label_basis']))
    return out


@torch.no_grad()
def features(model,patches):
    output=[]
    for start in range(0,len(patches),32):
        x=torch.from_numpy(np.stack([v16.photometry(p)[0] for p in patches[start:start+32]])).cuda();a=model.a(x);b=model.b(F.avg_pool2d(a,2));c=model.c(F.avg_pool2d(b,2));f=torch.cat((c.mean((2,3)),c.std((2,3))),1);output.append(f.cpu().numpy())
    return np.concatenate(output)


def fit(X,y,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);mean=X.mean(0);std=X.std(0).clip(.05);x=torch.from_numpy((X-mean)/std).cuda();labels=torch.from_numpy(y).long().cuda();m=nn.Sequential(nn.Linear(128,24),nn.SiLU(),nn.Dropout(.25),nn.Linear(24,2)).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.02);groups=[np.flatnonzero(y==i) for i in range(2)]
    for step in range(400):
        idx=np.r_[rng.choice(groups[0],24),rng.choice(groups[1],24)];opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(x[idx]),labels[idx]);loss.backward();opt.step()
    return m.eval(),mean,std


@torch.no_grad()
def score(m,mean,std,X):return m(torch.from_numpy((X-mean)/std).cuda()).softmax(1).cpu().numpy()


def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);folder=ROOT/'cunfa_feasibility';folder.mkdir(exist_ok=True)
    if (folder/'final.pt').exists():raise FileExistsError('Preserve completed style feasibility model')
    encoder=v16.RibbonNet().cuda();encoder.load_state_dict(torch.load(OLD/'aligned'/'final.pt',map_location='cuda',weights_only=False)['model']);encoder.eval();rows=images()
    for r in rows:r['features']=features(encoder,r['patches']);r.pop('patches')
    train=[r for r in rows if r['split']=='train'];test=[r for r in rows if r['split']!='train'];cv=[]
    for i,hold in enumerate(train):
        keep=[r for r in train if r is not hold];X=np.concatenate([r['features'] for r in keep]);y=np.concatenate([np.full(len(r['features']),r['label'],np.int64) for r in keep]);m,mu,sd=fit(X,y,202530+i);prob=score(m,mu,sd,hold['features']).mean(0);cv.append(dict(id=hold['id'],expected=LABELS[hold['label']],prediction=LABELS[int(prob.argmax())],probabilities=prob.tolist(),same_author_may_remain_in_training=True))
    X=np.concatenate([r['features'] for r in train]);y=np.concatenate([np.full(len(r['features']),r['label'],np.int64) for r in train]);m,mu,sd=fit(X,y,202540);result=[]
    for r in test:
        probs=score(m,mu,sd,r['features']);p=probs.mean(0);vote=np.bincount(probs.argmax(1),minlength=2).max()/len(probs);result.append(dict(id=r['id'],expected=LABELS[r['label']],raw_prediction=LABELS[int(p.argmax())],probabilities=p.tolist(),patch_vote_agreement=float(vote),released_prediction='unknown_research_only',author=r['author'],source_group=r['source_group']))
    torch.save(dict(model=m.state_dict(),feature_mean=mu,feature_std=sd,encoder_weights=str(OLD/'aligned'/'final.pt'),labels=LABELS,experimental_only=True,source_works_train=len(train),source_works_test=len(test)),folder/'final.pt');summary=dict(training_source_images=len(train),test_source_images=len(test),training_patches_are_not_independent_artworks=sum(len(r['features']) for r in train),weak_work_level_labels_not_stroke_expert_gold=True,work_leave_one_out=cv,leave_one_work_accuracy=float(np.mean([r['expected']==r['prediction'] for r in cv])),source_heldout_records=result,source_heldout_accuracy=float(np.mean([r['expected']==r['raw_prediction'] for r in result])),selected_updates=400,cv_updates=2000,known_classes=LABELS,unsupported_classes=['heye','jiesuo','yuntou','midian','other_mixed'],production_release=False,reason='Only five source images for training; weak inherited labels and mixed techniques, cannot claim reliable cunfa recognition')
    jsave(folder/'summary.json',summary);jsave(folder/'data_manifest.json',dict(records=[{k:v for k,v in r.items() if k!='features'} for r in rows],style_boxes=STYLE_BOXES,classification_did_not_update_line_model=True));print('V20_CUNFA',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
