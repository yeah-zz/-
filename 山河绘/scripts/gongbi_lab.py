"""Entry point for the E-drive-only gongbi training experiment."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure
configure()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['inspect','collect','prepare','train','infer','report','annotate','check','refine','compare','challenges','narrow_collect','narrow_select','narrow_prepare','narrow_report','narrow_evaluate','cunfa_collect','cunfa_prepare','cunfa_evaluate'])
    args, rest = parser.parse_known_args()
    if args.action == 'inspect':
        import torch, torchvision, json
        from gongbi_training.runtime import LAB
        result = dict(torch=torch.__version__, torchvision=torchvision.__version__,
                      cuda=torch.cuda.is_available(), lab=str(LAB))
        if result['cuda']:
            x = torch.rand(256,256,device='cuda',requires_grad=True)
            x.square().mean().backward()
            result.update(gpu=torch.cuda.get_device_name(0), gpu_memory=torch.cuda.get_device_properties(0).total_memory,
                          gradient_nonzero=bool(x.grad.abs().sum().item()>0))
        if '--model-smoke' in rest:
            from gongbi_training.model import GongbiLineNet,segmentation_loss
            torch.set_num_threads(4)
            model=GongbiLineNet(pretrained=True).cuda()
            image=torch.rand(2,3,255,319,device='cuda')
            target=torch.zeros(2,2,255,319,device='cuda');target[:,:,100:103,30:280]=1
            confidence=torch.ones_like(target)
            logits=model(image)
            assert logits.shape==target.shape
            loss,_=segmentation_loss(logits,target,confidence)
            loss.backward()
            assert torch.isfinite(loss) and model.head.weight.grad.abs().sum()>0
            result.update(model_parameters=sum(p.numel() for p in model.parameters()),
                          odd_image_shape_preserved=True,model_backward_finite=True,model_smoke_loss=float(loss.detach()))
            (LAB/'model_smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=True,indent=2))
    else:
        import importlib
        importlib.import_module('gongbi_training.'+args.action).main(rest)
