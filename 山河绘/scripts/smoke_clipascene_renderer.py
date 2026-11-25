import run_clipascene_landscape
import torch
import pydiffvg
torch.set_num_threads(4)
pydiffvg.set_use_gpu(False)
pydiffvg.set_device(torch.device('cpu'))
points=torch.tensor([[12.,12.],[25.,70.],[70.,25.],[100.,100.]],requires_grad=True)
path=pydiffvg.Path(num_control_points=torch.tensor([2],dtype=torch.int32),points=points,stroke_width=torch.tensor(1.5),is_closed=False)
group=pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]),fill_color=None,stroke_color=torch.tensor([0.,0.,0.,1.]))
scene=pydiffvg.RenderFunction.serialize_scene(128,128,[path],[group])
print('RENDER_BEGIN',flush=True)
im=pydiffvg.RenderFunction.apply(128,128,2,2,0,None,*scene)
print('RENDER_OK',im.shape,float(im[...,3].sum()),flush=True)
im[...,3].sum().backward()
print('GRADIENT_OK',points.grad,flush=True)
