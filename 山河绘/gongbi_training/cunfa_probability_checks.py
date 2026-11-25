"""Conditional role comparison which does not divide by an arbitrary epsilon."""
def conditional_role_check(original,refined):
    import torch
    a=original[:,1:].double();b=refined[:,1:].double()
    sa=a.sum(1,keepdim=True);sb=b.sum(1,keepdim=True)
    valid=(sa>1e-30)&(sb>1e-30)
    aa=a/sa.clamp_min(1e-300);bb=b/sb.clamp_min(1e-300)
    error=(aa-bb).abs().amax(1,keepdim=True)
    return dict(max_error=float(error[valid].max()) if valid.any() else 0.,
                zero_or_underflow_pixels_excluded=int((~valid).sum()),
                compared_pixels=int(valid.sum()))
