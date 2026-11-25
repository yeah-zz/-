"""Conservative candidate routing, not verified semantics or cunfa understanding."""
import numpy as np
def rock_candidate_mask(probabilities,confidence=.75,margin=.20):
    p=np.asarray(probabilities)
    if p.ndim!=3 or p.shape[0]!=4 or not np.isfinite(p).all():raise ValueError('Expected finite4xHxW scene probabilities')
    if np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(0),1,atol=1e-3):raise ValueError('Unnormalised scene probabilities')
    rivals=np.max(p[[0,2,3]],axis=0)
    return (p.argmax(0)==1)&(p[1]>=confidence)&((p[1]-rivals)>=margin)

def cunfa_candidate_route(probabilities,style_proposal=None,minimum_rock_fraction=.65,confidence=.75,margin=.20):
    mask=rock_candidate_mask(probabilities,confidence,margin);fraction=float(mask.mean());eligible=fraction>=minimum_rock_fraction
    return dict(rock_candidate_fraction=fraction,eligible_for_exploratory_cunfa=eligible,style_proposal=style_proposal if eligible else None,cunfa='unknown',object_verified=False,prediction_as_label=False,reason='candidate_rock_support' if eligible else 'insufficient_rock_or_uncertain',not_a_rock_instance=True,not_calibrated_probability=True)
