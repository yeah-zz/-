# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode=True
from stroke_geometry_brush_20261101 import *
from freeze_stability_brush_20261101 import ROOT
checks=[]
def check(name,ok):checks.append(dict(name=name,passed=bool(ok)));assert ok,name

def gaussian_strip(sigma,density):
    yy=np.arange(496,dtype=float)[:,None];d=density*np.exp(-.5*((yy-200)/sigma)**2);a=np.rint(240*np.exp(-d)).astype(np.uint8);return np.repeat(np.repeat(a[:,:,None],496,axis=1),3,axis=2)

def main():
    t=np.linspace(0,math.pi,100);p=np.column_stack([220+60*np.cos(t),220+60*np.sin(t)]);fitted,curves=fit_paths([p],(496,496));check('curve_deviation_bounded',curves[0]['max_sampled_hausdorff_px']<=.6);check('curve_endpoints_fixed',np.allclose(fitted[0][[0,-1]],p[[0,-1]]));check('curve_svg_primitives_reduced',len(curves[0]['segments'])<len(p)-1)
    corner=np.r_[np.column_stack([np.arange(50,101),np.full(51,80)]),np.column_stack([np.full(50,100),np.arange(81,131)])];q,r=fit_paths([corner],(496,496));check('sharp_corner_retained',float(cKDTree(q[0]).query([[100,80]])[0][0])<.1)
    p1=np.column_stack([np.arange(50,150),np.full(100,100)]);p2=p1+[0,2];q,_=fit_paths([p1,p2],(496,496));check('parallel_paths_stay_separate',cv2.connectedComponents(v4.raster((496,496),q).astype(np.uint8),8)[0]-1==2)
    line=np.column_stack([np.arange(70,180),np.full(110,200)])
    for sigma,d in [(.6,.3),(1.,1.),(2.4,.6)]:
        row=profile_parameters(gaussian_strip(sigma,d),line);valid=[x for x in row['width_px'] if x is not None];measured=float(np.median(valid));check('gaussian_FWHM_'+str(sigma),abs(measured-2.35482*sigma)<.4);check('gaussian_density_'+str(sigma),abs(float(np.median([x for x in row['optical_density'] if x is not None]))-d)<.035)
    row=profile_parameters(gaussian_strip(12,.6),line);check('broad_wash_not_silent_large_brush',not any(row['valid']));row=profile_parameters(gaussian_strip(1.,.01),line);check('too_faint_measurement_kept_unknown',not any(row['valid']))
    gradient=np.tile(np.linspace(170,235,496,dtype=np.uint8)[:,None,None],(1,496,3));row=profile_parameters(gradient,line);check('smooth_wash_not_measured_as_ink',not any(row['valid']))
    parameters=[dict(points=line.tolist(),width_px=[3.]*len(line),optical_density=[1.]*len(line),valid=[True]*len(line),reason=['synthetic']*len(line))];im=render_parameters((496,496),parameters);check('ribbon_has_no_repeated_stamp_beads',int(np.ptp(im[200,80:170]))<=2);check('ribbon_tone_matches_density',abs(float(np.median(im[200,80:170]))-255/math.e)<3)
    patch=dict(points=[[10.,10.],[11.,10.],[12.,10.]],width_px=[5.,5.,5.],optical_density=[1.,1.,1.],valid=[True]*3,reason=['synthetic']*3);patch=regularize_parameters(patch,1.);check('short_ink_blob_not_interpreted_as_wide_stroke',not any(patch['valid']) and patch['raw_profile_width_px']==[5.,5.,5.])
    widths=[2.]*31;widths[15]=9.;row=dict(points=[[float(x),20.] for x in range(31)],width_px=widths,optical_density=[.6]*31,valid=[True]*31,reason=['synthetic']*31);row=regularize_parameters(row,1.);check('isolated_width_spike_kept_unknown',row['width_px'][15] is None and row['raw_profile_width_px'][15]==9.)
    widths=[2.]*31;widths[15]=None;dark=[.6]*31;dark[15]=None;flags=[True]*31;flags[15]=False;row=dict(points=[[float(x),20.] for x in range(31)],width_px=widths,optical_density=dark,valid=flags,reason=['synthetic']*31);row=regularize_parameters(row,1.);check('unknown_measurement_not_filled_by_smoothing',row['width_px'][15] is None and row['optical_density'][15] is None)
    check('no_physical_pressure_inferred',CONFIG['physical_z'] is None and CONFIG['physical_speed'] is None and CONFIG['water_content'] is None);jsave(ROOT/'brush_behavior_checks.json',dict(checks=checks,count=len(checks),all_passed=True,synthetic_checks_not_physical_calibration=True));print('BRUSH_BEHAVIOR_CHECKS',len(checks),flush=True)
if __name__=='__main__':main()
