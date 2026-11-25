from run_validation_v5 import *
if __name__=='__main__':
    result=read(ROOT/'validation_results.json');assert result['complete']
    jsave(ROOT/'initial_display_threshold_diagnostics.json',result)
    # Confirm the discrete topology convention with a known closed pixel ring.
    ring=np.zeros((30,30),np.uint8);cv2.circle(ring,(15,15),7,1,1)
    count,labels,stats,_=cv2.connectedComponentsWithStats(1-ring,connectivity=4)
    assert sum(1 for s in stats[1:] if s[0]>0 and s[1]>0 and s[0]+s[2]<30 and s[1]+s[3]<30)>=1
    for r in result['records']:
        for version in ['v4','v5']:r[version]=metric(ROOT/'validation_outputs'/r['id']/version)
    result['diagnostic_revision']='Exact LINE_8 path raster with dual 4-connected white-space components; fixes diagonal breakage in antialiased PNG threshold analysis. Algorithms and parameters unchanged.'
    jsave(ROOT/'validation_results.json',result)
    print('REANALYZED_EXACT_GEOMETRY',len(result['records']),flush=True)
