"""Plot recorded servo registers; no robot/camera I/O and no force inference."""
import argparse,json,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.runtime/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');args=p.parse_args()
 rows=[json.loads(s) for s in Path(args.input).read_text().splitlines() if s.strip()]
 samples=[r for r in rows if r.get('type')=='sample'];first=samples[0]['capture_monotonic']
 t=np.array([r['capture_monotonic']-first for r in samples]);pos=np.array([r['position'] for r in samples]);goal=np.array([r['goal'] for r in samples])
 current=np.array([(-(r['current_raw']&32767) if r['current_raw']&32768 else r['current_raw'])*6.5 for r in samples])
 fig,axes=plt.subplots(3,1,figsize=(12,7.8),sharex=True,gridspec_kw={'height_ratios':[2,1,1]})
 fig.patch.set_facecolor('#f8fafd')
 axes[0].plot(t,(goal-1520)*360/4096,color='#4285f4',lw=2,label='Commanded angle')
 axes[0].plot(t,(pos-1520)*360/4096,color='#34a853',lw=1.2,label='Measured angle')
 axes[0].set_ylabel('Relative motor angle (deg)');axes[0].legend(loc='upper right',frameon=False)
 axes[1].plot(t,[abs(r['load_signed_pwm'])/10 for r in samples],color='#e5a000',lw=1.2)
 axes[1].axhline(10,color='#a7adb4',ls='--',lw=1);axes[1].set_ylim(-.5,11);axes[1].set_ylabel('Drive PWM (%)')
 axes[2].step(t,current,color='#ea4335',lw=.9,where='mid');axes[2].set_ylabel('Current register (mA)');axes[2].set_xlabel('Time since first sample (s)')
 for a in axes:
  a.set_facecolor('white');a.grid(axis='y',alpha=.15)
  a.spines[['top','right']].set_visible(False)
 for c in sorted({r['cycle'] for r in samples}):
  ts=[r['capture_monotonic']-first for r in samples if r['cycle']==c]
  axes[0].text((min(ts)+max(ts))/2,35,f'Empty cycle {c+1}',ha='center',color='#5f6368')
 fig.suptitle('LeRobot | measured empty-gripper baseline',x=.09,ha='left',fontsize=19,fontweight='bold')
 fig.text(.09,.925,f'{len(samples)} real samples · 50 ticks/s · temporary PWM cap 10% · original cap restored',color='#5f6368',fontsize=11)
 fig.text(.09,.02,'PWM and current are electrical measurements, not calibrated contact force. Angle is relative to encoder 1520, not jaw gap.',color='#5f6368',fontsize=9)
 fig.tight_layout(rect=[0,.04,1,.92]);Path(args.output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=170)
if __name__=='__main__':main()
