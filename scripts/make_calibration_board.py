"""Regenerate the A4 landscape / 20 mm board used by BoardSpec.

Optional document-generation dependency: reportlab. Does not touch devices.
Run from the repository root. The PDF embeds a system Unicode font.
"""
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path
pdfmetrics.registerFont(TTFont('BoardFont','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
p=Path('output/pdf/lerobot-a4-calibration-board.pdf')
p.parent.mkdir(parents=True,exist_ok=True)
Path('public/calibration').mkdir(parents=True,exist_ok=True)
c=canvas.Canvas(str(p),pagesize=(297*mm,210*mm),pageCompression=1)
c.setFillColorRGB(1,1,1);c.rect(0,0,297*mm,210*mm,fill=1,stroke=0);c.setFillColorRGB(0,0,0)
c.setTitle('LeRobot A4 calibration - 20 mm squares, 9 x 6 inner corners')
c.setFont('BoardFont',15);c.drawString(20*mm,199*mm,'LeRobot | 腕部相機校準棋盤格')
c.setFont('BoardFont',10);c.drawString(20*mm,190*mm,'A4 橫向。請選實際大小 / 100%，勿選符合頁面。每格 20 mm，內角點 9 x 6。')
x0=48.5*mm;y0=43*mm;s=20*mm
for row in range(7):
 for col in range(10):
  if (row+col)%2==0:c.rect(x0+col*s,y0+(6-row)*s,s,s,fill=1,stroke=0)
c.setLineWidth(.6);c.line(48.5*mm,33*mm,148.5*mm,33*mm)
for x in (48.5,148.5):c.line(x*mm,31*mm,x*mm,35*mm)
c.setFont('BoardFont',10);c.drawString(155*mm,31.5*mm,'核對線：列印後應為 100 mm / 10 公分。')
c.setFont('BoardFont',10)
c.drawString(20*mm,20*mm,'請貼平於硬板、避免反光。放在手臂前方，讓整張棋盤格入鏡；同一輪拍攝請保持紙張不動。')
c.drawString(20*mm,11*mm,'紙張移動後請重新開始校準。這張紙只供一次性設定，正式抓取不需要留在桌上。')
c.showPage();c.save()
Path('public/calibration/lerobot-a4-calibration-board.pdf').write_bytes(p.read_bytes())
print(p.resolve())
