# #################查看模型 的 anchor  #######################
import torch
from models.experimental import attempt_load

model = attempt_load(r'E:\yolov5-5.0-org\runs\train\jingsai\weights\best.pt', map_location=torch.device('cpu'))
m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
print(m.anchor_grid)

