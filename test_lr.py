import matplotlib.pyplot
from obj_detection.train import ObjDetectionExponentialDecayLR, ObjDetectionCosineAnnealingLR, ObjDetectionCosineDecayLR, ObjDetectionLogisticDecayLR, ObjDetectionLR, ObjDetectionRampUpLR
import matplotlib

lr = 1e-3
epochs = 1000
lr_ramp_down = 1000
steps = epochs*100

# scheduler = ObjDetectionLR(None, lr, 0.01, lr_ramp_down)
scheduler = ObjDetectionRampUpLR(None, lr, lr_ramp_down)
# scheduler = ObjDetectionCosineDecayLR(None, lr, 1e-8, steps, lr_ramp_down)
# scheduler = ObjDetectionExponentialDecayLR(None, lr, 1e-8, steps, lr_ramp_down)
# scheduler = ObjDetectionLogisticDecayLR(None, lr, 1e-8, steps, lr_ramp_down)
# scheduler = ObjDetectionCosineAnnealingLR(None, lr, 1e-8, lr_ramp_down, 1000, 2, 4)

lrs = []
for i in range(steps):
    lr = scheduler.get_last_lr()
    scheduler._current_step+=1
    lrs.append(lr)
    
matplotlib.pyplot.plot(lrs)
matplotlib.pyplot.show()