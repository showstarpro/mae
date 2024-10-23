import models_map
import torch


model = models_map.__dict__["mae_vit_base_patch16"](norm_pix_loss=True)
model = model.cuda()


inputs = torch.randn(2, 3, 224, 224)
inputs = inputs.cuda()

out = model(inputs)

print(out)