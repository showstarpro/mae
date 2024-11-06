import models_mae
import torch


model = models_mae.__dict__["mae_vit_base_patch16"](norm_pix_loss=True)
model = model.cuda()
model.train()

inputs = torch.randn(2, 3, 224, 224)
inputs = inputs.cuda()

loss, _, _ = model(inputs)

loss.backward()
print(loss)