import torch
from common import SmallCNN

ckpt_path = "outputs/best_model.pt"
out_path = "outputs/model_scripted.pt"

Ckpt = torch.load(ckpt_path, map_location="cpu")

num_classes = len(Ckpt["classes"])
in_ch = Ckpt.get("in_ch", 3)
model = SmallCNN(num_classes=num_classes, in_ch=in_ch)
model.load_state_dict(Ckpt["model_state"])
model.eval()

scripted = torch.jit.script(model)
scripted.save(out_path)

print(f"✅ model_scripted.pt saved at: {out_path}")
