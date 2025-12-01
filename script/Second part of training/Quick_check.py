# 1) Model vs eval class count
print("out_features =", model.classifier[-1].out_features)  # adapt to your head
print("n_eval_classes =", len(class_names))
assert model.classifier[-1].out_features == len(class_names)

# 2) What indices are actually predicted?
logits = ...  # [N, C]
pred_idx = logits.argmax(dim=1)
print("unique predicted indices:", sorted(torch.unique(pred_idx).cpu().tolist()))
print("histogram:", torch.bincount(pred_idx, minlength=len(class_names)).cpu().tolist())

# 3) Ensure identical order to training
import json
train_map = json.load(open("classes.json"))  # name -> idx saved at training time
train_order = [k for k, v in sorted(train_map.items(), key=lambda kv: kv[1])]
assert train_order == class_names, f"Mismatch:\ntrain={train_order}\neval ={class_names}"
