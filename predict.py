import torch
from translate import load_model
from transformer.dataset import DataBase

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- recreate SAME config used in training ----
class Opt:
    pass

opt = Opt()
opt.data = "DATASET/umls"
opt.jump = 3
opt.padding = 80
opt.n_head = 2
opt.d_v = 16
opt.n_layers = 2
opt.dropout = 0.1

# ---- load dataset ----
base_data = DataBase(opt.data, subgraph=opt.data + f'/subgraph{opt.jump}')
opt.src_vocab_size, opt.trg_vocab_size = base_data.getinfo()
opt.src_pad_idx = base_data.e2id['<pad>']

# ---- load model ----
model = load_model(opt, device, base_data.nebor_relation)

# 🔴 UPDATE THIS PATH (VERY IMPORTANT)
ckpt_path = "EXPS/distilbert-fixed-j3_20260412_16-01-31/Translator1.ckpt"
model.load_state_dict(torch.load(ckpt_path, map_location=device))

model.eval()

# ---- reverse mapping (id → entity name) ----
id2e = {v: k for k, v in base_data.e2id.items()}
id2r = base_data.id2r

# ---- take real example from dataset ----
sample = base_data.train_triples[0]  # (head, relation, tail)

head, relation, tail = sample

input_ids = torch.tensor([[head, relation]]).to(device)

# ---- predict ----
with torch.no_grad():
    output = model(input_ids)
    pred = output.argmax(-1)

pred_id = pred[0, -1].item()

# ---- print results ----
print("\n===== PREDICTION =====")

print("Input (IDs):", head, relation)
print("Input (Text):", id2e.get(head), "|", id2r.get(relation))

print("\nActual Tail:", id2e.get(tail))
print("Predicted Tail:", id2e.get(pred_id, "Unknown"))
