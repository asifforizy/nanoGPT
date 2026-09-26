import os

import torch
from torch.nn import functional as F

from app.model import BigramLanguageModel
from app.tokenizer import CharacterTokenizer


# -------------------------
# Hyperparameters
# -------------------------

batch_size = 16
block_size = 32

max_iters = 5000
eval_interval = 100
eval_iters = 200

learning_rate = 1e-3

n_embd = 64
n_head = 4
n_layer = 4

dropout = 0.0

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)


# -------------------------
# Paths
# -------------------------

DATA_PATH = "data/input.txt"
CHECKPOINT_DIR = "checkpoints"

MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "nanoGPT.pth"
)

TOKENIZER_PATH = os.path.join(
    CHECKPOINT_DIR,
    "tokenizer.json"
)


# -------------------------
# Load dataset
# -------------------------

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as f:
    text = f.read()


# -------------------------
# Tokenizer
# -------------------------

tokenizer = CharacterTokenizer(text)

vocab_size = tokenizer.vocab_size

data = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)


# -------------------------
# Train / validation split
# -------------------------

n = int(0.9 * len(data))

train_data = data[:n]

val_data = data[n:]


# -------------------------
# Batch loading
# -------------------------

def get_batch(split):

    data_split = (
        train_data
        if split == "train"
        else val_data
    )

    ix = torch.randint(
        len(data_split) - block_size,
        (batch_size,)
    )

    x = torch.stack(
        [
            data_split[i:i + block_size]
            for i in ix
        ]
    )

    y = torch.stack(
        [
            data_split[i + 1:i + block_size + 1]
            for i in ix
        ]
    )

    x = x.to(device)
    y = y.to(device)

    return x, y


# -------------------------
# Model
# -------------------------

model = BigramLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout
)

model = model.to(device)

print(
    sum(
        p.numel()
        for p in model.parameters()
    ) / 1e6,
    "M parameters"
)


# -------------------------
# Loss estimation
# -------------------------

@torch.no_grad()
def estimate_loss():

    out = {}

    model.eval()

    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            X, Y = get_batch(split)

            _, loss = model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out


# -------------------------
# Optimizer
# -------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# -------------------------
# Training
# -------------------------

print(f"Training on: {device}")

for iteration in range(max_iters):

    if (
        iteration % eval_interval == 0
        or iteration == max_iters - 1
    ):

        losses = estimate_loss()

        print(
            f"step {iteration}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    optimizer.step()


# -------------------------
# Save model
# -------------------------

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)

tokenizer.save(
    TOKENIZER_PATH
)


print()
print("Training completed.")
print(f"Model saved to: {MODEL_PATH}")
print(f"Tokenizer saved to: {TOKENIZER_PATH}")