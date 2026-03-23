import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformer_model import Transformer
from tarefa2_tokenizacao import preparar_dados


def treinar(epochs=15, batch_size=32, d_model=128, num_heads=4, d_ff=512,
            n_layers=2, lr=1e-4, n_frases=1000):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dados, tok = preparar_dados(n=n_frases)
    vocab_size = dados["vocab_size"]
    pad_id = dados["pad_id"]

    model = Transformer(
        src_vocab=vocab_size, tgt_vocab=vocab_size,
        d_model=d_model, num_heads=num_heads, d_ff=d_ff, n_layers=n_layers,
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    # ignore_index pro padding nao atrapalhar a loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(
        TensorDataset(dados["src"], dados["tgt_in"], dados["tgt_out"]),
        batch_size=batch_size, shuffle=True
    )

    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        count = 0

        for src_b, tin_b, tout_b in loader:
            src_b, tin_b, tout_b = src_b.to(device), tin_b.to(device), tout_b.to(device)

            logits = model(src_b, tin_b)
            # CrossEntropy quer (N, classes), entao achata batch*seq
            loss = loss_fn(logits.view(-1, vocab_size), tout_b.view(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            count += 1

        avg = running / count
        hist.append(avg)
        print(f"ep {ep:3d}/{epochs}  loss={avg:.4f}")

    print(f"loss {hist[0]:.4f} -> {hist[-1]:.4f}")
    return model, tok, dados, hist


if __name__ == "__main__":
    treinar()
