import sys

from tarefa3_treinamento import treinar
from tarefa4_overfitting import teste_overfitting


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "overfitting":
        teste_overfitting()
        return

    model, tok, dados, _ = treinar()
    device = next(model.parameters()).device
    src = dados["src"][0].unsqueeze(0).to(device)

    out = model.translate(src, dados["start_id"], dados["eos_id"])
    ids = [t for t in out.squeeze(0).tolist() if t not in (dados["start_id"], dados["eos_id"])]

    print(f"\nEN: {tok.decode(dados['src'][0].tolist(), skip_special_tokens=True)}")
    print(f"DE: {tok.decode(ids, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
