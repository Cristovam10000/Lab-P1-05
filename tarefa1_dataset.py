from datasets import load_dataset


def carregar_dataset(nome="bentrevett/multi30k", split="train", n=1000):
    print(f"Baixando {nome}...")
    ds = load_dataset(nome, split=split)
    ds = ds.select(range(min(n, len(ds))))

    pares = [(ex["en"], ex["de"]) for ex in ds]
    print(f"{len(pares)} pares carregados")
    return pares


if __name__ == "__main__":
    pares = carregar_dataset()
    for en, de in pares[:5]:
        print(f"  {en}\n  -> {de}\n")
