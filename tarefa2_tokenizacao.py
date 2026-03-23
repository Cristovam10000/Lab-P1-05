import torch
from transformers import AutoTokenizer
from tarefa1_dataset import carregar_dataset

MAX_LEN = 50  # trunca frases maiores q isso


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    tok.add_special_tokens({"additional_special_tokens": ["<START>", "<EOS>"]})
    return tok


def tokenizar_pares(pares, tok):
    start_id = tok.convert_tokens_to_ids("<START>")
    eos_id = tok.convert_tokens_to_ids("<EOS>")
    pad_id = tok.pad_token_id

    srcs, tgt_ins, tgt_outs = [], [], []

    for en, de in pares:
        s = tok.encode(en, add_special_tokens=False)[:MAX_LEN]
        t = tok.encode(de, add_special_tokens=False)[:MAX_LEN - 1]

        # decoder input: <START> + frase  |  label: frase + <EOS>
        ti = [start_id] + t
        to = t + [eos_id]

        # pad tudo pro mesmo tamanho
        srcs.append(s + [pad_id] * (MAX_LEN - len(s)))
        tgt_ins.append(ti + [pad_id] * (MAX_LEN - len(ti)))
        tgt_outs.append(to + [pad_id] * (MAX_LEN - len(to)))

    # print(f"src[0][:10] = {srcs[0][:10]}")  # debug
    return {
        "src": torch.tensor(srcs), "tgt_in": torch.tensor(tgt_ins),
        "tgt_out": torch.tensor(tgt_outs),
        "pad_id": pad_id, "start_id": start_id, "eos_id": eos_id,
        "vocab_size": len(tok),
    }


def preparar_dados(n=1000):
    """carrega dataset + tokeniza, retorna (dados_dict, tokenizer)"""
    pares = carregar_dataset(n=n)
    tok = get_tokenizer()
    return tokenizar_pares(pares, tok), tok


if __name__ == "__main__":
    dados, tok = preparar_dados(n=5)
    print(f"vocab={dados['vocab_size']} pad={dados['pad_id']} start={dados['start_id']} eos={dados['eos_id']}")
    print(f"src {dados['src'].shape}  tgt_in {dados['tgt_in'].shape}")
