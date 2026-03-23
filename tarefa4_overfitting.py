from tarefa3_treinamento import treinar


def teste_overfitting():
    # treina com poucas frases pra ver se o modelo decora (prova q gradientes tao ok)
    print("overfitting test - 10 frases, 100 epocas\n")

    model, tok, dados, hist = treinar(
        epochs=100, batch_size=10, lr=3e-4, n_frases=10,
    )

    device = next(model.parameters()).device
    sid = dados["start_id"]
    eid = dados["eos_id"]

    # pega a primeira frase e tenta traduzir
    src = dados["src"][0].unsqueeze(0).to(device)
    src_txt = tok.decode(dados["src"][0].tolist(), skip_special_tokens=True)

    # resposta certa (tira padding e eos)
    ref_ids = [t for t in dados["tgt_out"][0].tolist() if t != dados["pad_id"] and t != eid]
    ref_txt = tok.decode(ref_ids, skip_special_tokens=True)

    # gera traducao
    out = model.translate(src, sid, eid)
    out_ids = out.squeeze(0).tolist()
    # limpa tokens especiais
    if out_ids[0] == sid:
        out_ids = out_ids[1:]
    if out_ids and out_ids[-1] == eid:
        out_ids = out_ids[:-1]
    gen_txt = tok.decode(out_ids, skip_special_tokens=True)

    print(f"\nEN:       {src_txt}")
    print(f"esperado: {ref_txt}")
    print(f"gerado:   {gen_txt}")
    print(f"loss:     {hist[-1]:.6f}")
    # TODO: plotar curva de loss com matplotlib

    return model, hist


if __name__ == "__main__":
    teste_overfitting()
