# Lab P1-05: Treinamento Fim-a-Fim do Transformer

Último lab da Unidade I. Aqui a gente pega o Transformer que foi montado nos Labs anteriores (01 a 04), conecta num dataset real de tradução (EN→DE) do Hugging Face, e faz ele treinar de verdade com backpropagation.

O objetivo não é ter um tradutor bom — pra isso precisaria de dias de GPU. O ponto é provar que a arquitetura funciona: a loss cai, os gradientes fluem, e no teste de overfitting o modelo consegue decorar um conjunto pequeno de frases.

## Estrutura

| Arquivo | O que faz |
|---|---|
| `transformer_model.py` | Transformer completo (Encoder + Decoder), vem dos labs anteriores |
| `tarefa1_dataset.py` | Baixa o multi30k do HuggingFace e pega 1000 pares EN→DE |
| `tarefa2_tokenizacao.py` | Tokeniza com `bert-base-multilingual-cased`, adiciona `<START>`/`<EOS>`, faz padding |
| `tarefa3_treinamento.py` | Loop de treinamento: CrossEntropyLoss + Adam |
| `tarefa4_overfitting.py` | Treina com 10 frases pra provar que os gradientes tão funcionando |
| `executar_tudo.py` | Roda tudo junto |

## Como rodar

```bash
# criar venv e instalar deps
python -m venv venv
venv\Scripts\activate        # Windows
pip install torch datasets transformers

# treino normal (1000 frases, 15 epocas)
python executar_tudo.py

# teste de overfitting (10 frases, 100 epocas)
python executar_tudo.py overfitting
```

Também dá pra rodar cada tarefa separada: `python tarefa1_dataset.py`, `python tarefa2_tokenizacao.py`, etc.

## Hiperparâmetros

- d_model = 128, num_heads = 4, d_ff = 512, n_layers = 2
- Otimizador: Adam (lr=1e-4, no overfitting test lr=3e-4)
- Loss: CrossEntropyLoss com `ignore_index=pad_id` pra não penalizar o padding
- Batch size: 32 (treino normal), 10 (overfitting)

## Resultados

No treino normal a loss caiu de ~11.2 pra ~5.4 em 15 épocas, mostrando que o modelo tá aprendendo. A tradução ainda sai ruim (esperado com tão pouco treino), mas a convergência tá clara.

No teste de overfitting a loss deve chegar perto de 0 e o modelo reproduz a frase de treino, provando que a arquitetura inteira tá correta.

## Entrega (versionamento)

O contrato pede **tag Git `v1.0`** no GitHub (marca de versão; a mensagem do commit pode ser qualquer texto claro).

No commit que fecha a entrega:

```bash
git tag v1.0
git push origin v1.0
```

## Uso de IA

Usei o Claude Code (Anthropic) pra me ajudar com a parte de manipulação do dataset e tokenização (Tarefas 1 e 2) — especificamente pra entender a API do HuggingFace (`load_dataset`, `AutoTokenizer`) e a lógica de montar os tensores com padding. O training loop da Tarefa 3 e o teste de overfitting da Tarefa 4 foram escritos em cima das classes que construí nos labs anteriores, seguindo o fluxo de forward/backward.
