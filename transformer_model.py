import math
import torch
import torch.nn as nn


# eq. do paper Attention is All You Need (scaled dot product)
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_out):
        return self.norm(x + sublayer_out)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q_in, k_in, v_in, mask=None):
        b = q_in.size(0)
        q = self.w_q(q_in).view(b, -1, self.num_heads, self.d_head).transpose(1, 2)
        k = self.w_k(k_in).view(b, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = self.w_v(v_in).view(b, -1, self.num_heads, self.d_head).transpose(1, 2)

        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        out, w = scaled_dot_product_attention(q, k, v, mask)
        # concatena as heads de volta pra d_model
        out = out.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        return self.w_o(out), w


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = AddNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.norm2 = AddNorm(d_model)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, mask)  # self-attention
        x = self.norm1(x, attn_out)
        x = self.norm2(x, self.ffn(x))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, n_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(n_layers)])
        self.scale = math.sqrt(d_model)

    def forward(self, src, mask=None):
        x = self.pe(self.embed(src) * self.scale)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = AddNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = AddNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.norm3 = AddNorm(d_model)

    def forward(self, y, enc_out, tgt_mask=None):
        out, _ = self.self_attn(y, y, y, tgt_mask)  # masked self-attn
        y = self.norm1(y, out)
        out, _ = self.cross_attn(y, enc_out, enc_out)  # cross-attn (Q=dec, K=V=enc)
        y = self.norm2(y, out)
        y = self.norm3(y, self.ffn(y))
        return y


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, n_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff) for _ in range(n_layers)])
        self.scale = math.sqrt(d_model)

    def forward(self, tgt, memory, tgt_mask=None):
        y = self.pe(self.embed(tgt) * self.scale)
        for layer in self.layers:
            y = layer(y, memory, tgt_mask)
        return y


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, num_heads=4, d_ff=512, n_layers=2):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, num_heads, d_ff, n_layers)
        self.decoder = Decoder(tgt_vocab, d_model, num_heads, d_ff, n_layers)
        self.out_proj = nn.Linear(d_model, tgt_vocab)

    def _causal_mask(self, sz, device):
        return torch.tril(torch.ones(sz, sz, device=device)).bool().unsqueeze(0).unsqueeze(0)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        mask = self._causal_mask(tgt.size(1), tgt.device)
        dec_out = self.decoder(tgt, memory, mask)
        return self.out_proj(dec_out)  # logits, sem softmax (CrossEntropy ja aplica)

    @torch.no_grad()
    def translate(self, src, start_id, eos_id, max_len=50):
        """greedy decoding token a token"""
        self.eval()
        memory = self.encoder(src)
        seq = torch.tensor([[start_id]], dtype=torch.long, device=src.device)

        for _ in range(max_len):
            mask = self._causal_mask(seq.size(1), seq.device)
            out = self.decoder(seq, memory, mask)
            next_tok = self.out_proj(out[:, -1, :]).argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=1)
            if next_tok.item() == eos_id:
                break

        return seq
