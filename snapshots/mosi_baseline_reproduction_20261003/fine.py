"""Independent FINE reconstruction. See PROTOCOL.md for unresolved paper details."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


def mlp(a, b):
    return nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.Linear(b, b))


class QExpert(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 4, d) * .02)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d, 8, 4*d, .1, batch_first=True), 2)

    def forward(self, x, mask):
        return self.decoder(self.query.expand(x.size(0), -1, -1), x,
                            memory_key_padding_mask=~mask)


class MoQ(nn.Module):
    def __init__(self, inp, d):
        super().__init__()
        self.proj = nn.Linear(inp, d)
        self.router = nn.Linear(inp, 4)
        self.experts = nn.ModuleList([QExpert(d) for _ in range(4)])

    def forward(self, x, mask):
        # Explicitly prevent all-padding attention rows.
        mask = mask.clone()
        mask[~mask.any(1), 0] = True
        avg = (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        p = self.router(avg).softmax(-1)
        v, idx = p.topk(3, -1)
        gate = torch.zeros_like(p).scatter(1, idx, v)
        # Eq.18 TopK(Softmax), no post-selection renormalization.
        x = self.proj(x)
        outputs = torch.stack([e(x, mask) for e in self.experts], 1)
        out = (outputs * gate[:, :, None, None]).sum(1)
        fraction = (gate > 0).float().mean(0)
        aux = 4 * (fraction * gate.mean(0)).sum()
        return out, aux


class Critic(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d*2, d), nn.ReLU(), nn.Linear(d, 1))

    def scores(self, x, y):
        n = x.size(0)
        return self.net(torch.cat((x[:, None].expand(-1, n, -1),
                                   y[None].expand(n, -1, -1)), -1)).squeeze(-1)

    def lower(self, x, y):
        return F.cross_entropy(self.scores(x, y), torch.arange(x.size(0), device=x.device))

    def upper(self, x, y):
        scores = self.scores(x, y)
        return scores.diag().mean() - scores.mean()


class FINE(nn.Module):
    def __init__(self, bert_path, dims, train_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path, local_files_only=True)
        ds = [256, 128, 256]  # T,A,V; Table 7
        self.moq = nn.ModuleList([MoQ(i, d) for i, d in zip(dims, ds)])
        self.shared = nn.ModuleList([mlp(d, d//2) for d in ds])
        self.unique = nn.ModuleList([mlp(d, d//2) for d in ds])
        self.recon = nn.ModuleList([mlp(d, d) for d in ds])
        self.str_enc = nn.ModuleList([mlp(d//2, 128) for d in ds])
        self.utr_enc = nn.ModuleList([mlp(d//2, 128) for d in ds])
        self.mi_s_proj = nn.ModuleList([nn.Linear(d//2, 128) for d in ds])
        self.mi_u_proj = nn.ModuleList([nn.Linear(d//2, 128) for d in ds])
        self.label_s = mlp(1, 128)
        self.label_u = nn.ModuleList([mlp(1, 128) for _ in ds])
        self.sha_critics = nn.ModuleList([Critic(128) for _ in ds])
        self.uni_critics = nn.ModuleList([Critic(128) for _ in ds])
        self.str_critics = nn.ModuleList([Critic(256) for _ in ds])
        self.utr_critics = nn.ModuleList([Critic(128) for _ in ds])
        self.cls = nn.Parameter(torch.randn(3, 1, 256)*.02)
        self.uni_dec = nn.ModuleList([nn.TransformerDecoder(
            nn.TransformerDecoderLayer(256, 8, 1024, .1, batch_first=True), 1) for _ in ds])
        self.uni_head = nn.ModuleList([nn.Linear(256, 1) for _ in ds])
        self.fusion = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            256, 8, 1024, .1, batch_first=True), 2)
        self.head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 1))
        counts = torch.bincount(self.bins(torch.as_tensor(train_labels)), minlength=7)
        self.capacity = [max(int(.7*n), 8) for n in counts]
        self.queues = [[] for _ in range(7)]

    @staticmethod
    def bins(y):
        return torch.round(y.flatten()).clamp(-3, 3).long()+3

    def critics(self):
        return list(self.sha_critics)+list(self.uni_critics)+list(self.str_critics)+list(self.utr_critics)

    def forward(self, text, audio, vision, labels=None):
        tm = text[:, 1].bool()
        t = self.bert(input_ids=text[:, 0].long(), attention_mask=tm,
                      token_type_ids=text[:, 2].long()).last_hidden_state
        xs = [t, audio, vision]
        masks = [tm, audio.abs().sum(-1)>0, vision.abs().sum(-1)>0]
        qs, s, u, st, ut, logits, reps = [], [], [], [], [], [], []
        aux = t.new_zeros(())
        rec = t.new_zeros(())
        for m in range(3):
            q, a = self.moq[m](xs[m], masks[m]); qs.append(q); aux = aux+a
            sm, um = self.shared[m](q), self.unique[m](q)
            rec = rec+F.mse_loss(self.recon[m](torch.cat((sm, um), -1)), q)
            s.append(self.mi_s_proj[m](sm.mean(1)))
            u.append(self.mi_u_proj[m](um.mean(1)))
            stm, utm = self.str_enc[m](sm), self.utr_enc[m](um)
            st.append(stm.mean(1)); ut.append(utm.mean(1))
            r = torch.cat((stm, utm), -1)
            cls = self.cls[m:m+1].expand(t.size(0), -1, -1)
            dec = self.uni_dec[m](cls, r)
            logits.append(self.uni_head[m](dec[:, 0]))
            reps.append(torch.cat((cls, r), 1))
        fused = self.fusion(torch.cat(reps, 1))
        z = torch.cat([fused[:, m*5] for m in range(3)], -1)
        pred = self.head(z)
        out = {'logits_c': pred}
        if labels is None:
            return out  # No label encoder, queue or MI estimator is used for inference.
        ys = self.label_s(labels)
        yu = [enc(labels) for enc in self.label_u]
        pairs = [(0, 1), (0, 2), (1, 2)]
        mi = rec
        fit = t.new_zeros(())
        # Critic fitting uses detached representations; encoder objective uses frozen critics.
        for i, (a, b) in enumerate(pairs):
            ca, cb = torch.cat((st[a], ys), -1), torch.cat((st[b], ys), -1)
            for critic, left, right in ((self.sha_critics[i],s[a],s[b]),
                                       (self.uni_critics[i],u[a],u[b]),
                                       (self.str_critics[i],ca,cb)):
                fit = fit+critic.lower(left.detach(), right.detach())
            for group in self.critics():
                group.requires_grad_(False)
            mi = mi+self.sha_critics[i].lower(s[a],s[b])
            mi = mi+self.uni_critics[i].upper(u[a],u[b])+self.str_critics[i].upper(ca,cb)
            for group in self.critics():
                group.requires_grad_(True)
        for i in range(3):
            fit = fit+self.utr_critics[i].lower(ut[i].detach(),yu[i].detach())
            self.utr_critics[i].requires_grad_(False)
            mi = mi+self.utr_critics[i].lower(ut[i],yu[i])
            self.utr_critics[i].requires_grad_(True)
        cl = self.contrast(z, labels)
        up = sum(F.mse_loss(p, labels) for p in logits)
        loss = F.mse_loss(pred,labels)+.4*up+cl+.2*aux+.5*mi
        out.update(loss=loss, critic_loss=fit, parts={
            'mi': mi.detach(), 'cl':cl.detach(), 'aux':aux.detach(), 'up':up.detach()})
        return out

    def contrast(self, z, y):
        z = F.normalize(z, dim=-1); y = y.flatten()
        old = [item for q in self.queues for item in q]
        if old:
            kz = torch.cat((z, torch.stack([x[0] for x in old]).to(z.device)))
            ky = torch.cat((y, torch.stack([x[1] for x in old]).to(y.device)))
        else:
            kz, ky = z, y
        sim = (z@kz.T).clamp(-1+1e-6,1-1e-6)
        positive = self.bins(y)[:,None] == self.bins(ky)[None]
        valid = torch.ones_like(positive)
        valid[torch.arange(len(y)),torch.arange(len(y))] = False
        positive = positive & valid
        # Literal signed label difference from Eq.25; documented ambiguity.
        phi = math.pi*(1-(ky[None]-y[:,None])/6)
        comp = sim*phi.cos()-phi.sin().abs()*torch.sqrt(1-sim.square()+1e-8)
        scores = torch.where(positive,sim,comp)/.1
        scores = scores.masked_fill(~valid,float('-inf'))
        logp = scores-torch.logsumexp(scores,dim=1,keepdim=True)
        n = positive.sum(1)
        loss = -(logp.masked_fill(~positive,0).sum(1)/n.clamp_min(1))[n>0].mean() if (n>0).any() else z.sum()*0
        for zi,yi,bi in zip(z.detach().cpu(),y.detach().cpu(),self.bins(y).tolist()):
            self.queues[bi].append((zi,yi))
            self.queues[bi] = self.queues[bi][-self.capacity[bi]:]
        return loss
