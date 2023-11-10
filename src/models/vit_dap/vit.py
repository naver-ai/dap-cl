#!/usr/bin/env python3
"""
vit with DAP
"""

import os
from ..vit_backbones.vit import *

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair


MODEL_ZOO = {
    "sup_vitb16_imagenet21k": "imagenet21k_ViT-B_16.npz",
}

class ADPT_Block(nn.Module):
    def __init__(self, config, vis, dap_config):
        super(ADPT_Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)

        self.config = config
        self.attn = Attention(config, vis)

        # domain-adaptive prompts
        self.dap_config = dap_config
        self.dap_downsample = nn.Linear(197, dap_config.NUM_DAP_TOKENS)
        nn.init.zeros_(self.dap_downsample.weight)
        nn.init.zeros_(self.dap_downsample.bias)
        self.dap_film = nn.Linear(dap_config.TASK_EMB, config.hidden_size * 2)
        self.dap_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, x, task_id_estimated_emb=None, layer_index=None, cfg=None):
        if x.shape[1] == 197: # first layer
            x_norm = self.dap_norm(x)
            x_tran = torch.transpose(x_norm, 2, 1)
            down = self.dap_downsample(x_tran)

            film = self.dap_film(task_id_estimated_emb)
            gamma4 = film[:, :self.config.hidden_size]
            beta4 = film[:, self.config.hidden_size:]
            gamma_norm = gamma4.norm(p=2, dim=1, keepdim=True).detach()
            beta_norm = beta4.norm(p=2, dim=1, keepdim=True).detach()

            gamma4 = gamma4.div(gamma_norm).view(film.size(0), -1, 1)
            beta4 = beta4.div(beta_norm).view(film.size(0), -1, 1)
            down = gamma4 * down + beta4
            down = torch.transpose(down, 2, 1)

            x = torch.cat((
                x[:, :1, :],
                down,
                x[:, 1:, :]
            ), dim=1)
        else:
            x = torch.cat((
                x[:, :1, :],
                x[:, (1+self.dap_config.NUM_DAP_TOKENS):, :]
            ), dim=1)

            x_norm = self.dap_norm(x)
            x_tran = torch.transpose(x_norm, 2, 1)
            down = self.dap_downsample(x_tran)

            film = self.dap_film(task_id_estimated_emb)
            gamma4 = film[:, :self.config.hidden_size]
            beta4 = film[:, self.config.hidden_size:]
            gamma_norm = gamma4.norm(p=2, dim=1, keepdim=True).detach()
            beta_norm = beta4.norm(p=2, dim=1, keepdim=True).detach()

            gamma4 = gamma4.div(gamma_norm).view(film.size(0), -1, 1)
            beta4 = beta4.div(beta_norm).view(film.size(0), -1, 1)
            down = gamma4 * down + beta4
            down = torch.transpose(down, 2, 1)

            if not (layer_index == 11 and cfg.DATA.NAME == 'imagenet_r'):
            # for imagenet_r, do not append prompts on the last layer
                x = torch.cat((
                    x[:, :1, :],
                    down,
                    x[:, 1:, :]
                ), dim=1)

        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)

        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class ADPT_Encoder(nn.Module):
    def __init__(self, config, vis, dap_cfg):
        super(ADPT_Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

        self.num_layers = config.transformer["num_layers"]
        for _ in range(self.num_layers):
            layer = ADPT_Block(config, vis, dap_cfg)
            self.layer.append(copy.deepcopy(layer))

    def forward_films(self, task_id_emb):
        films = []
        for layer_block in self.layer:
            films.append(layer_block.dap_film(task_id_emb))
        return films

    def forward(self, hidden_states, task_id_estimated_emb=None, cfg=None):
        attn_weights = []
        for layer_index, layer_block in enumerate(self.layer):
            hidden_states, weights = layer_block(hidden_states, task_id_estimated_emb=task_id_estimated_emb, layer_index=layer_index, cfg=cfg)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

def expand_to_batch(x, batch_size, dim=0):
    shape = [1 for _ in x.shape]
    shape.insert(dim, batch_size)
    return torch.tile(torch.unsqueeze(x, dim=dim), shape).cuda()

class ADPT_Transformer(nn.Module):
    def __init__(self, config, img_size, vis, dap_cfg, model_type=None, model_root=None, total_cfg=None):
        super(ADPT_Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = ADPT_Encoder(config, vis, dap_cfg)

        self.pretrained_enc = VisionTransformer(model_type, img_size, num_classes=-1, vis=vis)
        self.pretrained_enc.load_from(np.load(os.path.join(model_root, MODEL_ZOO[model_type])))

        self.patch_size = _pair(config.patches["size"])
        self.prompt_dim = config.hidden_size
        self.pool_size = total_cfg.MODEL.DAP.PROMPT_POOL

        val = math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + self.prompt_dim))
        self.dap_key_embeddings = nn.Parameter(torch.zeros(self.pool_size, self.prompt_dim))
        nn.init.uniform_(self.dap_key_embeddings.data, -val, val)
        self.dap_emb = torch.nn.Embedding(dap_cfg.NUM_TASKS_FOR_EMB, dap_cfg.TASK_EMB)

        self.dap_cfg = dap_cfg
        self.cfg = total_cfg
        self.top_k = 1

    def forward(self, input_ids, task_id=None, is_train=None, cfg=None):
        B = input_ids.shape[0]
        x_cls_embed = self.pretrained_enc.forward_last_cls(input_ids).detach()

        if is_train:
            start = task_id * self.top_k
            end = (task_id + 1) * self.top_k
            prompt_mask = torch.arange(start, end).cuda()
            if end > self.pool_size:
                prompt_mask = None
        else:
            prompt_mask = None

        dap_prompt_key_norm = F.normalize(self.dap_key_embeddings, dim=-1)

        x_embed_norm = F.normalize(x_cls_embed, dim=-1)
        sim = torch.matmul(dap_prompt_key_norm,
                           torch.transpose(x_embed_norm, 1, 0))

        sim = torch.transpose(sim, 1, 0)
        (sim_top_k, idx) = torch.topk(sim, self.top_k)
        idx = idx.squeeze(dim=-1)

        prompt_id, id_counts = torch.unique(idx, return_counts=True)
        _, major_idx = torch.topk(id_counts, self.top_k)
        major_prompt_id = prompt_id[major_idx]
        idx = expand_to_batch(major_prompt_id, x_cls_embed.shape[0]).squeeze(dim=-1)

        task_id = major_prompt_id.cpu()[0]

        if prompt_mask is not None:
            idx = prompt_mask
            task_id = idx.cpu()[0]
            idx = expand_to_batch(idx, x_cls_embed.shape[0]).squeeze(dim=-1)

        task_id_estimated_emb = self.dap_emb(idx)

        i = torch.arange(B).reshape(B, 1, 1)
        l = torch.arange(self.prompt_dim).reshape(1, 1, self.prompt_dim)

        selected_prompt_key = dap_prompt_key_norm.repeat(B, 1, 1)[
            i, idx.unsqueeze(-1), l]

        x_embed_norm = x_embed_norm.unsqueeze(1)
        sim_pull = selected_prompt_key * x_embed_norm
        reduce_sim = torch.sum(sim_pull) / x_cls_embed.shape[0]

        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output, task_id_estimated_emb=task_id_estimated_emb, cfg=cfg)

        return encoded, attn_weights, reduce_sim, task_id


class ADPT_VisionTransformer(nn.Module):
    def __init__(
            self, model_type,
            img_size=224, num_classes=21843, vis=False, dap_cfg=None, model_root=None, total_cfg=None
    ):
        super(ADPT_VisionTransformer, self).__init__()
        config = CONFIGS[model_type]
        self.num_classes = num_classes
        self.classifier = config.classifier

        self.transformer = ADPT_Transformer(config, img_size, vis, dap_cfg, model_type=model_type, model_root=model_root, total_cfg=total_cfg)
        self.head = Linear(config.hidden_size, num_classes) if num_classes > 0 else nn.Identity()

        self.cfg = total_cfg
        self.top_k = 1

    def forward(self, x, task_id, vis=False, is_train=None):
        x, attn_weights, reduce_sim, task_id_out = self.transformer(x, task_id, is_train=is_train, cfg=self.cfg)

        logits = self.head(x[:, 0])

        if not vis:
            return logits, reduce_sim, task_id_out
        return logits, attn_weights, reduce_sim, task_id_out

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)