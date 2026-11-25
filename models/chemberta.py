"""Pretrained ChemBERTa-2 (SMILES Transformer) encoder with a readout head from models/readouts.py."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from models.readouts import build_readout


class SmilesDataset(Dataset):
    def __init__(self, encodings, y):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {
            'input_ids': self.input_ids[i],
            'attention_mask': self.attention_mask[i],
            'y': self.y[i],
        }


class ChemBERTaModel(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()
        self.cfg = cfg
        self._device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
        self.encoder = AutoModel.from_pretrained(cfg.checkpoint).to(self._device)
        hidden_dim = self.encoder.config.hidden_size
        self.readout = build_readout(readout, hidden_dim, n_targets).to(self._device)

        self.unfreeze_epoch = cfg.get('unfreeze_epoch', None)
        start_frozen = bool(cfg.freeze_encoder or (self.unfreeze_epoch is not None and self.unfreeze_epoch > 1))
        self.set_encoder_frozen(start_frozen)

    @property
    def device(self):
        return self._device

    def set_encoder_frozen(self, frozen: bool):
        self.encoder_frozen = frozen
        self.encoder.requires_grad_(not frozen)
        if frozen:
            self.encoder.eval()
        else:
            self.encoder.train()

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'encoder_frozen', self.cfg.freeze_encoder):
            self.encoder.eval()
        return self

    def fit_preprocessing(self, train):
        pass

    def loader(self, frame, targets, batch_size, shuffle=False):
        smiles = frame['SMILES'].astype(str).tolist()
        enc = self.tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors='pt',
        )
        y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)
        return DataLoader(SmilesDataset(enc, y), batch_size=batch_size, shuffle=shuffle)

    def param_groups(self):
        groups = [{'params': list(self.readout.parameters()), 'lr': self.cfg.readout_lr}]
        if not self.cfg.freeze_encoder:
            groups.append({'params': list(self.encoder.parameters()), 'lr': self.cfg.encoder_lr})
        return groups

    def forward(self, batch):
        input_ids = batch['input_ids'].to(self._device)
        attention_mask = batch['attention_mask'].to(self._device)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Attention-masked mean pooling over SMILES token embeddings
        token_emb = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        pooled = torch.sum(token_emb * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return self.readout(pooled)
