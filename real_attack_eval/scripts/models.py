"""
Centralized Model Registry: All Custom PyTorch Architectures
=============================================================
SYNCHRONIZED with optimize_path_a.py, optimize_path_b.py, optimize_path_c.py
These are the EXACT architectures used in successful sweep experiments.

Models are organized by detection path:
 - Path B (Topological Structure): DenseAE, LSTMAE, VAE, Conv1DAE, DeepSVDD
 - Path C (Temporal Chronology): CBOWPredictor, GRUPredictor, LSTMAESequence
"""

import torch
import torch.nn as nn

# Resolve device once for models that reference it internally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# PATH B: TOPOLOGICAL STRUCTURE MODELS (Spatial Geometry)
# Source: scripts/archive/optimization/optimize_path_b.py


class DenseAE(nn.Module):
    """
    Dense Autoencoder used by the Path B sweep.

    Architecture:
    input, Linear(mid), ReLU, Linear(latent_dim)
    Linear(mid), ReLU, Linear(input_dim), Sigmoid

    where mid = max(latent_dim * 2, 32)
    """
    def __init__(self, input_dim=20, latent_dim=16):
        super().__init__()
        mid = max(latent_dim * 2, 32)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid), nn.ReLU(),
            nn.Linear(mid, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, mid), nn.ReLU(),
            nn.Linear(mid, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class LSTMAE(nn.Module):
    """
 LSTM Autoencoder - EXACT architecture from optimize_path_b.py

 Architecture:
    input (B, seq_len, 1), LSTM encoder, hidden state,
    LSTM decoder, reconstruction (B, seq_len, 1)
 """
    def __init__(self, seq_len=20, hidden_dim=16):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(1, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(1, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        h = hidden[-1].unsqueeze(1)
        dummy = torch.zeros(x.size(0), self.seq_len, 1, device=x.device)
        decoded, _ = self.decoder(
            dummy,
            (h.transpose(0, 1).contiguous(), torch.zeros_like(h.transpose(0, 1).contiguous()))
        )
        return self.sigmoid(self.out(decoded))


class VAE(nn.Module):
    """
 Variational Autoencoder - EXACT architecture from optimize_path_b.py

 Architecture:
    input, Linear(32), mu/logvar, reparameterize, Linear(32), output
 """
    def __init__(self, input_dim=20, latent_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 32)
        self.fc4 = nn.Linear(32, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return torch.sigmoid(self.fc4(torch.relu(self.fc3(z))))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Conv1DAE(nn.Module):
    """
 Conv1D Autoencoder - EXACT architecture from optimize_path_b.py

 Architecture:
 Embedding(vocab, embed_dim)
    Conv1d(embed_dim to ch), ReLU, MaxPool1d(2)
    Conv1d(ch to ch//2), ReLU, MaxPool1d(2)
    ConvTranspose1d(ch//2 to ch), ReLU
    ConvTranspose1d(ch to embed_dim)

 where ch = max(embed_dim, 8)
 """
    def __init__(self, vocab_size, embed_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        ch = max(embed_dim, 8)
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, ch, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(ch, max(ch // 2, 4), kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(max(ch // 2, 4), ch, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose1d(ch, embed_dim, kernel_size=2, stride=2)
        )

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)   # (B, embed_dim, 20)
        enc = self.encoder(emb)
        dec = self.decoder(enc)
        return dec, emb


class DeepSVDD(nn.Module):
    """
 Deep SVDD - EXACT architecture from optimize_path_b.py

 Architecture:
    input, Linear(64), ReLU, Linear(32), ReLU, Linear(hidden)
 """
    def __init__(self, input_dim=20, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, hidden)
        )

    def forward(self, x):
        return self.net(x)



# PATH C: TEMPORAL CHRONOLOGY MODELS (Sequence Grammar)
# Source: scripts/archive/optimization/optimize_path_c.py


class CBOWPredictor(nn.Module):
    """
 CBOW Predictor - EXACT architecture from optimize_path_c.py

 Architecture:
    context (B, 10), embedding, flatten, Linear(vocab_size)
 """
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)
        self.fc = nn.Linear(embed_dim * 10, vocab_size + 1)

    def forward(self, x):
        emb = self.embedding(x)       # (B, 10, E)
        emb = emb.view(x.size(0), -1) # Flatten: (B, 10*E)
        return self.fc(emb)


class GRUPredictor(nn.Module):
    """
 GRU Predictor - EXACT architecture from optimize_path_c.py

 Architecture:
    input (B, seq_len), embedding, GRU(embed_dim, hidden_dim, num_layers),
    Linear(hidden_dim, vocab_size)
 """
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)
        self.gru = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size + 1)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        return self.fc(out)  # (B, seq_len, vocab_size)


class LSTMAESequence(nn.Module):
    """
 LSTM Autoencoder Sequence - EXACT architecture from optimize_path_c.py

 Architecture:
    input (B, W), embedding, LSTM encoder, hidden state,
    LSTM decoder, Linear, output (B, W, vocab)
 """
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, embed_dim, batch_first=True)
        self.out = nn.Linear(embed_dim, vocab_size + 1)

    def forward(self, x):
        e = self.embed(x)
        _, (h, c) = self.encoder(e)
        # Repeat hidden state for each timestep
        repeated = h.permute(1, 0, 2).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(repeated)
        return self.out(dec_out)

