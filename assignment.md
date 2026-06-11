# Assignment 6 — Changes Made

## Summary

`assignment_6.ipynb` is a fully implemented version of the skeleton notebook
`karpathy_style_mini_transformer_addition_assignment.ipynb`. The following
changes were made to go from the blank template to a working Tiny GPT that
learns 2-digit integer addition.

---

## 1. Imports & Globals (cell `Ji2K0IRNaBl9`)

- Added `import matplotlib.pyplot as plt` for the loss curve plot
- Added `device = 'cuda' if torch.cuda.is_available() else 'cpu'` (was missing)
- Added GPU version/name print statements for local debugging
- Changed `max_iters` from 3000 → 10000 → 20000 to allow longer training
- Moved `n_layer = 4` into globals (original hardcoded `n_layer = 2` inside `TinyGPT.__init__`)

---

## 2. Milestone 1 — Tokenizer & Dataset

### `generate_example` (cell `g2gCAEVnaBmA`)
- Implemented the function body: samples two integers with `random.randint(0, 99)`,
  computes their sum, returns a formatted string `"{a}+{b}={answer}\n"`
- Removed the trailing unreachable `pass`

### Dataset string (cell `YVzgTDqUaBmD`)
- Filled in the loop body: `train_text += generate_example()`

### Vocabulary / tokenizer (cell `JIspPafIaBmJ`)
- Replaced `chars = None` and `vocab_size = None` with:
  ```python
  chars = sorted(list(set(train_text)))
  vocab_size = len(chars)
  ```

### Convert to tokens (cell `ohV3NwQmaBmL`)
- Implemented: `train_data = encode(train_text)` and
  `data = torch.tensor(train_data, dtype=torch.long)`

### Batch loader (cell `xw0lZOktaBmM`)
- Implemented all three TODO sections:
  - Random start indices via `torch.randint`
  - X sequences via `torch.stack`
  - Y shifted sequences (X offset by 1)

---

## 3. Milestone 2 — Causal Multi-Head Attention

### `Head` (cell `WumQjzzjaBmO`)
- Added `self.key`, `self.query`, `self.value` linear projections
- **Removed `self.proj`** — the original skeleton included an unused per-head
  output projection (`nn.Linear(head_size, n_embd)`) that was commented out in
  `forward`. This added ~17K dead parameters. The output projection belongs in
  `MultiHeadAttention`, not `Head`.
- Implemented `forward`: computes Q, K, V; scales dot-product scores by
  `head_size ** -0.5` (not `C ** -0.5` as the skeleton had); applies causal
  mask; softmax; returns `weights @ v`

### `MultiHeadAttention` (cell `fYMRyM05aBmP`)
- Added `self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])`
- Added `self.proj = nn.Linear(num_heads * head_size, n_embd)`
- Implemented `forward`: concatenates head outputs along last dim, projects back

---

## 4. Milestone 3 — Transformer Block & Model

### `FeedForward` (cell `Xe361Vp1aBmS`)
- Implemented 2-layer MLP:
  ```python
  self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd),
  )
  ```
- `forward` returns `self.net(x)`

### `Block` (cell `Wa-fcglWaBmT`)
- Wired up attention (`self.sa`), feed-forward (`self.ffwd`), and two
  LayerNorms (`self.ln1`, `self.ln2`)
- `forward` applies pre-norm residual connections:
  ```python
  x = x + self.sa(self.ln1(x))
  x = x + self.ffwd(self.ln2(x))
  ```

### `TinyGPT` (cell `AReopFRLaBmU`)
- Added token embedding, positional embedding, blocks, final LayerNorm, lm_head
- `forward` computes `tok_emb + pos_emb`, passes through blocks, computes
  cross-entropy loss when targets are provided
- Removed the locally hardcoded hyperparameters — model now reads `n_embd`,
  `n_head`, `n_layer` from globals

---

## 5. Training Loop (cell `aIiV79Y_aBmW`)

Three improvements over the skeleton's empty loop:

1. **Cosine annealing LR scheduler** — decays learning rate from `1e-3` → `1e-5`
   over `max_iters` steps, preventing the loss plateau at ~1.1 that occurs with
   a fixed learning rate
2. **Gradient clipping** — `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
   placed between `loss.backward()` and `optimizer.step()` to prevent large
   gradient updates from destabilizing training
3. **Loss history** — appends `(step, loss)` to `loss_history` every
   `eval_interval` steps; print statement also shows current LR

---

## 6. Loss Curve Plot (new cell after training loop)

New cell added immediately after the training loop:
```python
steps, losses = zip(*loss_history)
plt.figure(figsize=(10, 4))
plt.plot(steps, losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## 7. Milestone 4 — Generation (cell `XZ15UfSMaBmX`)

- Implemented the two TODO sections:
  - `idx_next = torch.multinomial(probs, num_samples=1)`
  - `idx = torch.cat((idx, idx_next), dim=1)`

---

## Results

| Run | max_iters | Scheduler | Grad clip | Accuracy |
|-----|-----------|-----------|-----------|----------|
| Baseline | 10000 | No | No | 87% |
| + scheduler | 10000 | Cosine | No | 91% |
| + grad clip | 20000 | Cosine | Yes | 92% |

Target: ≥95% accuracy on 2-digit addition problems.
