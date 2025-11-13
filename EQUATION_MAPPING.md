# NECST Code-to-Equation Mapping

This document maps the code implementation to the equations in the paper "Neural Joint Source-Channel Coding" (https://arxiv.org/abs/1811.07557) by Choi et al., ICML 2019.

## Overview

NECST (Neural Error Correcting and Source Trimming) performs joint source-channel coding by learning to:
1. **Compress** data into a fixed-length binary code (source coding)
2. **Protect** against channel noise (channel coding)
3. **Reconstruct** the original data from noisy codes

The system is trained end-to-end using discrete latent codes and VIMCO for gradient estimation.

---

## 1. Model Architecture

### 1.1 Encoder q_φ(y|x) - Equation (1) in Paper

**Paper**: The encoder outputs parameters for the distribution q_φ(y|x) where y ∈ {0,1}^k

**Code Location**: `necst.py:150-167`

```python
def encoder(self, x, reuse=True):
    """
    Specifies the parameters for the mean and variance of p(y|x)
    """
    # ... neural network layers ...
    z_mean = tf.layers.dense(e, self.z_dim, activation=None,
                             use_bias=False, kernel_regularizer=regularizer,
                             reuse=reuse, name='fc-'+str(len(enc_layers)-1))
    return z_mean
```

**Mapping**:
- `z_mean` = logits for Bernoulli distribution = W_φ·h(x) + b_φ
- For MNIST: Uses fully connected layers with leaky ReLU activation
- For CelebA/CIFAR10: Uses convolutional encoder (`complex_encoder`, `cifar10_convolutional_encoder`)

**Equation**:
```
q_φ(y|x) = ∏_{i=1}^k Bernoulli(y_i | sigmoid(W_φ·h(x) + b_φ)_i)
```

### 1.2 Channel Model - Equation (2) in Paper

#### Binary Symmetric Channel (BSC)

**Paper**: For BSC with crossover probability ε, the received code z is obtained by flipping each bit in y with probability ε.

**Code Location**: `necst.py:572-619` (specifically lines 596-603)

```python
def create_collapsed_computation_graph(self, x, reuse=False):
    # Encoder output
    mean = self.encoder(x, reuse=reuse)

    # Collapsed channel model
    if self.noise != 0:
        y_hat_prob = tf.nn.sigmoid(mean)
        total_prob = y_hat_prob - (2 * y_hat_prob * self.noise) + self.noise
        q = Bernoulli(probs=total_prob)
    else:
        q = Bernoulli(logits=mean)
```

**Mapping**:
- `self.noise` = ε (channel crossover probability)
- `y_hat_prob` = p(y_i = 1|x) = sigmoid(encoder_output_i)
- `total_prob` = p(z_i = 1|x) after going through noisy channel

**Equation** (Collapsed BSC):
```
p(z_i = 1|x) = p(y_i = 1|x)(1 - ε) + (1 - p(y_i = 1|x))ε
            = p(y_i = 1|x) - 2·p(y_i = 1|x)·ε + ε
```

This is the **key "collapsed" trick** that allows backpropagation through discrete codes by marginalizing out the intermediate y.

#### Binary Erasure Channel (BEC)

**Code Location**: `necst.py:622-676` (specifically lines 642-657)

```python
def create_erasure_collapsed_computation_graph(self, x, reuse=False):
    mean = self.encoder(x, reuse=reuse)
    y_hat_prob = tf.nn.softmax(mean)

    # Construct mask for erasure channel
    mask = np.zeros((2,3))
    mask[0,0] = 1 - self.noise  # 0 -> 0
    mask[0,2] = self.noise      # 0 -> erasure
    mask[1,1] = 1 - self.noise  # 1 -> 1
    mask[1,2] = self.noise      # 1 -> erasure

    total_prob = tf.reshape(tf.reshape(y_hat_prob, [-1, 2]) @ mask,
                           [-1, self.z_dim, 3])
    q = Categorical(probs=total_prob)
```

**Equation** (BEC):
```
p(z_i = 0|x) = p(y_i = 0|x)(1 - ε)
p(z_i = 1|x) = p(y_i = 1|x)(1 - ε)
p(z_i = ?|x) = ε  [erasure symbol]
```

### 1.3 Decoder p_θ(x|z) - Equation (3) in Paper

**Paper**: The decoder reconstructs x from the noisy code z

**Code Location**: `necst.py:432-447`

```python
def decoder(self, z, reuse=True, use_bias=False):
    d = tf.convert_to_tensor(z)
    dec_layers = self.dec_layers

    with tf.variable_scope('model', reuse=reuse):
        with tf.variable_scope('decoder', reuse=reuse):
            for layer_idx, layer_dim in list(reversed(list(enumerate(dec_layers))))[:-1]:
                d = tf.layers.dense(d, layer_dim, activation=tf.nn.leaky_relu,
                                   reuse=reuse, name='fc-' + str(layer_idx),
                                   use_bias=use_bias)
            if self.is_binary:  # directly return logits
                x_reconstr_logits = tf.layers.dense(d, dec_layers[0], activation=None,
                                                   reuse=reuse, name='fc-0', use_bias=use_bias)
            else:  # gaussian decoder
                x_reconstr_logits = tf.layers.dense(d, dec_layers[0],
                                                   activation=self.last_layer_act,
                                                   reuse=reuse, name='fc-0', use_bias=use_bias)

    return x_reconstr_logits
```

**Mapping**:
- For binary data (BinaryMNIST): `p_θ(x|z) = ∏_i Bernoulli(x_i | sigmoid(decoder_logits_i))`
- For continuous data (MNIST): Gaussian decoder with sigmoid activation to bound [0,1]
- For images (CelebA, CIFAR10): Uses convolutional decoder

**Equation**:
```
# Binary decoder:
p_θ(x|z) = ∏_{i=1}^d Bernoulli(x_i | sigmoid(W_θ·g(z) + b_θ)_i)

# Gaussian decoder:
p_θ(x|z) = 𝒩(x | μ_θ(z), σ²I)  where μ_θ(z) = sigmoid(W_θ·g(z) + b_θ)
```

---

## 2. Loss Functions and Training

### 2.1 VIMCO Objective - Equation (4) in Paper

**Paper**: VIMCO (Variational Inference for Monte Carlo Objectives) provides low-variance gradient estimates for discrete latent variables using K samples.

**Code Location**: `necst.py:513-552` (VIMCO loss) and `necst.py:471-510` (VIMCO baseline)

```python
def vimco_loss(self, x, x_reconstr_logits):
    reg_loss = tf.losses.get_regularization_loss()

    # Reconstruction loss for K samples
    if self.is_binary:
        x = tf.expand_dims(x, axis=0)
        x = tf.tile(x, [self.vimco_samples, 1, 1])
        reconstr_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=x_reconstr_logits, labels=x),
            axis=-1)
    else:
        reconstr_loss = tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=-1)

    # Log probability of latent codes
    log_q_h_list = self.q.log_prob(self.z)
    log_q_h = tf.reduce_sum(log_q_h_list, axis=-1)

    # Learning signal
    loss = reconstr_loss

    # VIMCO baseline and importance weights
    local_l, w, full_loss = self.build_vimco_loss(loss)

    # Gradient estimates
    theta_loss = (w * reconstr_loss)
    phi_loss = (local_l * log_q_h) + theta_loss

    # Average over samples and batch
    theta_loss = tf.reduce_mean(tf.reduce_sum(theta_loss, axis=0))
    phi_loss = tf.reduce_mean(tf.reduce_sum(phi_loss, axis=0)) + reg_loss
    full_loss = tf.reduce_mean(full_loss)

    return theta_loss, phi_loss, full_loss
```

**VIMCO Baseline Computation** (`necst.py:471-510`):

```python
def build_vimco_loss(self, l):
    """
    l: Per-sample learning signal [K, batch_size]
    Returns: baseline to subtract from l
    """
    k, b = l.get_shape().as_list()
    kf = tf.cast(k, tf.float32)

    # Multi-sample lower bound
    l_logsumexp = tf.reduce_logsumexp(l, [0], keepdims=True)
    L_hat = l_logsumexp - tf.log(kf)

    # Sum of log f
    s = tf.reduce_sum(l, 0, keepdims=True)

    # Leave-one-out baseline
    diag_mask = tf.expand_dims(tf.diag(tf.ones([k], dtype=tf.float32)), -1)
    off_diag_mask = 1. - diag_mask

    diff = tf.expand_dims(s - l, 0)
    l_i_diag = 1. / (kf - 1.) * diff * diag_mask
    l_i_off_diag = off_diag_mask * tf.stack([l] * k)
    l_i = l_i_diag + l_i_off_diag
    L_hat_minus_i = tf.reduce_logsumexp(l_i, [1]) - tf.log(kf)

    # Importance weights
    w = tf.stop_gradient(tf.exp((l - l_logsumexp)))

    # Control variate
    local_l = tf.stop_gradient(L_hat - L_hat_minus_i)

    return local_l, w, L_hat[0, :]
```

**Equations** (VIMCO):

Multi-sample lower bound:
```
L̂ = log(1/K ∑_{k=1}^K f_k)  where f_k = p_θ(x|z^k) / q_φ(z^k|x)
```

Learning signal:
```
f_k = exp(-reconstr_loss_k) / q_φ(z^k|x)
    = exp(-log p_θ(x|z^k)) / q_φ(z^k|x)
```

Leave-one-out baseline:
```
L̂_{-i} = log(1/(K-1) ∑_{k≠i} f_k)
```

Gradient estimates:
```
∇_θ L̂ ≈ (1/K) ∑_k w_k ∇_θ log p_θ(x|z^k)
∇_φ L̂ ≈ (1/K) ∑_k (L̂ - L̂_{-k}) ∇_φ log q_φ(z^k|x) + w_k ∇_φ log p_θ(x|z^k)
```

where importance weights:
```
w_k = f_k / ∑_j f_j
```

### 2.2 Reconstruction Loss - For Decoder Parameters θ

**Code Location**: `necst.py:450-468` (Gumbel-Softmax version)

```python
def get_loss(self, x, x_reconstr_logits):
    reg_loss = tf.losses.get_regularization_loss()

    if self.is_binary:
        x = tf.expand_dims(x, axis=0)
        x = tf.tile(x, [self.vimco_samples, 1, 1])
        reconstr_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=x_reconstr_logits, labels=x))
    else:
        reconstr_loss = tf.reduce_mean(
            tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=1))

    total_loss = reconstr_loss + reg_loss
    return total_loss, reconstr_loss
```

**Equations**:

Binary cross-entropy (for binary data):
```
ℓ_recon(x, z) = -∑_i [x_i log σ(ŷ_i) + (1-x_i) log(1-σ(ŷ_i))]
              = ∑_i sigmoid_cross_entropy(ŷ_i, x_i)
```
where ŷ = decoder_logits(z)

Mean squared error (for continuous data):
```
ℓ_recon(x, z) = ∑_i (x_i - μ_θ(z)_i)²
```

### 2.3 Regularization Loss

**Code Location**: Throughout encoder/decoder definitions

```python
regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_param)
# Applied to weights in encoder/decoder layers
```

**Equation**:
```
ℓ_reg = λ·∑_W ||W||²_2
```
where λ = `FLAGS.reg_param` (default 0.0001)

### 2.4 Total Training Objective

**Mapping**:
```
Total Loss = VIMCO Lower Bound + Regularization
           = L̂(x; θ, φ) + λ·ℓ_reg
```

Optimized separately:
- **θ (decoder parameters)**: `theta_loss = (w * reconstr_loss)` - updates decoder
- **φ (encoder parameters)**: `phi_loss = (local_l * log_q_h) + theta_loss + reg_loss` - updates encoder

**Code Location**: `necst.py:123-140`

```python
if not self.discrete_relax:
    # Separate optimizers for encoder and decoder
    theta_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/decoder')
    self.discrete_train_op1 = self.theta_optimizer.minimize(
        self.theta_loss, global_step=self.global_step, var_list=theta_vars)

    phi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/encoder')
    self.discrete_train_op2 = self.phi_optimizer.minimize(
        self.phi_loss, global_step=self.global_step, var_list=phi_vars)
```

---

## 3. Training Procedure

### 3.1 Sampling Strategy

**VIMCO Sampling** - K=5 samples during training

**Code Location**: `necst.py:606`

```python
y = tf.cast(q.sample(self.vimco_samples), tf.float32)
```

Default: `FLAGS.vimco_samples = 5`

### 3.2 Training Loop

**Code Location**: `necst.py:810-916` and `necst.py:863-867`

```python
# Training step
if not self.discrete_relax:
    # VIMCO: update both encoder and decoder
    sess.run([self.discrete_train_op1, self.discrete_train_op2], feed_dict)
else:
    # Gumbel-Softmax: single optimizer
    sess.run(self.train_op, feed_dict)
```

**Algorithm**:
```
For each epoch:
    For each batch of data x:
        1. Encode: compute q_φ(z|x) with collapsed channel
        2. Sample: draw K samples {z^(k)} from q_φ(z|x)
        3. Decode: compute p_θ(x|z^(k)) for each sample
        4. Compute VIMCO loss and gradients
        5. Update θ (decoder parameters)
        6. Update φ (encoder parameters)
```

---

## 4. Example: BinaryMNIST with BSC

### Complete Forward Pass

**Input**: Binary MNIST image x ∈ {0,1}^784

**Step 1: Encoding** (`necst.py:150-167`)
```python
mean = encoder(x)  # shape: [batch_size, n_bits]
```
Equation: `logits_φ = W_φ · σ(W_h · x + b_h) + b_φ`

**Step 2: Collapsed Channel** (`necst.py:596-603`)
```python
y_hat_prob = sigmoid(mean)  # p(y_i=1|x)
total_prob = y_hat_prob - 2*y_hat_prob*noise + noise  # p(z_i=1|x)
q = Bernoulli(probs=total_prob)
```
Equation: `p(z_i=1|x) = p(y_i=1|x)(1-ε) + (1-p(y_i=1|x))ε`

**Step 3: Sampling** (`necst.py:606`)
```python
z = q.sample(vimco_samples)  # shape: [K, batch_size, n_bits]
```
Draw K=5 samples from q(z|x)

**Step 4: Decoding** (`necst.py:432-447`)
```python
x_reconstr_logits = decoder(z)  # shape: [K, batch_size, 784]
```
Equation: `ŷ = W_θ · σ(W_d · z + b_d) + b_θ`

**Step 5: Loss Computation** (`necst.py:513-552`)
```python
# Binary cross-entropy
reconstr_loss = sigmoid_cross_entropy(x_reconstr_logits, x)  # [K, batch_size]

# Log probability of samples
log_q_h = sum(log q(z_i|x))  # [K, batch_size]

# VIMCO
local_l, w, full_loss = build_vimco_loss(reconstr_loss)
theta_loss = mean(sum(w * reconstr_loss))
phi_loss = mean(sum(local_l * log_q_h + w * reconstr_loss)) + reg_loss
```

---

## 5. Command-Line to Paper Correspondence

### Model Hyperparameters

| Flag | Default | Paper Notation | Description |
|------|---------|----------------|-------------|
| `--n_bits` | 100 | k | Number of bits in latent code |
| `--noise` | 0.1 | ε | Channel noise probability (training) |
| `--test_noise` | 0.1 | ε_test | Channel noise probability (test) |
| `--channel_model` | 'bsc' | - | BSC or BEC |
| `--vimco_samples` | 5 | K | Number of VIMCO samples |
| `--enc_arch` | '500' | - | Encoder hidden layers |
| `--dec_arch` | '500,500' | - | Decoder hidden layers |
| `--reg_param` | 0.0001 | λ | L2 regularization weight |
| `--lr` | 0.001 | α | Learning rate |
| `--batch_size` | 100 | B | Minibatch size |

### Example from README

```bash
python3 main.py --datadir=./data --datasource=BinaryMNIST \
    --channel_model=bsc --noise=0.1 --test_noise=0.1 \
    --n_bits=100 --is_binary=True
```

**Corresponds to**:
- Dataset: Binary MNIST (x ∈ {0,1}^784)
- Latent code: z ∈ {0,1}^100 (compression ratio: 784/100 ≈ 7.8x)
- Channel: BSC with ε = 0.1
- Training: 5-sample VIMCO with Adam optimizer

---

## 6. Key Implementation Details

### 6.1 Collapsed Distribution Trick

**Key Innovation**: Instead of explicitly sampling y ~ q_φ(y|x) and then z ~ p(z|y,ε), the code marginalizes out y to directly sample z ~ p(z|x,ε).

**Benefit**: Allows gradient flow through the discrete channel noise.

**Mathematical Derivation**:
```
p(z_i = 1 | x) = ∑_{y_i ∈ {0,1}} p(z_i=1 | y_i, ε) · q_φ(y_i | x)
               = p(z_i=1 | y_i=0, ε) · q_φ(y_i=0 | x) + p(z_i=1 | y_i=1, ε) · q_φ(y_i=1 | x)
               = ε · (1 - p_φ(x)) + (1-ε) · p_φ(x)
               = p_φ(x) - 2·p_φ(x)·ε + ε
```
where p_φ(x) = sigmoid(encoder_output).

**Code**: `necst.py:598`

### 6.2 VIMCO Implementation

The VIMCO implementation follows Mnih & Rezende (2016):
- Uses K=5 samples (default)
- Computes leave-one-out control variates for variance reduction
- Separate gradient computation for encoder (φ) and decoder (θ)

**Reference**: Mnih, A., & Rezende, D. (2016). Variational inference for Monte Carlo objectives. ICML.

### 6.3 Test-Time Evaluation

**Code Location**: `necst.py:918-973`

Test loss is computed without VIMCO (single sample):
```python
test_loss = get_test_loss(x, test_x_reconstr_logits)
```

Metrics reported:
- L2 squared test loss (per image): `test_loss`
- L2 squared test loss (per pixel): `test_loss / input_dim`
- L2 test loss (per image): `sqrt(test_loss)`
- L2 test loss (per pixel): `sqrt(test_loss) / input_dim`

---

## 7. Architecture Variants

### 7.1 MNIST/BinaryMNIST

- **Encoder**: Fully connected (784 → 500 → 100)
- **Decoder**: Fully connected (100 → 500 → 500 → 784)
- **Code**: `encoder()` and `decoder()` functions

### 7.2 CelebA

- **Encoder**: Convolutional (`complex_encoder`, `necst.py:170-185`)
  - 5 conv layers: 32→32→64→64→256 channels
  - Final FC layer to n_bits
- **Decoder**: Deconvolutional (`complex_decoder`, `necst.py:391-429`)
  - FC layer to 256
  - 5 deconv layers to reconstruct 64×64×3 image

### 7.3 CIFAR-10

- **Encoder**: Convolutional with batch norm (`cifar10_convolutional_encoder`, `necst.py:254-279`)
  - 3 conv+pool layers: 64→32→16 channels
- **Decoder**: Upsampling (`cifar10_convolutional_decoder`, `necst.py:282-352`)
  - 3 deconv+upsample layers

---

## 8. Summary Table: Code Location → Paper Equation

| Paper Component | Equation/Section | Code Location | Line Numbers |
|----------------|------------------|---------------|--------------|
| Encoder q_φ(y\|x) | Eq. (1) | `necst.py:encoder()` | 150-167 |
| BSC Channel Model | Eq. (2) | `necst.py:create_collapsed_computation_graph()` | 596-603 |
| BEC Channel Model | Eq. (2) variant | `necst.py:create_erasure_collapsed_computation_graph()` | 642-657 |
| Decoder p_θ(x\|z) | Eq. (3) | `necst.py:decoder()` | 432-447 |
| VIMCO Objective | Eq. (4) | `necst.py:vimco_loss()` | 513-552 |
| VIMCO Baseline | Eq. (5) | `necst.py:build_vimco_loss()` | 471-510 |
| Reconstruction Loss | Section 3.2 | `necst.py:vimco_loss()` | 516-527 |
| Regularization | Section 3.3 | Throughout encoder/decoder | - |
| Training Algorithm | Algorithm 1 | `necst.py:train()` | 810-916 |
| Gradient Updates | Section 4 | `necst.py:__init__()` | 123-140 |

---

## References

1. Choi, K., Tatwawadi, K., Grover, A., Weissman, T., & Ermon, S. (2019). Neural Joint Source-Channel Coding. ICML 2019.
2. Mnih, A., & Rezende, D. (2016). Variational Inference for Monte Carlo Objectives. ICML 2016.
3. Paper: https://arxiv.org/abs/1811.07557
4. Code: https://github.com/ermongroup/necst
