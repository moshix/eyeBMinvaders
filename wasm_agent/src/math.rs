use rand::Rng;
use rand_chacha::ChaCha8Rng;

/// A fully-connected linear layer with Adam optimizer state.
pub struct Linear {
    pub weight: Vec<Vec<f32>>,  // [out_features][in_features]
    pub bias: Vec<f32>,         // [out_features]
    // Adam state
    m_weight: Vec<Vec<f32>>,
    v_weight: Vec<Vec<f32>>,
    m_bias: Vec<f32>,
    v_bias: Vec<f32>,
    // Gradient accumulators
    grad_weight: Vec<Vec<f32>>,
    grad_bias: Vec<f32>,
}

impl Linear {
    /// Create a new Linear layer with Kaiming uniform initialization.
    pub fn new(in_features: usize, out_features: usize, rng: &mut ChaCha8Rng) -> Self {
        let bound = (6.0 / in_features as f32).sqrt();
        let weight: Vec<Vec<f32>> = (0..out_features)
            .map(|_| {
                (0..in_features)
                    .map(|_| rng.gen_range(-bound..bound))
                    .collect()
            })
            .collect();
        let bias = vec![0.0; out_features];

        let zero_w = || vec![vec![0.0f32; in_features]; out_features];
        let zero_b = || vec![0.0f32; out_features];

        Self {
            weight,
            bias,
            m_weight: zero_w(),
            v_weight: zero_w(),
            m_bias: zero_b(),
            v_bias: zero_b(),
            grad_weight: zero_w(),
            grad_bias: zero_b(),
        }
    }

    pub fn in_features(&self) -> usize {
        if self.weight.is_empty() { 0 } else { self.weight[0].len() }
    }

    pub fn out_features(&self) -> usize {
        self.weight.len()
    }

    /// Forward pass: y = Wx + b, optionally followed by ReLU.
    pub fn forward(&self, x: &[f32], relu: bool) -> Vec<f32> {
        let out_f = self.out_features();
        let mut y = Vec::with_capacity(out_f);
        for i in 0..out_f {
            let mut sum = self.bias[i];
            for (j, &xj) in x.iter().enumerate() {
                sum += self.weight[i][j] * xj;
            }
            if relu && sum < 0.0 {
                sum = 0.0;
            }
            y.push(sum);
        }
        y
    }

    /// Backward pass: computes grad_input and accumulates grad_weight/grad_bias.
    /// `x` is the input that was fed to forward.
    /// `grad_output` is dL/dy of shape [out_features].
    /// If `relu` was used, `relu_mask` is derived from the forward output.
    /// We need the pre-relu output to know which units were active.
    /// For simplicity, we pass the forward output and apply the mask here.
    pub fn backward(&mut self, x: &[f32], grad_output: &[f32], forward_output: &[f32], relu: bool) -> Vec<f32> {
        let out_f = self.out_features();
        let in_f = self.in_features();

        // Apply ReLU gradient mask
        let grad_post: Vec<f32> = if relu {
            grad_output.iter().enumerate().map(|(i, &g)| {
                if forward_output[i] > 0.0 { g } else { 0.0 }
            }).collect()
        } else {
            grad_output.to_vec()
        };

        // Accumulate grad_weight and grad_bias
        for i in 0..out_f {
            self.grad_bias[i] += grad_post[i];
            for j in 0..in_f {
                self.grad_weight[i][j] += grad_post[i] * x[j];
            }
        }

        // Compute grad_input = W^T * grad_post
        let mut grad_input = vec![0.0f32; in_f];
        for i in 0..out_f {
            for j in 0..in_f {
                grad_input[j] += self.weight[i][j] * grad_post[i];
            }
        }
        grad_input
    }

    /// Adam optimizer step. Applies accumulated gradients then zeros them.
    pub fn update(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u32) {
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);

        let out_f = self.out_features();
        let in_f = self.in_features();

        for i in 0..out_f {
            // Bias
            self.m_bias[i] = beta1 * self.m_bias[i] + (1.0 - beta1) * self.grad_bias[i];
            self.v_bias[i] = beta2 * self.v_bias[i] + (1.0 - beta2) * self.grad_bias[i] * self.grad_bias[i];
            let m_hat = self.m_bias[i] / bc1;
            let v_hat = self.v_bias[i] / bc2;
            self.bias[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
            self.grad_bias[i] = 0.0;

            // Weights
            for j in 0..in_f {
                let g = self.grad_weight[i][j];
                self.m_weight[i][j] = beta1 * self.m_weight[i][j] + (1.0 - beta1) * g;
                self.v_weight[i][j] = beta2 * self.v_weight[i][j] + (1.0 - beta2) * g * g;
                let m_hat = self.m_weight[i][j] / bc1;
                let v_hat = self.v_weight[i][j] / bc2;
                self.weight[i][j] -= lr * m_hat / (v_hat.sqrt() + epsilon);
                self.grad_weight[i][j] = 0.0;
            }
        }
    }

    /// Compute L2 norm of all accumulated gradients.
    pub fn grad_norm_sq(&self) -> f32 {
        let mut norm = 0.0f32;
        for row in &self.grad_weight {
            for &g in row {
                norm += g * g;
            }
        }
        for &g in &self.grad_bias {
            norm += g * g;
        }
        norm
    }

    /// Scale all accumulated gradients by a factor.
    pub fn scale_grads(&mut self, factor: f32) {
        for row in &mut self.grad_weight {
            for g in row.iter_mut() {
                *g *= factor;
            }
        }
        for g in &mut self.grad_bias {
            *g *= factor;
        }
    }

    /// Zero all accumulated gradients.
    pub fn zero_grad(&mut self) {
        for row in &mut self.grad_weight {
            for g in row.iter_mut() {
                *g = 0.0;
            }
        }
        for g in &mut self.grad_bias {
            *g = 0.0;
        }
    }

    /// Get weights as a flat vector (row-major) for serialization.
    pub fn weights_flat(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.out_features() * self.in_features());
        for row in &self.weight {
            v.extend_from_slice(row);
        }
        v
    }

    /// Load weights from a flat vector (row-major).
    pub fn load_weights_flat(&mut self, data: &[f32]) {
        let in_f = self.in_features();
        for (i, row) in self.weight.iter_mut().enumerate() {
            let start = i * in_f;
            row.copy_from_slice(&data[start..start + in_f]);
        }
    }

    /// Load bias from a slice.
    pub fn load_bias(&mut self, data: &[f32]) {
        self.bias.copy_from_slice(data);
    }
}

/// Softmax over a slice, returns probabilities summing to 1.
pub fn softmax(x: &[f32]) -> Vec<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Log-softmax over a slice, returns log-probabilities.
pub fn log_softmax(x: &[f32]) -> Vec<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = x.iter().map(|&v| v - max_val).collect();
    let log_sum_exp = shifted.iter().map(|&v| v.exp()).sum::<f32>().ln();
    shifted.iter().map(|&v| v - log_sum_exp).collect()
}
