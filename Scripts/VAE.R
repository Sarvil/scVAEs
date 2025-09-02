library(torch)
library(luz)
library(Matrix)
library(ggplot2)

gc()
rm(list = ls())

device <- torch_device(
  if (torch::backends_mps_is_available()) {
    "mps"
  } else {
    "cpu"
  }
)
device

# Custom KL divergence for sparsity
kl_divergence <- function(rho, rho_hat) {
  rho * torch_log(rho / rho_hat) + 
    (1 - rho) * torch_log((1 - rho) / (1 - rho_hat))
}

# Sparse Autoencoder Module
sparse_autoencoder <- nn_module(
  "SparseAutoencoder",
  
  initialize = function(input_dim, hidden_dim, rho = 0.05, beta = 1) {
    self$encoder <- nn_linear(input_dim, hidden_dim)
    self$decoder <- nn_linear(hidden_dim, input_dim)
    
    self$rho <- rho
    self$beta <- beta
  },
  
  forward = function(x) {
    h <- torch_sigmoid(self$encoder(x))
    x_hat <- torch_sigmoid(self$decoder(h))
    list(output = x_hat, hidden = h)
  },
  
  loss = function(x, x_hat, hidden) {
    # Reconstruction loss (MSE)
    recon_loss <- nnf_mse_loss(x_hat, x)
    
    # Mean activation of hidden units over batch
    rho_hat <- torch_mean(hidden, dim = 1)
    
    # Sparsity penalty
    sparsity_loss <- torch_sum(kl_divergence(self$rho, rho_hat))
    
    # Total loss
    recon_loss + self$beta * sparsity_loss
  }
)

# Dummy data: 1000 samples, 20 features
x <- torch_randn(1000, 20)
dataset <- tensor_dataset(x)
dataloader <- dataloader(dataset, batch_size = 64, shuffle = TRUE)

# Model, optimizer
model <- sparse_autoencoder(input_dim = 20, hidden_dim = 10, rho = 0.05, beta = 0.5)
optimizer <- optim_adam(model$parameters, lr = 0.01)

# Training loop
for (epoch in 1:50) {
  total_loss <- 0
  coro::loop(for (batch in dataloader) {
    optimizer$zero_grad()
    
    out <- model(batch[[1]])
    loss <- model$loss(batch[[1]], out$output, out$hidden)
    
    loss$backward()
    optimizer$step()
    
    total_loss <- total_loss + loss$item()
  })
  cat(sprintf("Epoch %d: Loss = %.4f\n", epoch, total_loss))
}

out <- model(x)
latent <- out$hidden

library(uwot)
latent_cpu <- as.matrix(latent$to(device = "cpu"))
umap_result <- umap(latent_cpu)
plot(umap_result, col = 'blue', pch = 16)
