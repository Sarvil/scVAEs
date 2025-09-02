library(torch)
library(luz)
library(dplyr)
library(ggplot2)
library(Matrix)
library(umap)
library(magick)  # To read images
library(purrr)
library(stats)        # For kmeans, PCA
library(cluster)      # For silhouette score
library(Rtsne) 

gc()
rm(list = ls())


device <- torch_device(
  if(torch::backends_mps_is_available()) {
    "mps"
  } else {
    "cpu"
  }
)
device

# Encoder module
encoder <- nn_module(
  "Encoder",
  initialize = function(input_dim, hidden_dim, latent_dim) {
    self$fc1 <- nn_linear(input_dim, hidden_dim)
    self$fc_mu <- nn_linear(hidden_dim, latent_dim)
    self$fc_logvar <- nn_linear(hidden_dim, latent_dim)
  },
  forward = function(x) {
    h <- nnf_relu(self$fc1(x))
    mu <- self$fc_mu(h)
    logvar <- self$fc_logvar(h)
    list(mu, logvar)
  }
)

# Decoder module
decoder <- nn_module(
  "Decoder",
  initialize = function(latent_dim, hidden_dim, output_dim) {
    self$fc1 <- nn_linear(latent_dim, hidden_dim)
    self$fc2 <- nn_linear(hidden_dim, output_dim)
  },
  forward = function(z) {
    h <- nnf_relu(self$fc1(z))
    nnf_sigmoid(self$fc2(h))
  }
)

# VAE model
vae <- nn_module(
  "VAE",
  initialize = function(input_dim, hidden_dim, latent_dim) {
    self$encoder <- encoder(input_dim, hidden_dim, latent_dim)
    self$decoder <- decoder(latent_dim, hidden_dim, input_dim)
  },
  reparameterize = function(mu, logvar) {
    std <- torch_exp(0.5 * logvar)
    eps <- torch_randn_like(std)
    mu + eps * std
  },
  forward = function(x) {
    encoded <- self$encoder(x)
    mu <- encoded[[1]]
    logvar <- encoded[[2]]
    z <- self$reparameterize(mu, logvar)
    reconstructed <- self$decoder(z)
    list(reconstructed, mu, logvar)
  }
)

vae_loss <- function(reconstructed, x, mu, logvar) {
  # Binary cross-entropy loss
  recon_loss <- nnf_binary_cross_entropy(reconstructed, x, reduction = "sum")
  # KL divergence
  kl_loss <- -0.5 * torch_sum(1 + logvar - mu$pow(2) - logvar$exp())
  recon_loss + kl_loss
}

set.seed(42)
n <- 1000

df <- data.frame(
  age = runif(n, 18, 80),          # Age in years
  systolic_bp = rnorm(n, 120, 15), # Systolic blood pressure
  diastolic_bp = rnorm(n, 80, 10), # Diastolic blood pressure
  cholesterol = rnorm(n, 200, 30), # Total cholesterol (mg/dL)
  glucose = rnorm(n, 100, 15),     # Fasting glucose (mg/dL)
  bmi = runif(n, 18, 35),          # Body Mass Index
  smoker = rbinom(n, 1, 0.25),     # Current smoker (1/0)
  activity = sample(0:2, n, replace = TRUE), # Physical activity level (0=low, 2=high)
  family_history = rbinom(n, 1, 0.18)        # Family history of disease (1/0)
)

# Construct the outcome variable using the correct references to df columns
df$outcome <- as.numeric(
  0.001 * df$age +
    0.02 * (df$systolic_bp - 120) +
    0.015 * (df$cholesterol - 180) +
    0.03 * df$glucose + 
    0.4 * df$smoker + 
    0.2 * df$family_history - 
    0.15 * df$activity + 
    0.025 * df$bmi + 
    rnorm(n, 0, 0.5) > 2
)

# Normalize features (except for the binary columns and outcome)
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
cols_to_norm <- setdiff(names(df), c("smoker", "family_history", "outcome"))
df[cols_to_norm] <- lapply(df[cols_to_norm], normalize)


X <- as.matrix(df)
X <- torch_tensor(X, dtype = torch_float())
batch_size <- 64
dataset <- tensor_dataset(X)
dataloader <- dataloader(dataset, batch_size = batch_size, shuffle = TRUE)

# Hyperparameters
input_dim <- ncol(X)
hidden_dim <- 128
latent_dim <- 10
learning_rate <- 0.001
epochs <- 100

model <- vae(input_dim, hidden_dim, latent_dim)
optimizer <- optim_adam(model$parameters, lr = learning_rate)

for (epoch in 1:epochs) {
  total_loss <- 0
  coro::loop(for (batch in dataloader) {
    optimizer$zero_grad()
    output <- model(batch[[1]])
    loss <- vae_loss(output[[1]], batch[[1]], output[[2]], output[[3]])
    loss$backward()
    optimizer$step()
    total_loss <- total_loss + loss$item()
  })
  cat(sprintf("Epoch %d | Loss: %.2f\n", epoch, total_loss / length(dataset)))
}

# After training, get the reduced-dimension representation:
encoded_results <- model$encoder(torch_tensor(as.matrix(df), dtype = torch_float()))
latent_mu_matrix <- as_array(encoded_results[[1]])

plot(latent_mu_matrix)

# 1. Get latent representation (mu from encoder)
latent <- as_array(model$encoder(X)[[1]])  # shape: samples x latent_dim

# 2. Clustering - example with K-means
k <- 3  # Choose number of clusters (if known or via methods like elbow)
set.seed(123)
kmeans_res <- kmeans(latent, centers = k)

# 3. Visualize latent space with t-SNE for 2D plotting
tsne_res <- Rtsne(latent, dims = 2, perplexity = 30)
plot(tsne_res$Y, col = kmeans_res$cluster, pch = 19,
     main = "t-SNE Plot of VAE Latent Space Clusters")

luz_save(fitted, "~/Documents/ISM/PhD/3rd Year/Results/vae_model_weights.pt")