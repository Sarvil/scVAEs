##### Libraries #####
library(torch)
library(dplyr)
library(ggplot2)
library(luz)
library(Matrix)
library(coro)
library(uwot)

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

##### Data #####
raw_data <- readMM("~/Documents/R/filtered_gene_bc_matrices/hg19/matrix.mtx")
#raw_data <- t(raw_data)
#rownames(raw_data) <- raw_data[,1]
#raw_data <- raw_data[,-1]
raw_data <- as.matrix(raw_data)
raw_data <- log1p(raw_data)
raw_tensor <- torch_tensor(raw_data, device = device, dtype = torch_float32())
ds <- tensor_dataset(raw_tensor, raw_tensor)
dl <- dataloader(ds, batch_size = 128, shuffle = TRUE)

train_ids <- sample(1:length(ds), size = 0.6 * length(ds))
valid_ids <- sample(setdiff(1:length(ds), train_ids), size = 0.2 * length(ds))
test_ids <- setdiff(1:length(ds), union(train_ids, valid_ids))

train_ds <- dataset_subset(ds, indices = train_ids)
valid_ds <- dataset_subset(ds, indices = valid_ids)
test_ds <- dataset_subset(ds, indices = test_ids)

train_dl <- dataloader(train_ds, batch_size = 128, shuffle = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = 128)
test_dl <- dataloader(test_ds, batch_size = 128)

nb_loss <- function(x, mu, theta) {
  theta <- theta$unsqueeze(1)  
  log_prob <- torch_lgamma(theta + x) - torch_lgamma(theta) - torch_lgamma(x + 1) +
    theta * (torch_log(theta) - torch_log(theta + mu)) +
    x * (torch_log(mu) - torch_log(theta + mu))
  
  -(log_prob$sum(dim=2))$mean()  
}

encoder_gen <- nn_module(
  "Encoder",
  initialize = function(dim.input, dim.l1, dim.l2, dim.latent) {
    self$compressor <- nn_sequential(
      nn_linear(dim.input, dim.l1),
      nn_batch_norm1d(dim.l1),
      nn_relu(),
      nn_linear(dim.l1, dim.l2),
      nn_batch_norm1d(dim.l2),
      nn_relu()
    )
    self$mean <- nn_linear(dim.l2, dim.latent)
    self$log_var <- nn_linear(dim.l2, dim.latent)
  },
  forward = function(x) {
    x <- x$to(device=device)
    shared <- self$compressor(x)
    list(mean = self$mean(shared), log_var = self$log_var(shared))
  }
)

decoder_gen <- nn_module(
  "Decoder",
  initialize = function(dim.latent, dim.l2, dim.l1, dim.input) {
    self$decompressor <- nn_sequential(
      nn_linear(dim.latent, dim.l2),
      nn_batch_norm1d(dim.l2),
      nn_relu(),
      nn_linear(dim.l2, dim.l1),
      nn_batch_norm1d(dim.l1),
      nn_relu(),
      nn_linear(dim.l1, dim.input),
      nn_softplus()
    )
    self$dispersion <- nn_parameter(torch_ones(dim.input))
  },
  forward = function(z) {
    mu <- self$decompressor(z)$clamp(min = 1e-4) 
    theta <- self$dispersion$clamp(min=1e-2)  
    list(mu = mu, theta = theta)
  }
)

vae_gen <- nn_module(
  "VAE",
  initialize = function(dim.input, dim.l1=1000, dim.l2=100, dim.latent=25) {
    self$encoder <- encoder_gen(dim.input, dim.l1, dim.l2, dim.latent)
    self$decoder <- decoder_gen(dim.latent, dim.l2, dim.l1, dim.input)
    self$latent_dim <- dim.latent
  },
  forward = function(x) {
    q <- self$encoder(x)
    mu <- q$mean
    log_var <- q$log_var
    std <- torch_exp(0.5 * log_var)
    eps <- torch_randn_like(std)
    z <- mu + eps * std
    self$z <- z
    recon <- self$decoder(z)
    list(mu=mu, log_var=log_var, z=z, recon_mu=recon$mu, recon_theta=recon$theta)
  },
  loss = function(pred, target) {
    recon_loss <- nb_loss(target, pred$recon_mu, pred$recon_theta)
    kl <- -0.5 * torch_sum(1 + pred$log_var - pred$mu$pow(2) - torch_exp(pred$log_var), dim=2)
    kl <- kl$mean()
    beta <- 0.5
    recon_loss + beta * kl
  }
)

model <- vae_gen %>%
  setup(optimizer = optim_adam) %>%
  set_hparams(dim.input = ncol(raw_data),
              dim.l1 = 1000,
              dim.l2 = 100,
              dim.latent = 25)

lr_rate_finder <- lr_finder(model, train_dl, start_lr = 0.0001, end_lr = 0.3)
plot(lr_rate_finder)

fitted <- model %>%
  fit(train_dl, valid_data = valid_dl, epochs = 200,
      callbacks = list(
        luz_callback_early_stopping(patience = 25),
        luz_callback_gradient_clip(max_norm = 1.0),
        luz_callback_lr_scheduler(
          lr_one_cycle,
          max_lr = 0.5,
          epochs = 200,
          steps_per_epoch = length(train_dl)
        ),
      ),
      verbose = TRUE
    )

##### Predict #####
fitted %>% predict(train_dl)
length(train_dl)
fitted$model$z
plot(fitted$model$z)

get_latents <- function(model, dataloader) {
  model$eval()
  latents <- list()
  coro::loop(for (batch in dataloader) {
    x <- batch[[1]]
    with_no_grad({
      out <- model$encoder(x)
      # use 'mean' as latent code
      latents[[length(latents)+1]] <- out$mean$to(device = "cpu") %>% as_array()
    })
  })
  latent_mat <- do.call(rbind, latents)
  rownames(latent_mat) <- NULL
  return(latent_mat)
}

# 1. Extract all latent means as before
latent_25d <- get_latents(fitted$model, dl)

set.seed(123)
k <- 3
kmeans_res <- kmeans(latent_25d, centers = k)

cluster_labels <- factor(kmeans_res$cluster)

set.seed(123)
umap_res <- umap(latent_25d, n_neighbors = 20, min_dist = 0.01)
df_umap <- data.frame(
  UMAP1 = umap_res[,1],
  UMAP2 = umap_res[,2],
  cluster = cluster_labels
)

ggplot(df_umap, aes(x = UMAP1, y = UMAP2, color = cluster)) +
  geom_point(alpha = 0.7, size = 1) +
  theme_minimal()
