library(torch)
library(luz)
library(dplyr)
library(ggplot2)
library(Matrix)
library(umap)

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

encoder_gen <- nn_module(
  classname = "Encoder",
  initialize = function(dim.input, dim.l1, dim.l2, dim.latent) {
    self$compressor <- nn_sequential(
      nn_linear(dim.input, dim.l1),
      nn_dropout(p = 0.5),
      nn_relu(),
      nn_linear(dim.l1, dim.l2),
      nn_dropout(0.5),
      nn_relu(),
    )
  },
  forward = function(x) {
    x <- x$to(device = device)
    cat("Shape of Input Layer: ", x$shape, "\n")
    shared_layer <- self$compressor(x)
    cat("Shape of Shared Layer: ", shared_layer$shape, "\n")
    shared_layer
  }
)

decoder_gen <- nn_module(
  classname = "Decoder",
  initialize = function(dim.latent, dim.l2, dim.l1, dim.input) {
    self$decompressor <- nn_sequential(
      nn_linear(dim.latent, dim.l2),
      nn_dropout(p = 0.5),
      nn_relu(),
      nn_linear(dim.l2, dim.l1),
      nn_dropout(p = 0.5),
      nn_relu(),
      nn_linear(dim.l1, dim.input)
    )
  },
  forward = function(x) {
    x_pred = self$decompressor(x)
    
    x_pred
  }
)

vae_gen <- nn_module(
  classname = "VAE",
  initialize = function(dim.input, dim.l1 = 1024, dim.l2 = 128, dim.latent = 25, param.beta = 0.1) {
    self$encoder <- encoder_gen(dim.input, dim.l1, dim.l2, dim.latent)
    self$mean <- nn_linear(dim.l2, dim.latent)
    self$log_var <- nn_linear(dim.l2, dim.latent)
    self$decoder <- decoder_gen(dim.latent, dim.l2, dim.l1, dim.input)
    self$latent_dim <- dim.latent
    self$param_beta <- param.beta
  },
  forward = function(x) {
    shared_layer <- self$encoder(x)
    self$mu <- self$mean(shared_layer)
    self$log_v <- self$log_var(shared_layer)
    self$log_v <- torch_clamp(self$log_v, min = -10, max = 10)
    z <- self$mu + torch_exp(self$log_v$mul(0.5)) * torch_randn(c(dim(x)[1], self$latent_dim))$to(device = device)
    self$z <- z
    self$x_hat <- self$decoder(z)

    self$x_hat
  },
  loss = function(pred, target) {
    recon_loss <- nnf_mse_loss(target, pred, reduction = "sum")
    kl_loss <- -0.5 * torch_sum(1 + self$log_v - self$mu$pow(2) - self$log_v$exp())
    
    self$.kl_loss <- kl_loss$item()
    self$.recon_loss <- recon_loss$item()
    
    recon_loss + kl_loss * self$param_beta
  }
)

metrics_track_callback <- luz_callback(
  classname = "trackKLReconLoss",
  initialize = function() {
    self$kl_loss = c()
    self$recon_loss = c()
  },
  on_train_batch_end = function() {
    self$kl_loss <- c(self$kl_loss, ctx$model$.kl_loss)
    self$recon_loss <- c(self$recon_loss, ctx$model$.recon_loss)
  },
  on_valid_batch_end = function() {
    self$kl_loss <- c(self$kl_loss, ctx$model$.kl_loss)
    self$recon_loss <- c(self$recon_loss, ctx$model$.recon_loss)
  },
  on_train_end = function() {
    kl_mean <- mean(self$kl_loss)
    recon_mean <- mean(self$recon_loss)
    ctx$log_metric("kl_loss", as.numeric(kl_mean))
    ctx$log_metric("recon_loss", as.numeric(recon_mean))
    cat("KL_Loss: ", kl_mean, "Recon Loss: ", recon_mean, "\n")
    self$kl_loss = c()
    self$recon_loss = c()
  },
  on_valid_end = function() {
    kl_mean <- mean(self$kl_loss)
    recon_mean <- mean(self$recon_loss)
    ctx$log_metric("kl_loss", as.numeric(kl_mean))
    ctx$log_metric("recon_loss", as.numeric(recon_mean))
    cat("KL_Loss: ", kl_mean, "Recon Loss: ", recon_mean, "\n")
    self$kl_loss = c()
    self$recon_loss = c()
  }
)

raw_data <- readMM("~/Documents/R/filtered_gene_bc_matrices/hg19/matrix.mtx")
raw_data <- as.matrix(raw_data)
raw_data <- log1p(raw_data)

batch_size = 32

raw_tensor <- torch_tensor(raw_data, dtype = torch_float32(), device = device)
ds <- tensor_dataset(raw_tensor, raw_tensor)
dl <- dataloader(ds, batch_size = batch_size, shuffle = TRUE)

train_ids <- sample(1:length(ds), size = 0.6 * length(ds))
valid_ids <- sample(setdiff(1:length(ds), train_ids), size = 0.2 * length(ds))
test_ids <- setdiff(1:length(ds), union(train_ids, valid_ids))

train_ds <- dataset_subset(ds, indices = train_ids)
valid_ds <- dataset_subset(ds, indices = valid_ids)
test_ds <- dataset_subset(ds, indices = test_ids)

train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = batch_size)
test_dl <- dataloader(test_ds, batch_size = batch_size)


input_dim = ncol(raw_data)
num_cells = nrow(raw_data)
latent1_dim = 1024
latent2_dim = 128
latent_dim = 25
num_epochs = 200
learning_rate = 0.01
beta_param = 0.05

model <- vae_gen %>%
  setup(
    optimizer = optim_adam,
    ) %>%
  set_opt_hparams(lr = 1e-4) %>%
  set_hparams(
    dim.input = input_dim, dim.l1 = latent1_dim, dim.l2 = latent2_dim, dim.latent = latent_dim, param.beta = beta_param
  )

lr_finder_rate <- lr_finder(model, train_dl, steps = num_epochs, start_lr = 1e-07, end_lr = 0.3)
plot(lr_finder_rate)

fitted <- model %>%
  fit(train_dl, epochs = num_epochs, valid_data = valid_dl,
  callbacks = list(
    metrics_track_callback(),
    luz_callback_early_stopping(patience = 25),
    luz_callback_lr_scheduler(lr_one_cycle,
                              max_lr = (learning_rate*1.5),
                              epochs = num_epochs,
                              steps_per_epoch = length(train_dl),
                              call_on = "on_batch_end",
                              )
  ),
  verbose = TRUE
)

luz_save(fitted, "~/Documents/ISM/PhD/3rd Year/Results/vae_model_weights.pt")

fitted <- luz_load("~/Documents/ISM/PhD/3rd Year/Results/vae_model_weights.pt")

get_latents = function(model, dataloader) {
  model$model$to(device = device)
  model$model$eval()
  latents <- list()
  coro::loop(for (batch in dataloader) {
    x <- batch[[1]]
   with_no_grad({
     out <- model$model$forward(x)
     latents[[length(latents) + 1]] <- model$model$z %>% as_array()
   })
  })
  latent_mat <- do.call(rbind, latents)
  rownames(latent_mat) <- NULL
  return(latent_mat)
}

# 1. Extract all latent means as before
latent_25d <- get_latents(fitted, dl)
latent_25d <- as.matrix(latent_25d)
plot(latent_25d)

set.seed(123)
k <- 10
kmeans_res <- kmeans(latent_25d, centers = k, iter.max = 1e7)

cluster_labels <- factor(kmeans_res$cluster)

umap_res <- umap(latent_25d, n_neighbors = 5, min_dist = 0.1)
df_umap <- data.frame(
  UMAP1 = umap_res$layout[,1],
  UMAP2 = umap_res$layout[,2],
  cluster = cluster_labels
)

ggplot(df_umap, aes(x = UMAP1, y = UMAP2, color = cluster)) +
  geom_point(alpha = 0.7, size = 1) +
  theme_minimal()
