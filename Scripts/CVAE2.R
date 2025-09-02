library(torch)
library(torchvision)
library(luz)
library(dplyr)
library(ggplot2)
library(magick)

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

images_size = 224 # resize all images into this size of square image.
images_batch_size = 32

image_dataset <- torch::dataset(
  name = "image_dataset",
  initialize = function(root_dir, img_size = 64, transform = NULL) {
    self$root_dir <- root_dir
    self$img_size <- img_size
    self$transform <- transform
    self$classes <- c("Normal", "OSCC")
    files <- list.files(root_dir, recursive = TRUE, full.names = TRUE)
    files <- files[grepl("\\.(jpg|jpeg|png)$", files, ignore.case = TRUE)]
    self$images <- files
    self$labels <- sapply(files, function(x) {
      if (grepl("Normal", x, ignore.case = TRUE)) {
        0
      } else {
        1
      }
    })
  },
  .getitem = function(index) {
    img_path <- self$images[index]
    img <- image_read(img_path)
    #/Users/sarvilmunipally/Documents/R/train/Normal/aug_95_6757.jpg
    img <- transform_to_tensor(img = img) %>%
      (function(x) x$to(device = device))
    img <- transform_resize(img, c(self$img_size, self$img_size))
    img <- transform_convert_image_dtype(img, dtype = torch::torch_float())
    if (!is.null(self$transform)) {
      img <- self$transform(img)
    }
    label <- torch_tensor(
      as.integer(self$labels[index]),
      dtype = torch_long(),
      device = device
    )
    list(x = img, y = img) # REMEMBER AND CHANGE THIS to list(x = img, y = label)
  },
  .length = function() {
    length(self$images)
  },
  .display = function(index) {
    img_path <- self$images[index]
    img <- image_read(img_path)
    img <- transform_to_tensor(img = img) %>%
      (function(x) x$to(device = device))
    img <- transform_resize(img, c(self$img_size, self$img_size))
    img <- transform_convert_image_dtype(img, dtype = torch::torch_uint8())
    tensor_image_browse(img)
  },
  .shape = function(index) {
    img_path <- self$images[index]
    img <- image_read(img_path)
    img <- transform_to_tensor(img = img) %>%
      (function(x) x$to(device = device))
    img <- transform_resize(img, c(self$img_size, self$img_size))
    img <- transform_convert_image_dtype(img, dtype = torch::torch_uint8())
    img$shape
  }
)

train_path <-  "/Users/sarvilmunipally/Documents/R/train/"
train_ds <-  image_dataset(train_path, img_size = images_size)
train_dl <- dataloader(train_ds, shuffle = TRUE, batch_size = images_batch_size)

valid_path <-  "/Users/sarvilmunipally/Documents/R/val/"
test_path <-  "/Users/sarvilmunipally/Documents/R/test\ 2/"

valid_ds <-  image_dataset(valid_path, img_size = images_size)
test_ds <-  image_dataset(test_path, img_size = images_size)

valid_dl <- dataloader(valid_ds, batch_size = images_batch_size)
test_dl <- dataloader(test_ds, batch_size = images_batch_size)

view <- nn_module(
  classname = "View",
  initialize = function(shape) {
    self$shape <-  shape
  },
  forward = function(x) {
    x$view(self$shape)
  }
)

encoder_gen <- nn_module(
  classname = "encoder_gen",
  initialize = function(dim.input) {
    self$compressor <- nn_sequential(
      nn_conv2d(dim.input, dim.input * 2, kernel_size = 3, stride = 2, padding = 1),
      nn_batch_norm2d(dim.input * 2),
      nn_leaky_relu(),
      nn_conv2d(dim.input * 2, dim.input * 4, kernel_size = 3, stride = 2, padding = 1),
      nn_batch_norm2d(dim.input * 4),
      nn_leaky_relu(),
      nn_conv2d(dim.input * 4, dim.input * 8, kernel_size = 3, stride = 2, padding = 1),
      nn_batch_norm2d(dim.input * 8),
      nn_leaky_relu()
    )
  },
  forward = function(x) {
    x <- x$to(device = device)
    shared_layer <- self$compressor(x)
    shared_layer <- torch_flatten(shared_layer, start_dim = 2)
    shared_layer
  }
)

decoder_gen <- nn_module(
  classname = "decoder_gen",
  initialize = function(dim.input) {
    self$decompressor <- nn_sequential(
      nn_linear(25, 18816),
      view(c(-1, dim.input * 8, 28, 28)),
      nn_conv_transpose2d(dim.input * 8, dim.input * 4, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = FALSE),
      nn_batch_norm2d(dim.input * 4),
      nn_leaky_relu(),
      nn_conv_transpose2d(dim.input * 4, dim.input * 2, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = FALSE),
      nn_batch_norm2d(dim.input * 2),
      nn_leaky_relu(),
      nn_conv_transpose2d(dim.input * 2, 3, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = FALSE),
      nn_sigmoid()
    )
  },
  forward = function(x) {
    x_pred <- self$decompressor(x)
    x_pred
  }
)

VAE_gen <- nn_module(
  classname = "VAE_gen",
  initialize = function(dim.input, dim.latent = 25) {
    self$latent_dim <- dim.latent
    self$encoder <- encoder_gen(dim.input)
    self$mean <- nn_linear(18816, dim.latent)
    self$log_var <- nn_linear(18816, dim.latent)
    self$decoder <- decoder_gen(dim.input)
  },
  forward = function(x) {
    shared_layer <- self$encoder(x)
    self$mu <- self$mean(shared_layer)
    self$log_var_val <- self$log_var(shared_layer)
    self$log_var_val <- torch_clamp(self$log_var_val, min = -10, max = 10)
    z <- self$mu + torch_exp(self$log_var_val$mul(0.5)) * torch_randn(c(dim(x)[1], self$latent_dim))$to(device = device)
    self$z <- z
    self$x_hat <- self$decoder(z)
    self$x_hat
  },
  loss = function(pred, target) {
    recon_loss <- nnf_mse_loss(target, pred, reduction = "mean")
    kl_loss <- -0.5 * torch_sum(1 + self$log_var_val - self$mu$pow(2) - self$log_var_val$exp())
    
    self$.kl_loss <- kl_loss$item()
    self$.recon_loss <- recon_loss$item()
    recon_loss + kl_loss
  }
)

# callback_plot_loss <- luz_callback(
#   name = "plot_loss",
#   initialize = function() {
#     self$losses <- c()
#     plot(1, type = "n", xlim = c(1, 200), 
#          xlab = "Iteration", ylab = "Loss")
#   },
#   on_batch_end = function() {
#     l <- ctx$loss$item()
#     self$losses <- c(self$losses, l)
#     points(length(self$losses), l, col = "blue", pch = 20)
#     Sys.sleep(0)
#   }
# )

metrics_track_callback <- luz_callback(
  name = "Track KL and Reconstruction Loss",
  initialize = function() {
    self$kl_loss <- c()
    self$recon_loss <- c()
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
    cat("KL Loss: ", kl_mean, "\tReconstruction Loss: ", recon_mean, "\n")
    self$recon_loss = c()
    self$kl_loss = c()
  },
  on_valid_end = function() {
    kl_mean = mean(self$kl_loss)
    recon_mean = mean(self$recon_loss)
    ctx$log_metric("kl_loss", as.numeric(kl_mean))
    ctx$log_metric("recon_loss", as.numeric(recon_mean))
    cat("KL Loss: ", kl_mean, "\tReconstruction Loss: ", recon_mean, "\n")
    self$recon_loss = c()
    self$kl_loss = c()
  }
)

num_epochs = 200
input_dim = 3 # No.of channels in Input Images after the last encoder layer images_size -> images_size / 8.
latent_dim = 25 # No.of Dimensions in latent Representation
learning_rate_alpha = 0.001
model <- VAE_gen %>%
  setup(
    optimizer = optim_adam,
    #metrics = luz_callback_metrics()
  ) %>%
  set_opt_hparams(lr = learning_rate_alpha) %>%
  set_hparams(
    dim.input = input_dim, dim.latent = latent_dim
  )

lr_finder_rate <- lr_finder(model, train_dl, steps = num_epochs, start_lr = 1e-07, end_lr = 0.3)
plot(lr_finder_rate)

fitted <- model %>%
  fit(train_dl, epochs = num_epochs, valid_data = valid_dl,
      callbacks = list(
        metrics_track_callback(),
        luz_callback_early_stopping(patience = 15),
        luz_callback_lr_scheduler(
          lr_one_cycle,
          max_lr = (learning_rate_alpha * 1.5),
          epochs = num_epochs,
          steps_per_epoch = length(train_dl),
          call_on = "on_batch_end"
        )
         # callback_plot_loss()
      ),
      verbose = TRUE
  )

luz_save(fitted, "~/Documents/ISM/PhD/3rd Year/Results/CVAE2_mean.pt")

fitted <- luz_load("~/Documents/ISM/PhD/3rd Year/Results/CVAE2.pt")

output <- fitted$model$forward(test_dl)
