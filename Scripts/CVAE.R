library(torch)
library(torchvision)
library(luz)
library(ggplot2)
library(magick)
library(dplyr)

gc()
rm(list = ls())

device <- torch_device(
  if (torch::backends_mps_is_available())
  {
    "mps"
  } else {
    "cpu"
  }
)
device

image_dataset <- torch::dataset(
  name = "image_dataset",
  initialize = function(img_dir, transform = NULL) {
    self$img_dir = img_dir
    self$files <- list.files(img_dir, full.names = TRUE)
  },
  .getitem = function(index) {
    img_path <- self$files[index]
    img <- torchvision::base_loader(img_path)
    img <- torch_tensor(aperm(img, c(3, 1, 2)), dtype = torch_float(), device = device) / 255
    img <- torchvision::transform_resize(img, c(64, 64))
    img <- torchvision::transform_normalize(
      img,
      mean = c(0.5, 0.5, 0.5),
      std = c(0.5, 0.5, 0.5)
    )
    if (!is.null(self$transform)) {
      img <- self$transform(img)
    }
    img
  },
  .length = function() {
    length(self$files)
  }
)

transform <- function(x) {
  (x - 0.5) / 0.5
}

img_dir_path = "/Users/sarvilmunipally/Documents/ISM/Year/PythonProjects/GENAI_Conda/OnlyImages/images"

dataset <- image_dataset(img_dir_path, transform = transform)
dl <- dataloader(dataset, batch_size = 64, shuffle = TRUE)
batch <- dl$.iter()$.next()
batch
