x <- torch_randn(20, 15, 64, 64)
m <- nn_conv2d(15, 30, kernel_size = 3, stride = 2, padding = 1)
output <- m(x)
output

t = torch_randn(32, 24, 28, 28)
t
torch_flatten(t, start_dim = 1)
conv <-  nn_conv2d(3, 6, kernel_size = 3, stride = 2, padding = 1)
output <- conv(t)
output
t

lin_layer <- nn_linear(32*24*28*28, 25)
mu <- lin_layer(t)
tconv <- nn_conv_transpose2d(24, 12, kernel_size = 3, stride = 2, padding = 0, bias = FALSE)
out <- tconv(t)
out
