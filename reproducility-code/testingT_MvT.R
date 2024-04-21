# testing closed form predictive MULTIVARIATE

# generate data
n <- 500
u <- 1000
p <- 2
q <- 2

# parameters
B     <- matrix(c(-0.75, 1.85, -1.1, 1.9), p, q)
Sigma <- matrix(c(1, 0.3, 0.4, 1), q, q)
alfa <- 0.8
phi   <- 4

set.seed(1997)
# generate sintethic data
crd  <- matrix(runif((n+u) * 2), ncol = 2)
X_or <- cbind(rep(1, n+u), matrix(runif((p-1)*(n+u)), ncol = (p-1)))
D    <- as.matrix(dist(crd))
Rphi <- exp(-phi * D)
W_or <- matrix(0, n+u, q) + mniw::rMNorm(1, Lambda = matrix(0, n+u, q), SigmaR = Rphi, SigmaC = Sigma)
Y_or <- X_or %*% B + W_or + mniw::rMNorm(1, Lambda = matrix(0, n+u, q), SigmaR = diag((1/alfa)-1, n+u), SigmaC = Sigma)

# sample data
crd_s <- crd[1:n, ]
X     <- X_or[1:n, ]
W     <- W_or[1:n, ]
Y     <- Y_or[1:n, ]

# prediction data
crd_u <- crd[-(1:n), ]
X_u   <- X_or[-(1:n), ]
W_u   <- W_or[-(1:n), ]
Y_u   <- Y_or[-(1:n), ]


# fit, post and pred ------------------------------------------------------

tictoc::tic()
fit1 <- fit_cpp_MvT(data   = list(Y = Y, X = X),
                    priors = list(mu_B = matrix(0, nrow = p, ncol = q),
                                  V_r = diag(10, p),
                                  Psi = diag(1, q),
                                  nu = 3),
                    coords   = crd_s,
                    hyperpar = list(alpha = c(0.8),
                                    phi   = c(4)))
tictoc::toc()

crd_us <- rbind(crd_u, crd_s)
Rphi_us <- exp(-phi * as.matrix(dist(crd_us))[1:u, (u+1):(u+n)])

# fit, post and pred ------------------------------------------------------

fit1 <- fit_latent_cpp(data   = list(Y = Y, X = X),
                       priors = list(mu_B = matrix(0, nrow = p, ncol = q),
                                     V_r = diag(10, p),
                                     Psi = diag(1, q),
                                     nu = 3),
                       coords   = crd_s,
                       hyperpar = list(alpha = c(0.8),
                                       phi   = c(4)))

tictoc::tic()
post1 <- post_draws_MvT(poster = fit1,
                        R      = 1,
                        par    = F,
                        p      = p)
tictoc::toc()

post1$Sigmas |> mean()
post1$Betas |> colMeans()


tictoc::tic()
pred1 <- r_pred_cpp_MvT1(data     = list(Y = Y, X = X),
                         poster     = fit1,
                         hyperpar = list(alpha = 0.8,
                                         phi   = 4),
                         X_u      = X_u,
                         d_u      = as.matrix(dist(crd_u)),
                         d_us     = as.matrix(dist(rbind(crd_u, crd_s))),
                         R        = 1)
tictoc::toc()


tictoc::tic()
pred4 <- r_pred_latent_MC2(data     = list(Y = Y, X = X),
                           poster     = fit1,
                           hyperpar = list(alpha = 0.8,
                                           phi   = 4),
                           X_u      = X_u,
                           d_u      = as.matrix(dist(crd_u)),
                           d_us     = as.matrix(dist(rbind(crd_u, crd_s))),
                           post = post1)
tictoc::toc()


tictoc::tic()
pred2 <- r_pred_latent_MC(data     = list(Y = Y, X = X),
                          poster     = fit1,
                          hyperpar = list(alpha = 0.8,
                                          phi   = 4),
                          X_u      = X_u,
                          d_u      = as.matrix(dist(crd_u)),
                          d_us     = as.matrix(dist(rbind(crd_u, crd_s))),
                          beta = post1[[1]]$beta,
                          sigma = post1[[1]]$sigma)
tictoc::toc()

tictoc::tic()
pred3 <- r_pred_cpp_MvTHALF(data     = list(Y = Y, X = X),
                            poster     = fit1,
                            hyperpar = list(alpha = 0.8,
                                            phi   = 4),
                            X_u      = X_u,
                            d_u      = as.matrix(dist(crd_u)),
                            d_us     = as.matrix(dist(rbind(crd_u, crd_s))),
                            R        = 1)
tictoc::toc()

((pred1$Wu |> apply(c(1,2), mean)) - W_u)^2 |> mean() |> sqrt()
((pred1$Yu |> apply(c(1,2), mean)) - Y_u)^2 |> mean() |> sqrt()
((pred2$Wu |> apply(c(1,2), mean)) - W_u)^2 |> mean() |> sqrt()
((pred2$Yu |> apply(c(1,2), mean)) - Y_u)^2 |> mean() |> sqrt()
((pred3$Wu |> apply(c(1,2), mean)) - W_u)^2 |> mean() |> sqrt()
((pred3$Yu |> apply(c(1,2), mean)) - Y_u)^2 |> mean() |> sqrt()
((pred4$Wu |> apply(c(1,2), mean)) - W_u)^2 |> mean() |> sqrt()
((pred4$Yu |> apply(c(1,2), mean)) - Y_u)^2 |> mean() |> sqrt()


u_set <- 1:nrow(X_u)
d_pred_mvt(data     = list(Y = Y, X = X),
           post     = post1,
           hyperpar = list(alpha = 0.8,
                           phi   = 4),
           Y_u      = Y_u[u_set,],
           X_u      = X_u[u_set,],
           d_u      = as.matrix(dist(crd_u[u_set,])),
           d_us     = as.matrix(dist(rbind(crd_u[u_set,], crd_s))),
           iRphi_s  = fit1$iRphi_s) |> mean()

u_set <- 10#:nrow(X_u)
d_pred_latent_cppT(data     = list(Y = Y, X = X),
                   poster   = fit1,
                   hyperpar = list(alpha = 0.8,
                                   phi   = 4),
                   Y_u      = matrix(Y_u[u_set,], ncol = q),
                   X_u      = matrix(X_u[u_set,], ncol = p),
                   d_u      = as.matrix(as.matrix(dist(crd_u[u_set,]))[1:length(u_set), 1:length(u_set)]),
                   d_us     = as.matrix(dist(rbind(crd_u[u_set,], crd_s))))


# dens loocv, kcv, models, conv_opt ---------------------------------------

dens_loocv_latentT(data   = list(Y = Y, X = X),
                   priors = list(mu_B = matrix(0, nrow = p, ncol = q),
                                 V_r = diag(10, p),
                                 Psi = diag(1, q),
                                 nu = 3),
                   coords   = crd_s,
                   hyperpar = list(alpha = c(0.8),
                                   phi   = c(4)))

dens_kcv_latentT(data   = list(Y = Y, X = X),
                 priors = list(mu_B = matrix(0, nrow = p, ncol = q),
                               V_r = diag(10, p),
                               Psi = diag(1, q),
                               nu = 3),
                 coords   = crd_s,
                 hyperpar = list(alpha = c(0.8),
                                 phi   = c(4)),
                 K = 5)

scores1 <- models_dens_latentT(data   = list(Y = Y, X = X),
                               priors = list(mu_B = matrix(0, nrow = p, ncol = q),
                                             V_r = diag(10, p),
                                             Psi = diag(1, q),
                                             nu = 3),
                               coords   = crd_s,
                               hyperpar = list(alpha = c(0.8, 0.9),
                                               phi   = c(3, 4, 5)),
                               useKCV = T,
                               K = 5)

conv_opt(scores = scores1) |> round(2)
expand_grid_cpp(c(0.8, 0.9), c(3, 4, 5))


# subsetdata, BPS weights, pred, postdraws --------------------------------

subdata <- subset_data(data = list(Y = Y, X = X, crd = crd_s), K = 5)


BPS_weights_MvT(data   = list(Y = Y, X = X),
                list(mu_B = matrix(0, nrow = p, ncol = q),
                     V_r = diag(10, p),
                     Psi = diag(1, q),
                     nu = 3),
                coords   = crd_s,
                hyperpar = list(alpha = c(0.75, 0.8, 0.85),
                                phi   = c(3, 4, 5)),
                K = 5)

fit_list <- vector(mode = "list", length = K)
for (k in 1:K) {

  data_k <- list(Y = subdata$Y_list[[k]], X = subdata$X_list[[k]])
  crd_k <- subdata$crd_list[[k]]

  fit_list[[k]] <- BPS_weights_MvT(data   = data_k,
                                   list(mu_B = matrix(0, nrow = p, ncol = q),
                                        V_r = diag(10, p),
                                        Psi = diag(1, q),
                                        nu = 3),
                                   coords   = crd_k,
                                   hyperpar = list(alpha = c(0.75, 0.85),
                                                   phi   = c(4)),
                                   K = 5)


}


BPS_combine(fit_list = fit_list, K = 5, 1)

wbma <- BPS_PseudoBMA(fit_list = fit_list)

BPS_pred_MvTT(data     = list(Y = Y, X = X),
              X_u      = X_u,
              crd_u    = crd_u,
              coords   = crd_s,
              hyperpar = list(alpha = c(0.75, 0.85),
                              phi   = c(4)),
              priors   = list(mu_B = matrix(0, nrow = p, ncol = q),
                              V_r = diag(10, p),
                              Psi = diag(1, q),
                              nu = 3),
              W        = wbma$W_list[[1]],
              R        = 1)

r <- 20
predic1 <- vector(mode = "list", length = r)
j <- 1
while (j <= r) {

  k <- sample(1:K, 1, F, wbma$W)
  data_k <- list(Y = subdata$Y_list[[k]], X = subdata$X_list[[k]])
  crd_k  <- subdata$crd_list[[k]]
  W_k    <- wbma$W_list[[k]]

  predic1[[j]] <- BPS_pred_MvTT(data     = data_k,
                                X_u      = X_u,
                                crd_u    = crd_u,
                                coords   = crd_k,
                                hyperpar = list(alpha = c(0.75, 0.85),
                                                phi   = c(4)),
                                priors   = list(mu_B = matrix(0, nrow = p, ncol = q),
                                                V_r = diag(10, p),
                                                Psi = diag(1, q),
                                                nu = 3),
                                W        = W_k,
                                R        = 1)

  j <- j+1

}


Wmap <- sapply(1:r, function(j)predic1[[j]][[1]]$Wu, simplify = "array") |> apply(c(1, 2), mean)
Ymap <- sapply(1:r, function(j)predic1[[j]][[1]]$Yu, simplify = "array") |> apply(c(1, 2), mean)
(Wmap - W_u)^2 |> mean() |> sqrt()
(Ymap - Y_u)^2 |> mean() |> sqrt()


postsp1 <- BPS_post_mvtTversion(data     = list(Y = Y, X = X),
                                X_u      = X_u,
                                crd_u    = crd_u,
                                coords   = crd_s,
                                hyperpar = list(alpha = c(0.75, 0.85),
                                                phi   = c(4)),
                                priors   = list(mu_B = matrix(0, nrow = p, ncol = q),
                                                V_r = diag(10, p),
                                                Psi = diag(1, q),
                                                nu = 3),
                                W        = wbma$W_list[[1]],
                                R        = r)

sapply(1:r, function(j)postsp1$Post[[j]]$beta[1:p,,], simplify = "array") |> apply(c(1, 2), mean)
sapply(1:r, function(j)postsp1$Post[[j]]$sigma, simplify = "array") |> apply(c(1, 2), mean)

Wmap <- sapply(1:r, function(j)postsp1$Pred[[j]]$Wu, simplify = "array") |> apply(c(1, 2), mean)
Ymap <- sapply(1:r, function(j)postsp1$Pred[[j]]$Yu, simplify = "array") |> apply(c(1, 2), mean)
(Wmap - W_u)^2 |> mean() |> sqrt()
(Ymap - Y_u)^2 |> mean() |> sqrt()

