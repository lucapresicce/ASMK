n <- 500
u <- 10
p <- 2

# parameters
B <- c(-0.75, 1.85)
tau2 <- 0.25
sigma2 <- 1
delta <- tau2/sigma2
phi <- 4

set.seed(20081997)
# generate sintethic data
crd <- matrix(runif((n+u) * 2), ncol = 2)
X_or <- cbind(rep(1, n+u), matrix(runif((p-1)*(n+u)), ncol = (p-1)))
D <- as.matrix(dist(crd))
gc()
Rphi <- exp(-phi * D)
rm("D"); gc()
W_or <- matrix(0, n+u) + mniw::rmNorm(1, rep(0, n+u), sigma2*Rphi)
rm("Rphi"); gc()
Y_or <- X_or %*% B + W_or + mniw::rmNorm(1, rep(0, n+u), diag(delta*sigma2, n+u))
gc()

# sample data
crd_s <- crd[1:n, ]
X <- X_or[1:n, ]
W <- W_or[1:n, ]
Y <- matrix(Y_or[1:n, ])

# prediction data
crd_u <- crd[-(1:n), ]
X_u <- X_or[-(1:n), ]
W_u <- W_or[-(1:n), ]
Y_u <- matrix(Y_or[-(1:n), ])


fit1 <- fit_cpp(data   = list(Y = Y, X = X),
                   priors = list(mu_b = rep(0, p),
                                 V_b  = diag(10, p),
                                 a    = 2,
                                 b    = 2),
                   coords   = crd_s,
                   hyperpar = list(delta = 0.25,
                                   phi   = 4))

crd_us <- rbind(crd_u, crd_s)
Rphi_us <- exp(-phi * as.matrix(dist(crd_us))[1:u, (u+1):(u+n)])

# t univariate functions --------------------------------------------------

prova4$W_u <- r_pred_T(data     = list(Y = Y, X = X),
                   poster   = fit1,
                   hyperpar = list(delta = 0.25,
                                   phi   = 4),
                   X_u      = X_u,
                   d_u      = as.matrix(dist(crd_u)),
                   d_us     = as.matrix(dist(rbind(crd_u, crd_s))),
                   R = 1)

(prova4$W_u |> colMeans()-W_u)^2 |> mean() |> sqrt()
(prova4$Y_u |> colMeans()-Y_u)^2 |> mean() |> sqrt()

prova4 <- r_pred_MC(data     = list(Y = Y, X = X),
                     hyperpar = list(delta = 0.25,
                                     phi   = 4),
                     X_u      = X_u,
                     d_u      = as.matrix(dist(crd_u)),
                     d_us     = as.matrix(dist(rbind(crd_u, crd_s))),
                     post = post_draws(fit1, R = 250, par = F, p = 1),
                     iRphi_s = fit1$iRphi_s)

(prova4$Z_u |> rowMeans()-W_u)^2 |> mean() |> sqrt()
(prova4$Y_u |> rowMeans()-Y_u)^2 |> mean() |> sqrt()


u_set <- 1#:nrow(X_u)
d_pred_MC(data     = list(Y = Y, X = X),
          poster   = post_draws(fit1, R = 10, par = F, p = p),
          hyperpar = list(delta = 0.25,
                         phi   = 4),
          Y_u      = matrix(Y_u[u_set,]),
          X_u      = matrix(X_u[u_set,], ncol = p),
          d_u      = as.matrix(as.matrix(dist(crd_u[u_set,]))[u_set, u_set]),
          d_us     = as.matrix(dist(rbind(crd_u[u_set,], crd_s))),
          iRphi_s = fit1$iRphi_s) |> mean()

d_pred_T(data     = list(Y = Y, X = X),
         poster   = fit1,
         hyperpar = list(delta = 0.25,
                         phi   = 4),
         Y_u      = matrix(Y_u[u_set,]),
         X_u      = matrix(X_u[u_set,], ncol = p),
         d_u      = as.matrix(as.matrix(dist(crd_u[u_set,]))[u_set, u_set]),
         d_us     = as.matrix(dist(rbind(crd_u[u_set,], crd_s))))


tictoc::tic()
P_u2 <- dens_loocvT(data   = list(Y = Y, X = X),
                           priors = list(mu_b = rep(0, p),
                                         V_b  = diag(10, p),
                                         a    = 2,
                                         b    = 2),
                           coords   = crd_s,
                           hyperpar = list(delta = 0.25,
                                           phi   = 4))
tictoc::toc()

hist(P_u)
hist(P_u2)
mean( (P_u - P_u2)^2 )
cbind(P_u2, P_u)

plot(P_u[1:100], type = "l")
lines(P_u2[1:100], col = 2)

dens_loocv(data   = list(Y = Y, X = X),
           priors = list(mu_b = rep(0, p),
                         V_b  = diag(10, p),
                         a    = 2,
                         b    = 2),
           coords   = crd_s,
           hyperpar = list(delta = 0.25,
                           phi   = 4),
           g = 5)

dens_loocvT(data   = list(Y = Y, X = X),
                   priors = list(mu_b = rep(0, p),
                                 V_b  = diag(10, p),
                                 a    = 2,
                                 b    = 2),
                   coords   = crd_s,
                   hyperpar = list(delta = 0.25,
                                   phi   = 4))

tictoc::tic()
dens_kcv(data   = list(Y = Y, X = X),
         priors = list(mu_b = rep(0, p),
                       V_b  = diag(10, p),
                       a    = 2,
                       b    = 2),
         coords   = crd_s,
         hyperpar = list(delta = 0.25,
                         phi   = 4),
         K = 5,
         g = 5)
tictoc::toc()

tictoc::tic()
dens_kcvT(data   = list(Y = Y, X = X),
                 priors = list(mu_b = rep(0, p),
                               V_b  = diag(10, p),
                               a    = 2,
                               b    = 2),
                 coords   = crd_s,
                 hyperpar = list(delta = 0.25,
                                 phi   = 4),
                 K = 5)
tictoc::toc()

tictoc::tic()
scores1 <- models_dens(data   = list(Y = Y, X = X),
                       priors = list(mu_b = rep(0, p),
                                     V_b  = diag(10, p),
                                     a    = 2,
                                     b    = 2),
                       coords   = crd_s,
                       hyperpar = list(delta = c(0.15, 0.25, 0.35),
                                       phi   = c(3, 4)),
                       useKCV = T,
                       K = 5,
                       g = 5)
tictoc::toc()

tictoc::tic()
scores2 <- models_densT(data   = list(Y = Y, X = X),
                               priors = list(mu_b = rep(0, p),
                                             V_b  = diag(10, p),
                                             a    = 2,
                                             b    = 2),
                               coords   = crd_s,
                               hyperpar = list(delta = c(0.15, 0.25, 0.35),
                                               phi   = c(3, 4)),
                               useKCV = T,
                               K = 5)
tictoc::toc()

conv_opt(scores = scores1) |> round(2)
conv_opt(scores = scores2) |> round(2)

###################################################################################################

subdata <- subset_data(data = list(Y = Y, X = X, crd = crd_s), K = 5)

tictoc::tic()
BPS_weightsT(data   = list(Y = Y, X = X),
                    priors = list(mu_b = rep(0, p),
                                  V_b  = diag(10, p),
                                  a    = 2,
                                  b    = 2),
                    coords   = crd_s,
                    hyperpar = list(delta = c(0.15, 0.25, 0.35),
                                    phi   = c(3, 4, 5)),
                    K = 5)
tictoc::toc()

fit_list <- vector(mode = "list", length = K)
for (k in 1:K) {

  data_k <- list(Y = subdata$Y_list[[k]], X = subdata$X_list[[k]])
  crd_k <- subdata$crd_list[[k]]

  fit_list[[k]] <- BPS_weightsT(data   = data_k,
                                       priors = list(mu_b = rep(0, p),
                                                     V_b  = diag(10, p),
                                                     a    = 2,
                                                     b    = 2),
                                       coords   = crd_k,
                                       hyperpar = list(delta = c(0.25, 0.35),
                                                       phi   = c(4, 5)),
                                       K = 5)

}


wbma <- BPS_combine(fit_list = fit_list, K = 5, rp = 1)

# wbma <- BPS_PseudoBMA(fit_list = fit_list)

BPS_predT(data     = list(Y = Y, X = X),
                 X_u      = X_u,
                 crd_u    = crd_u,
                 coords   = crd_s,
                 hyperpar = list(delta = c(0.25, 0.35),
                                 phi   = c(4, 5)),
                 priors   = list(mu_b = rep(0, p),
                                 V_b  = diag(10, p),
                                 a    = 2,
                                 b    = 2),
                 W        = wbma$W_list[[1]],
                 R        = 5)

r <- 20
predic1 <- vector(mode = "list", length = r)
j <- 1
while (j <= r) {

  k <- sample(1:K, 1, F, wbma$W)
  data_k <- list(Y = subdata$Y_list[[k]], X = subdata$X_list[[k]])
  crd_k  <- subdata$crd_list[[k]]
  W_k    <- wbma$W_list[[k]]

  predic1[[j]] <- BPS_predT(data     = data_k,
                                   X_u      = X_u,
                                   crd_u    = crd_u,
                                   coords   = crd_k,
                                   hyperpar = list(delta = c(0.25, 0.35),
                                                   phi   = c(4, 5)),
                                   priors   = list(mu_b = rep(0, p),
                                                   V_b  = diag(10, p),
                                                   a    = 2,
                                                   b    = 2),
                                   W        = W_k,
                                   R        = 5)

  j <- j+1

}

Wmap <- sapply(1:r, function(j)predic1[[j]]$W_hat) |> rowMeans()
# Wmap <- sapply(1:r, function(j)predic1[[j]]$Z_hat) |> rowMeans()
Ymap <- sapply(1:r, function(j)predic1[[j]]$Y_hat) |> rowMeans()
(Wmap - W_u)^2 |> mean() |> sqrt()
(Ymap - Y_u)^2 |> mean() |> sqrt()


spPredict_ASMK(data     = list(Y = Y, X = X),
                       X_u      = X_u,
                       crd_u    = crd_u,
                       coords   = crd_s,
                       hyperpar = list(delta = c(0.25, 0.35),
                                       phi   = c(4, 5)),
                       priors   = list(mu_b = rep(0, p),
                                       V_b  = diag(10, p),
                                       a    = 2,
                                       b    = 2),
                       W        = wbma$W_list[[1]],
                       R        = 5,
                       J = 5)

test1 <- spASMKTversion(data     = list(Y = Y, X = X, crd = crd_s),
                        priors   = list(mu_b = rep(0, p),
                                        V_b  = diag(10, p),
                                        a    = 2,
                                        b    = 2),
                        hyperpar = list(delta = c(0.25, 0.35),
                                        phi   = c(4)),
                        newdata  = list(X = X_u, crd = crd_u),
                        K        = 5,
                        R        = 250)

test1 <- spASMK(data     = list(Y = Y, X = X, crd = crd_s),
                priors   = list(mu_b = rep(0, p),
                                V_b  = diag(10, p),
                                a    = 2,
                                b    = 2),
                hyperpar = list(delta = c(0.25, 0.35),
                                phi   = c(4)),
                newdata  = list(X = X_u, crd = crd_u),
                K        = 5,
                R        = 250)

Wmap <- test1$Predictions$W_hat |> rowMeans()
Ymap <- test1$Predictions$Y_hat |> rowMeans()
(Wmap - W_u)^2 |> mean() |> sqrt()
(Ymap - Y_u)^2 |> mean() |> sqrt()

test1$Comb_weights

BPS_postdraws(data     = list(Y = Y, X = X, crd = crd_s),
              priors   = list(mu_b = rep(0, p),
                              V_b  = diag(10, p),
                              a    = 2,
                              b    = 2),
              coords = crd_s, hyperpar = list(delta = c(0.25, 0.35),
                                              phi   = c(4, 5)),
              W = wbma$W_list[[1]], R = 10)


