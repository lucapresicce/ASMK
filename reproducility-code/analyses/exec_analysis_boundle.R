#####################################################################################################################################################
## ASMK DATA ANALYSIS - univariate model #########################################
rm(list = ls())
gc()
setwd("~/MetaApproaches")

# Packages --------------------------------------------------------------------
library(ASMK)
library(Rcpp)
library(RcppArmadillo)
library(mniw)
library(MCMCpack)
library(ggplot2)
library(tictoc)
library(parallel)
library(doParallel)
library(foreach)
library(MBA)
library(classInt)
library(RColorBrewer)
library(sp)
library(fields)
library(mapproj)
library(ggplot2)
library(rworldmap)
library(sf)
library(geoR)
library(spBayes)
library(bayesplot)

# Data loading ----------------------------------------------------------------

# load preprocessed RData
load("AccelerateSMK/SST data/SST_data.RData")

# sinusoidally projected coordinates (scaled to 1000km units) as explanatory variables
knots.sinusoidal <- mapproject(SSTdata$lon, SSTdata$lat, 
                               projection = "sinusoidal")

radius.of.earth = 6.371            ## 6.371 * 1000 kilometers 
knots.sinusoidal = radius.of.earth * (cbind(knots.sinusoidal$x, 
                                            knots.sinusoidal$y))
SSTdata$projX <- knots.sinusoidal[, 1]
SSTdata$projY <- knots.sinusoidal[, 2]

# take a look to data structure and response variable
head(SSTdata)
dim(SSTdata)

# train data
set.seed(1997)
ds_ind <- sample.int(dim(SSTdata)[1], 5e6)
SST_ds <- SSTdata[ds_ind,]

# test data
set.seed(1997)
test_ind <- sample.int(dim(SST_ds)[1], floor(0.025 * length(SST_ds$sst)))
SST_train <- SST_ds[-test_ind, ]
SST_test <- SST_ds[test_ind, ]

# define train dimensions
N <- nrow(SST_train)
crd_S <- SST_train[,c("lon","lat")]
Y <- SST_train$sst
X <- SST_train[,c("projX","projY")]
p <- ncol(X)

# define test dimensions
U <- nrow(SST_test)
crd_U <- SST_test[,c("lon","lat")]
Y_U <- SST_test$sst
X_U <- SST_test[,c("projX","projY")]

# remove full dataset and free memory
rm(list = c("SSTdata", "knots.sinusoidal", "SST_ds"))
gc()

# EDA -------------------------------------------------------------------------

# linear model to collect residual
lm.obj <- lm(Y ~ as.matrix(X))
summary(lm.obj)

# subsample for feasible EDA
set.seed(1997)
subind <- sample.int(N, round(N*0.005))

# computing the maximum distance
d.max <- sqrt((max(SST_train$projX) - min(SST_train$projX))^2 + 
                (max(SST_train$projY) - min(SST_train$projY))^2)
d.max # around 43.350 KM

# check the variogram 
v.resid <- variog(coords = crd_S[subind, ], data = resid(lm.obj)[subind], 
                  uvec = (seq(0, 3*d.max, length = 30))) # 30

par(mfrow=c(1,1))
vario.fit <- variofit(v.resid, cov.model="exponential")
summary(vario.fit)

# free memory
rm("lm.obj")
gc()

# SubSubsample for model testing ----------------------------------------------

set.seed(1997)
n <- 1000000
subsample <- sample(1:N, n)
y <- Y[subsample]
x <- cbind(1, matrix(as.matrix(X[subsample, ]), ncol = 2))
crd_s <- matrix(as.matrix(crd_S[subsample, ]), ncol = 2)

u <- 2500
subsampleu <- sample(1:U, u)
y_u <- Y_U[subsampleu]
x_u <- cbind(1, matrix(as.matrix(X_U[subsampleu, ]), ncol = 2))
crd_u <- matrix(as.matrix(crd_U[subsampleu, ]), ncol = 2)


# Fit Bayesian linear model----------------------------------------------------

# dimension
n <- nrow(x)
p <- ncol(x)

set.seed(1234)
bLM <- bayesLMConjugate(y~x-1, n.samples = 300, 
                        beta.prior.mean = rep(0, times = p),
                        beta.prior.precision = matrix(0, nrow=p, ncol=p),
                        prior.shape = 2, prior.rate = 1)

round(summary(bLM$p.beta.tauSq.samples)$statistics, 2)
round(summary(bLM$p.beta.tauSq.samples)$quantiles, 2)

# posterior predictive
bLM.pred <- spPredict(bLM, pred.covars = x_u, pred.coords = crd_u,
                      start = 1)

y.bLM <- apply(bLM.pred$p.y.predictive.samples, 1, mean)

(RMSPE_bLM <- sqrt(mean((y_u - y.bLM)^2)))
(RMSPE_LM <- sqrt(mean((y_u - x_u %*% lm.fit(x = x,y = y)$coef)^2)))


#####################################################################################################################################################
# Subset posterior models -----------------------------------------------------

# chioce the hyperparameter values by looking at: summary(vario.fit)
delta_seq <- ifelse(vario.fit$nugget/vario.fit$cov.pars[1]==0, 1e-6, vario.fit$nugget/vario.fit$cov.pars[1])
phi_vario <- 1 / vario.fit$cov.pars[2]
phi_step <- (1 / vario.fit$cov.pars[2])/3
phi_seq <- seq(-1*phi_step, 1*phi_step, phi_step) + phi_vario

# function for the fit loop
fit_loop <- function(i) {
  
  Yi <- data_part$Y_list[[i]]; Xi <- data_part$X_list[[i]]; crd_i <- data_part$crd_list[[i]]
  p <- ncol(Xi)
  bps <- ASMK::BPS_weights(data = list(Y = Yi, X = Xi),
                           priors = list(mu_b = matrix(rep(0, p)),
                                         V_b = diag(10, p),
                                         a = 2,
                                         b = 2), coords = crd_i,
                           hyperpar = list(delta = delta_seq, phi = phi_seq), K = 5)
  
  
  w_hat <- bps$W
  epd <- bps$epd
  
  result <- list(epd, w_hat)
  return(result)
  
}

# function for the pred loop
pred_loop <- function(r) {
  
  ind_s <- subset_ind[r]
  Ys <- matrix(data_part$Y_list[[ind_s]]); Xs <- data_part$X_list[[ind_s]]; crds <- data_part$crd_list[[ind_s]]; Ws <- W_list[[ind_s]]
  result <- ASMK::BPS_post(data = list(Y = Ys, X = Xs), coords = crds,
                           X_u = x_u, crd_u = crd_u,
                           priors = list(mu_b = matrix(rep(0, p)),
                                         V_b = diag(10, p),
                                         a = 2,
                                         b = 2),
                           hyperpar = list(delta = delta_seq, phi = phi_seq),
                           W = Ws, R = 1)
  
  return(result)
}

# subsetting data
subset_size <- 500
K <- n/subset_size
data_part <- subset_data(data = list(Y = matrix(y), X = x, crd = crd_s), K = K)

# ASMK parallel fit -------------------------------------------------------

# number of clusters for parallel implementation
n.core <- parallel::detectCores(logical=F)-1

# list of function
funs_fit <- lsf.str()[which(lsf.str() != "fit_loop")]

# list of function
funs_pred <- lsf.str()[which(lsf.str() != "pred_loop")]

# starting cluster
cl <- makeCluster(n.core)  
registerDoParallel(cl)

# timing
tic("total")

# parallelized subset computation of GP in different cores
tic("fit")
obj_fit <- foreach(i = 1:K, .noexport = funs_fit) %dopar% { fit_loop(i) }
fit_time <- toc()

gc()
# Combination using double BPS
tic("comb")
comb_bps <- BPS_PseudoBMA(obj_fit)
comb_time <- toc()
Wbps <- comb_bps$W
W_list <- comb_bps$W_list

gc()
# parallelized subset computation of GP in different cores
R <- 250
subset_ind <- sample(1:K, R, T, Wbps)
tic("prediction")
predictions <- foreach(r = 1:R, .noexport = funs_pred) %dopar% { pred_loop(r) }
prd_time <- toc()

# timing
tot_time <- toc()

# closing cluster
stopCluster(cl)
gc()


# Results collection ----------------------------------------------------------

# statistics computations W
pred_mat_W <- sapply(1:R, function(r){predictions[[r]][[1]]})
post_mean_W <- rowMeans(pred_mat_W)
post_var_W <- apply(pred_mat_W, 1, sd)
post_qnt_W <- apply(pred_mat_W, 1, quantile, c(0.025, 0.975))

# statistics computations Y
pred_mat_Y <- sapply(1:R, function(r){predictions[[r]][[2]]})
post_mean_Y <- rowMeans(pred_mat_Y)
post_var_Y <- apply(pred_mat_Y, 1, sd)
post_qnt_Y <- apply(pred_mat_Y, 1, quantile, c(0.025, 0.975))

# Empirical coverage for Y
coverage_Y <- mean(y_u >= post_qnt_Y[1,] & y_u <= post_qnt_Y[2,])
cat("Empirical coverage for Response:", round(coverage_Y, 3))
(CI_avlen_asmk <- mean(post_qnt_Y[2,]-post_qnt_Y[1,]))

# Root Mean Square Prediction Error
(rmspe_Y <- sqrt( mean( (y_u - post_mean_Y)^2 ) ))

# Naive linear model RMSPE
naive <- x_u %*% lm.fit(y = y, x = x)$coef
(rmspe_naive <- sqrt( mean( (y_u - naive)^2 ) ))

# Posterior inference -----------------------------------------------------

# collecting posterior sample
smp <- sapply(1:R, function(r){c(predictions[[r]][[4]], predictions[[r]][[3]][,1:p])})
post_mean_smp <- rowMeans(smp)
post_var_smp <- apply(smp, 1, sd)
post_qnt_smp <- apply(smp, 1, quantile, c(0.05, 0.95))
post_mean_hyp <- sapply(1:K, function(k)t(expand_grid_cpp(delta_seq, phi_seq)) %*% W_list[[k]]) %*% Wbps
post_var_hyp <- sapply(1:K, function(k)t(expand_grid_cpp(delta_seq, phi_seq)) %*% W_list[[k]])^2 %*% Wbps - (post_mean_hyp^2)  

posterior_asmk <- t(smp)
colnames(posterior_asmk) <- c("sigma^2", "beta[1]", "beta[2]", "beta[3]")

# plotting asmk
bayesplot_theme_set(theme_default(base_size = 14, base_family = "sans"))
color_scheme_set("red")
mcmc_areas(posterior_asmk,
           prob = 0.95,
           point_est = "mean") + 
  scale_y_discrete(labels = c(expression(sigma^2), expression(beta[1]), expression(beta[2]), expression(beta[3])))


# Save timing result ----------------------------------------------------------

elapsed_times <- c("Fitting" = as.numeric(fit_time$toc-fit_time$tic),
                   "Combination" = as.numeric(comb_time$toc-comb_time$tic),
                   "Prediction" = as.numeric(prd_time$toc-prd_time$tic),
                   "Total time" = as.numeric(tot_time$toc-tot_time$tic))

cat("minutes elapsed for fully model-based uncertainty quantification : \n"); round(elapsed_times/60, 2)


# Plotting data --------------------------------------------------

# plot data on worldmap
newmap <- getMap(resolution = "low")
col.pal <- colorRampPalette(brewer.pal(11,'RdBu')[1:11])
colors <- rev(col.pal(101))
# training data points
plot(newmap, xlim = c(-180, 180), ylim = c(-90, 90), asp = 1)
zcolor <- colors[(y - min(y)) /
                   diff(range(y))*100 + 1]
points(crd_s[, 1], crd_s[, 2], col = zcolor, cex = .1)
# test data points
plot(newmap, xlim = c(-180, 180), ylim = c(-90, 90), asp = 1)
zcolor <- colors[(y_u - min(y_u)) /
                   diff(range(y_u))*100 + 1]
points(crd_u[, 1], crd_u[, 2], col = zcolor, cex = .1)


# world surface interpolation
h <- 12
surf.raw.train <- mba.surf(cbind(crd_s, y), no.X = 300, 
                           no.Y = 300, exten = F, sp = TRUE, h = h)$xyz.est
surf.raw.test <- mba.surf(cbind(crd_u, y_u), no.X = 300, 
                          no.Y = 300, exten = F, sp = TRUE, h = h)$xyz.est

surf.brks <- classIntervals(surf.raw.train$z, 50, 'pretty')$brks
col.pal <- colorRampPalette(brewer.pal(11, 'RdBu')[1:11])
xlim <- range(crd_u[, 1])
zlim <- range(surf.raw.train[["z"]][which(!is.na(surf.raw.train[["z"]]))])

width <- 360
height <- 360
pointsize <- 12

# ggplot version plot
surf_df <- as.data.frame(surf.raw.train)
(combined_train <- ggplot() +
  geom_tile(data = surf_df, aes(x = x, y = y, fill = z)) +
  scale_fill_gradientn(colours = rev(col.pal(length(surf.brks)-1)), limits = zlim) +
  geom_sf(data = st_as_sf(newmap), fill = "#009E73") +
  coord_sf(xlim = c(-175, 175), ylim = c(-60, 85), expand = FALSE) +
  # labs(title = "Train data interpolation") +
  theme_minimal())
surf_df <- as.data.frame(surf.raw.test)
(combined_test <- ggplot() +
  geom_tile(data = surf_df, aes(x = x, y = y, fill = z)) +
  scale_fill_gradientn(colours = rev(col.pal(length(surf.brks)-1)), limits = zlim) +
  geom_sf(data = st_as_sf(newmap), fill = "#BDFD9C") +
  coord_sf(xlim = c(-175, 175), ylim = c(-60, 85), expand = FALSE) +
  # labs(title = "Test data interpolation") +
  theme_minimal())


# Plotting results ------------------------------------------------------------

h <- 12
surf.raw.pred <- mba.surf(cbind(crd_u, post_mean_Y), no.X = 300, 
                          no.Y = 300, exten = F, sp = TRUE, h = h)$xyz.est

surf.brks <- classIntervals(surf.raw.pred$z, 50, 'pretty')$brks
col.pal <- colorRampPalette(brewer.pal(11, 'RdBu')[1:11])
xlim <- range(crd_u[, 1])
zlim <- range(surf.raw.pred[["z"]][which(!is.na(surf.raw.pred[["z"]]))])

# ggplot version plot
surf_df <- as.data.frame(surf.raw.pred)
(combined_pred <- ggplot() +
  geom_tile(data = surf_df, aes(x = x, y = y, fill = z)) +
  scale_fill_gradientn(colours = rev(col.pal(length(surf.brks)-1)), limits = zlim) +
  geom_sf(data = st_as_sf(newmap), fill = "#009E73") +
  coord_sf(xlim = c(-175, 175), ylim = c(-60, 85), expand = FALSE) +
  # labs(title = "Predicted data interpolation") +
  theme_minimal())

# graphical UC for Y (ordered)
ord_y <- order(y_u)
set.seed(1995)
plot_ind <- sample(1:u, 250)
df <- data.frame(
  x_ax = (1:u)[plot_ind],
  Yu_ord = y_u[ord_y][plot_ind],
  CI_Y_lower = post_qnt_Y[, ord_y][1,][plot_ind],
  CI_Y_upper = post_qnt_Y[, ord_y][2,][plot_ind],
  Ymap_ord = post_mean_Y[ord_y][plot_ind])
# Create the ggplot
(uc_Y <- ggplot(df, aes(x = x_ax, y = Yu_ord)) +
  geom_point(pch = 18, size = 3.5, col = "#1A85FF") +
  geom_errorbar(aes(ymin = CI_Y_lower, ymax = CI_Y_upper), 
                width = 0.05, 
                linetype = "dashed",
                linewidth = 0.05,
                color = "#D41159") +
  ylim(range(c(df$CI_Y_lower, df$CI_Y_upper))) +
  labs(x = "Ordered locations", y = "Response values") +
  geom_point(aes(y = Ymap_ord), pch = 15, size = 1.5, col = "#D41159") +
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.line = element_line(colour = "black")))


# Save results ----------------------------------------------------------------

gc()
# Save the entire environment
results <- list("time"    = elapsed_times,
                "fit"     = obj_fit,
                "comb"    = comb_bps,
                "pred"    = predictions,
                "post"    = smp,
                "metrics" = c("RMSPE" = rmspe_Y, "naive" = rmspe_naive, "CI_len" = CI_avlen_asmk, "ECoverage" = coverage_Y))

rm(list = ls()[which(!(ls() %in% c("results")))])
save.image(file = "datanalysis_univariate_fast.RData")
# load("datanalysis_univariate_fast.RData")
# results$time; cat("minutes elapsed for fully model-based uncertainty quantification : \n"); round(results$time/60, 2)
# results$metrics
# results$comb$W

#####################################################################################################################################################
## MASMK DATA ANALYSIS - multivariate model #########################################
rm(list = ls())
gc()
setwd("~/MetaApproaches")

# Packages --------------------------------------------------------------------
library(ASMK)
library(Rcpp)
library(RcppArmadillo)
library(mniw)
library(ggplot2)
library(tictoc)
library(parallel)
library(doParallel)
library(foreach)
library(MBA)
library(classInt)
library(RColorBrewer)
library(sp)
library(fields)
library(ggplot2)
library(sf)
library(geoR)
library(spBayes)
library(bayesplot)
library(raster)
library(corrplot)


# Data loading ----------------------------------------------------------------

# load preprocessed RData
load("MultivariateASMK/NDVI data/cleaned_data2_expanded.RData")
full_data <- data_cleaned2
rm(list = c("data_cleaned2"))
names(full_data)

# take a look to data structure and responses variables
head(full_data)
dim(full_data)

# train data and test data
set.seed(1997)
test_ind <- sample.int(nrow(full_data), floor(0.025 * nrow(full_data)) )
train_data <- full_data[-test_ind, ]
test_data<- full_data[test_ind, ]

# select responses and predictors sets
response_set <- c("NDVI", "red reflectance")
q <- length(response_set)
predictor_set <- c("Non_Vegetated_or_Builtup_Lands")
p <- length(predictor_set)+1

# define train dimensions
N <- nrow(train_data)
crd_S <- matrix(as.matrix(train_data[, c("scaled_x","scaled_y")]), ncol = 2)
Y_S   <- matrix(as.matrix(train_data[, response_set]), ncol = q)
X_S   <- cbind(1, matrix(as.matrix(train_data[, predictor_set]), ncol = (p-1)))

# define test dimensions
U <- nrow(test_data)
crd_U <- matrix(as.matrix(test_data[, c("scaled_x","scaled_y")]), ncol = 2)
Y_U   <- matrix(as.matrix(test_data[, response_set]), ncol = q)
X_U   <- cbind(1, matrix(as.matrix(test_data[, predictor_set]), ncol = (p-1)))

# remove full dataset and free memory
rm(list = c("full_data", "train_data", "test_data"))
gc()


# EDA -------------------------------------------------------------------------

# return to the original scale
Y_S[,1] <- (exp(Y_S[,1])-1)

# linear model to collect residual
lin_res <- Y_S - X_S %*% solve(crossprod(X_S))%*%(crossprod(X_S, Y_S))
summary(lin_res); cov(lin_res)

# subsample for feasible EDA
set.seed(1997)
eda_ind <- sample.int(N, round(N*0.001))

# computing the maximum distance
d.max <- sqrt((max(crd_S[,1]) - min(crd_S[,1]))^2 +
                (max(crd_S[,2]) - min(crd_S[,2]))^2)
d.max # around 1.572

# check the variogram for the first response
v.res_1 <- variog(coords = crd_S[eda_ind, ], data = lin_res[eda_ind, 1],
                  uvec = (seq(0, 0.675, length = 30))) # 30

par(mfrow=c(1,1))
vario.fit_1 <- variofit(v.res_1, cov.model="exponential")
summary(vario.fit_1)

variofitphi.resid1 <- 1 / vario.fit_1$cov.pars[2]; variofitphi.resid1
variofitalpha.resid1 <- vario.fit_1$cov.pars[1] / (vario.fit_1$nugget+vario.fit_1$cov.pars[1]); variofitalpha.resid1

# check the variogram for the second response
v.res_2 <- variog(coords = crd_S[eda_ind, ], data = lin_res[eda_ind, 2],
                  uvec = (seq(0, 0.5, length = 30))) # 30

par(mfrow=c(1,1))
vario.fit_2 <- variofit(v.res_2, cov.model="exponential")
summary(vario.fit_2)

variofitphi.resid2 <- 1 / vario.fit_2$cov.pars[2]; variofitphi.resid2
variofitalpha.resid2 <- vario.fit_2$cov.pars[1] / (vario.fit_2$nugget+vario.fit_2$cov.pars[1]); variofitalpha.resid2
# # variofitalpha.resid2 <- vario.fit_2$cov.pars[1] / (v.res_2$v[1]-1e-4+vario.fit_2$cov.pars[1]); variofitalpha.resid2

# check the variogram for the first response
v.res_y1 <- variog(coords = crd_S[eda_ind, ], data = Y_S[eda_ind, 1],
                   # uvec = (seq(0, 3*d.max, length = 50)[-c(7:11)])) # 30
                   uvec = (seq(0, 3*d.max, length = 50)[c(1:6,12:13)])) # 30

par(mfrow=c(1,1))
vario.fit_y1 <- variofit(v.res_y1, cov.model="exponential")
summary(vario.fit_y1)

# check the variogram for the second response
v.res_y2 <- variog(coords = crd_S[eda_ind, ], data = Y_S[eda_ind, 2],
                   # uvec = (seq(0, 3*d.max, length = 50)[-c(6:11)])) # 30
                   uvec = (seq(0, 3*d.max, length = 50)[c(1:5, 12:13)])) # 30

par(mfrow=c(1,1))
vario.fit_y2 <- variofit(v.res_y2, cov.model="exponential")
summary(vario.fit_y2)

# free memory
rm(list = c("lin_res"))
gc()

# SubSubsample for model testing ----------------------------------------------

set.seed(1997)
n <- 1000000
subsample <- sample(1:N, n)
crd_s <- crd_S[subsample, ]
y     <- Y_S[subsample, ]
x     <- X_S[subsample, ]

u <- 2500
subsampleu <- sample(1:U, u)
crd_u <- crd_U[subsampleu, ]
y_u   <- Y_U[subsampleu, ]
x_u   <- X_U[subsampleu, ]


# Fit linear model----------------------------------------------------

# dimension
n <- nrow(x)
p <- ncol(x)

naive <- x_u[,1] %*% solve(crossprod(x[,1]))%*%(crossprod(x[,1], y))

(rmspe_naive <- sqrt(colMeans((y_u - naive)^2))); mean(rmspe_naive)
(mape_naive <- colMeans( abs(y_u - naive) ) ); mean(mape_naive)

#####################################################################################################################################################

# Subset posterior models -----------------------------------------------------

# hyperparameters values by looking at variograms
(alfa_seq <- sort(c(variofitalpha.resid1, variofitalpha.resid2)))
(phi_seq <- sort(2*c(variofitphi.resid1, variofitphi.resid2)))

# function for the fit loop
fit_loop <- function(i) {
  
  Yi <- data_part$Y_list[[i]]; Xi <- data_part$X_list[[i]]; crd_i <- data_part$crd_list[[i]]
  p <- ncol(Xi); q <- ncol(Yi)
  bps <- ASMK::BPS_weights_MvT(data = list(Y = Yi, X = Xi),
                               priors = list(mu_B = matrix(0, nrow = p, ncol = q),
                                             V_r = diag(10, p),
                                             Psi = diag(1, q),
                                             nu = 3), coords = crd_i,
                               hyperpar = list(alpha = alfa_seq, phi = phi_seq), K = 5)
  w_hat <- bps$W
  epd <- bps$epd
  
  result <- list(epd, w_hat)
  return(result)
  
}

# function for the pred loop
pred_loop <- function(r) {
  
  ind_s <- subset_ind[r]
  Ys <- data_part$Y_list[[ind_s]]; Xs <- data_part$X_list[[ind_s]]; crds <- data_part$crd_list[[ind_s]]; Ws <- W_list[[ind_s]]
  result <- ASMK::BPS_post_MvT(data = list(Y = Ys, X = Xs), coords = crds,
                               X_u = x_u, crd_u = crd_u,
                               priors = list(mu_B = matrix(0, nrow = p, ncol = q),
                                             V_r = diag(10, p),
                                             Psi = diag(1, q),
                                             nu = 3),
                               hyperpar = list(alpha = alfa_seq, phi = phi_seq),
                               W = Ws, R = 1)
  
  return(result)
}

# subsetting data
subset_size <- 500
K <- n/subset_size
data_part <- subset_data(data = list(Y = y, X = x, crd = crd_s), K = K)

# ASMK parallel fit -------------------------------------------------------

# number of clusters for parallel implementation
n.core <- parallel::detectCores(logical = F)-1

# list of function
funs_fit <- lsf.str()[which(lsf.str() != "fit_loop")]

# list of function
funs_pred <- lsf.str()[which(lsf.str() != "pred_loop")]

# starting cluster
cl <- makeCluster(n.core)
registerDoParallel(cl)

# timing
tic("total")

# parallelized subset computation of GP in different cores
tic("fit")
obj_fit <- foreach(i = 1:K, .noexport = funs_fit) %dopar% { fit_loop(i) }
fit_time <- toc()

gc()
# Combination using double BPS
tic("comb")
comb_bps <- BPS_PseudoBMA(obj_fit)
comb_time <- toc()
Wbps <- comb_bps$W
W_list <- comb_bps$W_list

gc()
# parallelized subset computation of GP in different cores
R <- 250
subset_ind <- sample(1:K, R, T, Wbps)
tic("prediction")
predictions <- foreach(r = 1:R, .noexport = funs_pred) %dopar% { pred_loop(r) }
prd_time <- toc()

# timing
tot_time <- toc()

# closing cluster
stopCluster(cl)
gc()


# Results collection ----------------------------------------------------------

# statistics computations W
pred_mat_W <- sapply(1:R, function(r){predictions[[r]]$Pred[[1]]$Wu}, simplify = "array")
post_mean_W <- apply(pred_mat_W, c(1,2), mean)
post_var_W <- apply(pred_mat_W, c(1,2), sd)
post_qnt_W <- apply(pred_mat_W, c(1,2), quantile, c(0.025, 0.975))

# statistics computations Y
pred_mat_Y <- sapply(1:R, function(r){predictions[[r]]$Pred[[1]]$Yu}, simplify = "array")
post_mean_Y <- apply(pred_mat_Y, c(1,2), mean)
post_var_Y <- apply(pred_mat_Y, c(1,2), sd)
post_qnt_Y <- apply(pred_mat_Y, c(1,2), quantile, c(0.025, 0.975))

# Empirical coverage for Y
coverage_Y <- c(mean(y_u[,1] >= post_qnt_Y[1,,1] & y_u[,1] <= post_qnt_Y[2,,1]),
                mean(y_u[,2] >= post_qnt_Y[1,,2] & y_u[,2] <= post_qnt_Y[2,,2]))
# mean(Y_u >= post_qnt_Y[1,,] & Y_u <= post_qnt_Y[2,,])
cat("Empirical average coverage for Spatial process:", round(mean(coverage_Y), 3))
(CI_avlen_masmk <- mean(post_qnt_Y[2,,]-post_qnt_Y[1,,]))

# Root Mean Square Prediction Error
(rmspe_Y <- sqrt( colMeans( (y_u - post_mean_Y)^2 ) )); mean(rmspe_Y)
(mape_Y <- colMeans( abs(y_u - post_mean_Y) ) ); mean(mape_Y)


# Posterior inference -----------------------------------------------------

beta_smp <- sapply(1:R, function(r){predictions[[r]]$Post[[1]]$beta[1:p,]}, simplify = "array")
post_mean_beta <-  apply(beta_smp, c(1,2), mean)
post_var_beta <- apply(beta_smp, c(1,2), sd)
post_qnt_beta <- apply(beta_smp, c(1,2), quantile, c(0.05, 0.95))

sigma_smp <- sapply(1:R, function(r){predictions[[r]]$Post[[1]]$sigma}, simplify = "array")
post_mean_sigma <- apply(sigma_smp, c(1,2), mean)
post_var_sigma <- apply(sigma_smp, c(1,2), sd)
post_qnt_sigma <- apply(sigma_smp, c(1,2), quantile, c(0.05, 0.95))

(post_mean_hyp <- sapply(1:K, function(k)t(expand_grid_cpp(alfa_seq, phi_seq)) %*% W_list[[k]]) %*% Wbps)
post_var_hyp <- sapply(1:K, function(k)t(expand_grid_cpp(alfa_seq, phi_seq)) %*% W_list[[k]])^2 %*% Wbps - (post_mean_hyp^2)

# Save timing result ----------------------------------------------------------

elapsed_times <- c("Fitting" = as.numeric(fit_time$toc-fit_time$tic),
                   "Combination" = as.numeric(comb_time$toc-comb_time$tic),
                   "Prediction" = as.numeric(prd_time$toc-prd_time$tic),
                   "Total time" = as.numeric(tot_time$toc-tot_time$tic))

cat("minutes elapsed for fully model-based uncertainty quantification : \n"); round(elapsed_times/60, 2)


# Plotting data --------------------------------------------------

# subsetting for plotting interpolation
set.seed(1997)
plot_ind <- sample(1:N, 5000)

# setting plot dimensions
width <- 360
height <- 360
pointsize <- 12

# interpolation
h <- 12
surf.raw.NDVI <- mba.surf(cbind(crd_S[plot_ind,], Y_S[plot_ind, 1]), no.X = 300,
                          no.Y = 300, exten = F, sp = TRUE, h = h)$xyz.est

# Color palettes
col.pal1 <- colorRampPalette(RColorBrewer::brewer.pal(11, 'RdBu')[1:11])
colors1 <- col.pal1(5)

# plot limits
xlim <- range(crd_S[plot_ind, 1])
zlim <- range(surf.raw.NDVI[["z"]][which(!is.na(surf.raw.NDVI[["z"]]))])

# ggplot version plot
surf_df <- as.data.frame(surf.raw.NDVI)
(ggNDVI <- ggplot() +
  geom_tile(data = surf_df, aes(x = x, y = y, fill = z)) +
  scale_fill_gradientn(colours = colors1, limits = zlim) +
  # labs(title = "NDVI - train data interpolation") +
  theme_minimal())


# interpolation
h <- 12
surf.raw.RR <- mba.surf(cbind(crd_S[plot_ind,], Y_S[plot_ind, 2]), no.X = 300,
                        no.Y = 300, exten = F, sp = TRUE, h = h)$xyz.est
# Color palettes
col.pal2 <- colorRampPalette(RColorBrewer::brewer.pal(9, 'YlGn')[1:9])
colors2 <- rev(col.pal2(5))

# plot limits
xlim <- range(crd_S[plot_ind, 1])
zlim <- range(surf.raw.RR[["z"]][which(!is.na(surf.raw.RR[["z"]]))])

# ggplot version plot
surf_df <- as.data.frame(surf.raw.RR)
(ggRR <- ggplot() +
  geom_tile(data = surf_df, aes(x = x, y = y, fill = z)) +
  scale_fill_gradientn(colours = colors2, limits = zlim) +
  theme_minimal())



# Plotting test data --------------------------------------------------

# setting plot dimensions
width <- 360
height <- 360
pointsize <- 12

# interpolation
h <- 12
surf.raw.NDVIu <- mba.surf(cbind(crd_u, y_u[, 1]), no.X = 300,
                           no.Y = 300, exten = F, sp = TRUE, h = h)$xyz.est

# Color palettes
col.pal1 <- colorRampPalette(RColorBrewer::brewer.pal(11, 'RdBu')[1:11])
colors1 <- col.pal1(5)

# plot limits
xlim <- range(crd_u[, 1])
zlim <- range(surf.raw.NDVIu[["z"]][which(!is.na(surf.raw.NDVIu[["z"]]))])

# ggplot version plot
surf_df <- as.data.frame(surf.raw.NDVIu)
(ggNDVIu <- ggplot() +
  geom_tile(data = surf_df, aes(x = x, y = y, fill = z)) +
  scale_fill_gradientn(colours = colors1, limits = zlim) +
  theme_minimal())

# interpolation
h <- 12
surf.raw.RRu <- mba.surf(cbind(crd_u, y_u[, 2]), no.X = 300,
                         no.Y = 300, exten = F, sp = TRUE, h = h)$xyz.est
# Color palettes
col.pal2 <- colorRampPalette(RColorBrewer::brewer.pal(9, 'YlGn')[1:9])
colors2 <- rev(col.pal2(5))

# plot limits
xlim <- range(crd_u[, 1])
zlim <- range(surf.raw.RRu[["z"]][which(!is.na(surf.raw.RRu[["z"]]))])

# ggplot version plot
surf_df <- as.data.frame(surf.raw.RRu)
(ggRRu <- ggplot() +
  geom_tile(data = surf_df, aes(x = x, y = y, fill = z)) +
  scale_fill_gradientn(colours = colors2, limits = zlim) +
  theme_minimal())



# Plotting results ------------------------------------------------------------

# setting plot dimensions
width <- 360
height <- 360
pointsize <- 12

# interpolation
h <- 12
surf.raw.NDVIhat <- mba.surf(cbind(crd_u, post_mean_Y[, 1]), no.X = 300,
                             no.Y = 300, exten = F, sp = TRUE, h = h)$xyz.est
# surf.raw.NDVIhat <- mba.surf(cbind(crd_u, naive[, 1]), no.X = 300,
#                             no.Y = 300, exten = F, sp = TRUE, h = h)$xyz.est

# Color palettes
col.pal1 <- colorRampPalette(RColorBrewer::brewer.pal(11, 'RdBu')[1:11])
colors1 <- col.pal1(5)

# plot limits
xlim <- range(crd_u[, 1])
zlim <- range(surf.raw.NDVIhat[["z"]][which(!is.na(surf.raw.NDVIhat[["z"]]))])

# ggplot version plot
surf_df <- as.data.frame(surf.raw.NDVIhat)
(ggNDVIhat <- ggplot() +
  geom_tile(data = surf_df, aes(x = x, y = y, fill = z)) +
  scale_fill_gradientn(colours = colors1, limits = zlim) +
  theme_minimal())


# interpolation
h <- 12
surf.raw.RRhat <- mba.surf(cbind(crd_u, post_mean_Y[, 2]), no.X = 300,
                           no.Y = 300, exten = F, sp = TRUE, h = h)$xyz.est

# Color palettes
col.pal2 <- colorRampPalette(RColorBrewer::brewer.pal(9, 'YlGn')[1:9])
colors2 <- rev(col.pal2(5))

# plot limits
xlim <- range(crd_u[, 1])
zlim <- range(surf.raw.RRhat[["z"]][which(!is.na(surf.raw.RRhat[["z"]]))])

# ggplot version plot
surf_df <- as.data.frame(surf.raw.RRhat)
(ggRRhat <- ggplot() +
  geom_tile(data = surf_df, aes(x = x, y = y, fill = z)) +
  scale_fill_gradientn(colours = colors2, limits = zlim) +
  theme_minimal())


# graphical UC for Y1 (ordered)
ord_y <- order(y_u[,1])
plot_ind <- sample(1:u, 250)
df <- data.frame(
  x_ax = (1:u)[plot_ind],
  Yu_ord = y_u[ord_y, 1][plot_ind],
  CI_Y_lower = post_qnt_Y[1, ord_y, 1][plot_ind],
  CI_Y_upper = post_qnt_Y[2, ord_y, 1][plot_ind],
  Ymap_ord = post_mean_Y[ord_y, 1][plot_ind])

# Create the ggplot
(uc_Y1 <- ggplot(df, aes(x = x_ax, y = Yu_ord)) +
  geom_point(pch = 18, size = 3.5, col = "#1A85FF") +
  geom_errorbar(aes(ymin = CI_Y_lower, ymax = CI_Y_upper),
                width = 0.05,
                linetype = "dashed",
                linewidth = 0.05,
                color = "#D41159") +
  ylim(range(c(df$CI_Y_lower, df$CI_Y_upper))) +
  labs(x = "Ordered locations", y = "Response values") +
  geom_point(aes(y = Ymap_ord), pch = 15, size = 1.5, col = "#D41159") +
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.line = element_line(colour = "black")))


# graphical UC for Y2 (ordered)
ord_y <- order(y_u[,2])
plot_ind <- sample(1:u, 250)
df <- data.frame(
  x_ax = (1:u)[plot_ind],
  Yu_ord = y_u[ord_y, 2][plot_ind],
  CI_Y_lower = post_qnt_Y[1, ord_y, 2][plot_ind],
  CI_Y_upper = post_qnt_Y[2, ord_y, 2][plot_ind],
  Ymap_ord = post_mean_Y[ord_y, 2][plot_ind])

# Create the ggplot
(uc_Y2 <- ggplot(df, aes(x = x_ax, y = Yu_ord)) +
  geom_point(pch = 18, size = 3.5, col = "#1A85FF") +
  geom_errorbar(aes(ymin = CI_Y_lower, ymax = CI_Y_upper),
                width = 0.05,
                linetype = "dashed",
                linewidth = 0.05,
                color = "#D41159") +
  ylim(range(c(df$CI_Y_lower, df$CI_Y_upper))) +
  labs(x = "Ordered locations", y = "Response values") +
  geom_point(aes(y = Ymap_ord), pch = 15, size = 1.5, col = "#D41159") +
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.line = element_line(colour = "black")))



# Save results ----------------------------------------------------------------

gc()
# Save the entire environment
results <- list("time"    = elapsed_times,
                "fit"     = obj_fit,
                "comb"    = comb_bps,
                "pred"    = predictions,
                "post"    = list(beta_smp, sigma_smp),
                "metrics" = c("RMSPE" = rmspe_Y, "naiveR" = rmspe_naive, "CI_len" = CI_avlen_masmk, "ECoverage" = coverage_Y, "MAPE" = mape_Y, "naiveM" = mape_naive))

rm(list = ls()[which(!(ls() %in% c("results")))])
save.image(file = "datanalysis_multivariate_fast.RData")
# load("datanalysis_multivariate_fast.RData")
results$time; cat("minutes elapsed for fully model-based uncertainty quantification : \n"); round(results$time/60, 2)
results$metrics
results$comb

