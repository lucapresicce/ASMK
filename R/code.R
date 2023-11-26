#' Compute the BPS stacking weights for univariate latent spatial regression model
#'
#' @param Y a "N \times 1" matrix of sample response variable
#' @param X a "N \times P" matrix of sample covarites
#' @param crd_s a "N \times 2" matrix of sample coordinates
#' @param Delta (univariate models) a vector of candidate values for hyperparameter \delta
#' @param Alfa (mulrivariate models) a vector of candidate values for hyperparameter \alpha
#' @param Fi a vector of candidate values for hyperparameter \phi
#' @param KCV a boolean flag to use K-fold cross validation instead of LOOCV (default FALSE)
#' @param K if KCV = TRUE, represent the number of desired K-fold
#'
#' @return a matrix with models configuration and the associated weights, and a matrix with the stacking weights
#'
#' @importFrom CVXR Variable Maximize Problem solve
#'
#' @export
PredictiveStackingWeights_cpp <- function(Y, X, crd_s, Delta = NULL, Alfa = NULL, Fi, KCV = F, K = 10) {

  ## mandatory packages --------------------------------------------------------
  # library(CVXR, quietly = T)

  ## evaluate the loocv predictive density -------------------------------------
  q <- ncol(Y)
  if (q > 1) {

    ## control on Alfa
    if (is.null(Alfa)) stop("Set of values for alpha (Alfa) is missing")

    ## dimensions for priors
    q <- ncol(Y)
    p <- ncol(X)

    ## evaluate the loocv/kcv predictive density
    out <- models_dens_latent(data = list(Y = Y, X = X), coords = crd_s,
                              priors = list(mu_B = matrix(0, nrow = p, ncol = q),
                                            V_r = diag(10, p),
                                            Psi = diag(1, q),
                                            nu = 3),
                              hyperpar = list(alpha = Alfa,
                                              phi = Fi),
                              useKCV = KCV, K = K)
  } else {

    ## control on Delta
    if (is.null(Delta)) stop("Set of values for delta (Delta) is missing")

    ## dimensions for priors
    q <- ncol(Y)
    p <- ncol(X)

    ## evaluate the loocv/kcv predictive density
    out <- models_dens(data = list(Y = Y, X = X), coords = crd_s,
                       priors = list(mu_b = matrix(rep(0, p)),
                                     V_b = diag(10, p),
                                     a = 2,
                                     b = 2),
                       hyperpar = list(delta = Delta,
                                       phi = Fi),
                       useKCV = KCV, K = K)
  }

  ## solve the convex optimization problem -------------------------------------

  # cat("\n Recovering the Predictive Stacking weights : \n")
  # tic()
  # declare variable
  scores <- out
  weights <- Variable( ncol(scores) )

  # set up minimization problem and solve it
  constraints <- list(weights >= 0, sum(weights) == 1)
  # the constraint for sum up to 1 with positive weights
  f <- Maximize( mean( log( scores %*% weights ) ) )
  problem <- Problem(f, constraints)
  result <- solve(problem, solver = "ECOS_BB") # ECOS, SCS, OSQP

  # define the stacking predictive distribution obtained
  w_hat <- result$getValue(weights)

  # tcop <- toc(quiet = T)
  # message(tcop$callback_msg)

  ## function return
  if (q > 1) {
    grid <- expand_grid_cpp(Alfa, Fi)
    dfw <- as.matrix(data.frame(
      "W" = round(w_hat, 3), "alfa" = grid[, 1], "phi" = grid[, 2]
    ))
  } else {
    grid <- expand_grid_cpp(Delta, Fi)
    dfw <- as.matrix(data.frame(
      "W" = round(w_hat, 3), "delta" = grid[, 1], "phi" = grid[, 2]
    ))
  }

  return(list("Grid" = dfw, "W" = w_hat))

}


#' Compute the BPS stacking weights for univariate latent spatial regression model
#'
#' @param Y a "N \times 1" matrix of sample response variable
#' @param X a "N \times P" matrix of sample covarites
#' @param crd_s a "N \times 2" matrix of sample coordinates
#' @param Delta a vector of candidate values for hyperparameter \delta
#' @param Fi a vector of candidate values for hyperparameter \phi
#' @param K if KCV = TRUE, represent the number of desired K-fold
#'
#' @return a matrix with models configuration and the associated weights, and a matrix with the stacking weights
#'
#' @importFrom CVXR Variable Maximize Problem solve
#'
#' @export
# Compute the stacking weights for univariate spatial regression (latent model)
PredictiveStackingWeights_cpp2 <- function(Y, X, crd_s, Delta, Fi, K = 10) {

  ## mandatory packages --------------------------------------------------------
  # library(CVXR, quietly = T)

  ## evaluate the loocv predictive density
  out <- models_dens2(data = list(Y = Y, X = X), coords = crd_s,
                      priors = list(mu_b = matrix(rep(0, p)),
                                    V_b = diag(10, p),
                                    a = 2,
                                    b = 2),
                      hyperpar = list(delta = Delta,
                                      phi = Fi), K = K)

  ## solve the convex optimization problem -------------------------------------

  # cat("\n Recovering the Predictive Stacking weights : \n")
  # tic()
  # declare variable
  scores <- out$out
  weights <- Variable( ncol(scores) )

  # set up minimization problem and solve it
  constraints <- list(weights >= 0, sum(weights) == 1)
  # the constraint for sum up to 1 with positive weights
  f <- Maximize( mean( log( scores %*% weights ) ) )
  problem <- Problem(f, constraints)
  result <- solve(problem, solver = "ECOS_BB") # ECOS, SCS, OSQP

  # define the stacking predictive distribution obtained
  w_hat <- result$getValue(weights)

  # tcop <- toc(quiet = T)
  # message(tcop$callback_msg)

  ## function return
  grid <- expand_grid_cpp(Delta, Fi)
  dfw <- as.matrix(data.frame(
    "W" = round(w_hat, 3), "delta" = grid[, 1], "phi" = grid[, 2]
  ))
  return(list("Grid" = dfw, "W" = w_hat, "beta" = out$beta, "sigma" = out$sigma))

}
