% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/code.R
\name{PredictiveStackingWeights_cpp}
\alias{PredictiveStackingWeights_cpp}
\title{Compute the BPS stacking weights for univariate latent spatial regression model}
\usage{
PredictiveStackingWeights_cpp(
  Y,
  X,
  crd_s,
  Delta = NULL,
  Alfa = NULL,
  Fi,
  KCV = F,
  K = 10
)
}
\arguments{
\item{Y}{\link{matrix} \eqn{N \times 1} of sample response variable}

\item{X}{\link{matrix} \eqn{N \times P} matrix of sample covarites}

\item{crd_s}{\link{matrix} \eqn{N \times 2} matrix of sample coordinates}

\item{Delta}{\link{vector} (univariate models) candidate values for hyperparameter \eqn{\delta}}

\item{Alfa}{\link{vector} (mulrivariate models) candidate values for hyperparameter \eqn{\alpha}}

\item{Fi}{\link{vector} candidate values for hyperparameter \eqn{\phi}}

\item{KCV}{\link{boolean} flag to use K-fold cross validation instead of LOOCV (\code{default = FALSE})}

\item{K}{\link{integer} if \code{KCV = TRUE}, represent the number of desired K-fold}
}
\value{
A list A list with the following components:
\item{Grid}{\link{matrix} models configuration and the associated weights}
\item{W}{\link{matrix} stacking weights}
}
\description{
Implement Bayesian predictive stacking for the univariate latent spatial regression models.
}
