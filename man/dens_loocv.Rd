% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{dens_loocv}
\alias{dens_loocv}
\title{Compute the LOOCV of the density evaluations for fixed values of the hyperparameters}
\usage{
dens_loocv(data, priors, coords, hyperpar)
}
\arguments{
\item{data}{\link{list} two elements: first named \eqn{Y}, second named \eqn{X}}

\item{priors}{\link{list} priors: named \eqn{\mu_b},\eqn{V_b},\eqn{a},\eqn{b}}

\item{coords}{\link{matrix} sample coordinates for X and Y}

\item{hyperpar}{\link{list} two elemets: first named \eqn{\delta}, second named \eqn{\phi}}
}
\value{
\link{vector} posterior predictive density evaluations
}
\description{
Compute the LOOCV of the density evaluations for fixed values of the hyperparameters
}
