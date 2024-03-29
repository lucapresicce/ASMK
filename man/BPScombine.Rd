% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/code.R
\name{BPScombine}
\alias{BPScombine}
\title{Using Bayesian Predictive Stacking to combine subset models}
\usage{
BPScombine(fit_list)
}
\arguments{
\item{fit_list}{\link{list} of fitted models for all the subsets}
}
\value{
A list with the following components:
\item{W}{\link{matrix} stacking weights of all model configurations}
\item{W_lsit}{\link{list} stackign weights for each model configurations}
}
\description{
Using Bayesian Predictive Stacking to combine subset models
}
