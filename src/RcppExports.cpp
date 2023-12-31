// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// arma_dist
arma::mat arma_dist(const arma::mat& X);
RcppExport SEXP _ASMK_arma_dist(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(arma_dist(X));
    return rcpp_result_gen;
END_RCPP
}
// expand_grid_cpp
arma::mat expand_grid_cpp(const arma::vec& x, const arma::vec& y);
RcppExport SEXP _ASMK_expand_grid_cpp(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(expand_grid_cpp(x, y));
    return rcpp_result_gen;
END_RCPP
}
// sample_index
arma::uvec sample_index(const int& size, const int& length, const arma::vec& p);
RcppExport SEXP _ASMK_sample_index(SEXP sizeSEXP, SEXP lengthSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const int& >::type size(sizeSEXP);
    Rcpp::traits::input_parameter< const int& >::type length(lengthSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_index(size, length, p));
    return rcpp_result_gen;
END_RCPP
}
// fit_cpp
List fit_cpp(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar);
RcppExport SEXP _ASMK_fit_cpp(SEXP dataSEXP, SEXP priorsSEXP, SEXP coordsSEXP, SEXP hyperparSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const List& >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    rcpp_result_gen = Rcpp::wrap(fit_cpp(data, priors, coords, hyperpar));
    return rcpp_result_gen;
END_RCPP
}
// post_draws
List post_draws(const List& poster, const int& R, const bool& par, const int& p);
RcppExport SEXP _ASMK_post_draws(SEXP posterSEXP, SEXP RSEXP, SEXP parSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type poster(posterSEXP);
    Rcpp::traits::input_parameter< const int& >::type R(RSEXP);
    Rcpp::traits::input_parameter< const bool& >::type par(parSEXP);
    Rcpp::traits::input_parameter< const int& >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(post_draws(poster, R, par, p));
    return rcpp_result_gen;
END_RCPP
}
// r_pred_cpp
List r_pred_cpp(const List& data, const arma::mat& X_u, const arma::mat& Rphi_s, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster);
RcppExport SEXP _ASMK_r_pred_cpp(SEXP dataSEXP, SEXP X_uSEXP, SEXP Rphi_sSEXP, SEXP d_uSEXP, SEXP d_usSEXP, SEXP hyperparSEXP, SEXP posterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_u(X_uSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Rphi_s(Rphi_sSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type d_u(d_uSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type d_us(d_usSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    Rcpp::traits::input_parameter< const List& >::type poster(posterSEXP);
    rcpp_result_gen = Rcpp::wrap(r_pred_cpp(data, X_u, Rphi_s, d_u, d_us, hyperpar, poster));
    return rcpp_result_gen;
END_RCPP
}
// d_pred_cpp
arma::vec d_pred_cpp(const List& data, const arma::mat& X_u, const arma::vec& Y_u, const arma::mat& Rphi_s, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster);
RcppExport SEXP _ASMK_d_pred_cpp(SEXP dataSEXP, SEXP X_uSEXP, SEXP Y_uSEXP, SEXP Rphi_sSEXP, SEXP d_uSEXP, SEXP d_usSEXP, SEXP hyperparSEXP, SEXP posterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_u(X_uSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y_u(Y_uSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Rphi_s(Rphi_sSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type d_u(d_uSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type d_us(d_usSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    Rcpp::traits::input_parameter< const List& >::type poster(posterSEXP);
    rcpp_result_gen = Rcpp::wrap(d_pred_cpp(data, X_u, Y_u, Rphi_s, d_u, d_us, hyperpar, poster));
    return rcpp_result_gen;
END_RCPP
}
// dens_loocv
arma::vec dens_loocv(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar);
RcppExport SEXP _ASMK_dens_loocv(SEXP dataSEXP, SEXP priorsSEXP, SEXP coordsSEXP, SEXP hyperparSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const List& >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    rcpp_result_gen = Rcpp::wrap(dens_loocv(data, priors, coords, hyperpar));
    return rcpp_result_gen;
END_RCPP
}
// dens_kcv
arma::vec dens_kcv(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const int& K);
RcppExport SEXP _ASMK_dens_kcv(SEXP dataSEXP, SEXP priorsSEXP, SEXP coordsSEXP, SEXP hyperparSEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const List& >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    Rcpp::traits::input_parameter< const int& >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(dens_kcv(data, priors, coords, hyperpar, K));
    return rcpp_result_gen;
END_RCPP
}
// models_dens
arma::mat models_dens(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const bool& useKCV, const int& K);
RcppExport SEXP _ASMK_models_dens(SEXP dataSEXP, SEXP priorsSEXP, SEXP coordsSEXP, SEXP hyperparSEXP, SEXP useKCVSEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const List& >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    Rcpp::traits::input_parameter< const bool& >::type useKCV(useKCVSEXP);
    Rcpp::traits::input_parameter< const int& >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(models_dens(data, priors, coords, hyperpar, useKCV, K));
    return rcpp_result_gen;
END_RCPP
}
// dens_kcv2
List dens_kcv2(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const int& K);
RcppExport SEXP _ASMK_dens_kcv2(SEXP dataSEXP, SEXP priorsSEXP, SEXP coordsSEXP, SEXP hyperparSEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const List& >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    Rcpp::traits::input_parameter< const int& >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(dens_kcv2(data, priors, coords, hyperpar, K));
    return rcpp_result_gen;
END_RCPP
}
// models_dens2
List models_dens2(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const int& K);
RcppExport SEXP _ASMK_models_dens2(SEXP dataSEXP, SEXP priorsSEXP, SEXP coordsSEXP, SEXP hyperparSEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const List& >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    Rcpp::traits::input_parameter< const int& >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(models_dens2(data, priors, coords, hyperpar, K));
    return rcpp_result_gen;
END_RCPP
}
// BPS_SpatialPrediction_cpp
List BPS_SpatialPrediction_cpp(const List& data, const arma::mat& X_u, const List& priors, const arma::mat& coords, const arma::mat& crd_u, const List& hyperpar, const arma::vec& W, const int& R);
RcppExport SEXP _ASMK_BPS_SpatialPrediction_cpp(SEXP dataSEXP, SEXP X_uSEXP, SEXP priorsSEXP, SEXP coordsSEXP, SEXP crd_uSEXP, SEXP hyperparSEXP, SEXP WSEXP, SEXP RSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_u(X_uSEXP);
    Rcpp::traits::input_parameter< const List& >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type crd_u(crd_uSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type W(WSEXP);
    Rcpp::traits::input_parameter< const int& >::type R(RSEXP);
    rcpp_result_gen = Rcpp::wrap(BPS_SpatialPrediction_cpp(data, X_u, priors, coords, crd_u, hyperpar, W, R));
    return rcpp_result_gen;
END_RCPP
}
// BPS_post_draws
arma::mat BPS_post_draws(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const arma::vec& W, const int& R);
RcppExport SEXP _ASMK_BPS_post_draws(SEXP dataSEXP, SEXP priorsSEXP, SEXP coordsSEXP, SEXP hyperparSEXP, SEXP WSEXP, SEXP RSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const List& >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type W(WSEXP);
    Rcpp::traits::input_parameter< const int& >::type R(RSEXP);
    rcpp_result_gen = Rcpp::wrap(BPS_post_draws(data, priors, coords, hyperpar, W, R));
    return rcpp_result_gen;
END_RCPP
}
// fit_latent_cpp
List fit_latent_cpp(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar);
RcppExport SEXP _ASMK_fit_latent_cpp(SEXP dataSEXP, SEXP priorsSEXP, SEXP coordsSEXP, SEXP hyperparSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const List& >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    rcpp_result_gen = Rcpp::wrap(fit_latent_cpp(data, priors, coords, hyperpar));
    return rcpp_result_gen;
END_RCPP
}
// fit_latent_cpp2
List fit_latent_cpp2(const List& data, const List& priors, const arma::mat& dist, const List& hyperpar);
RcppExport SEXP _ASMK_fit_latent_cpp2(SEXP dataSEXP, SEXP priorsSEXP, SEXP distSEXP, SEXP hyperparSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const List& >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type dist(distSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    rcpp_result_gen = Rcpp::wrap(fit_latent_cpp2(data, priors, dist, hyperpar));
    return rcpp_result_gen;
END_RCPP
}
// post_draws_latent
List post_draws_latent(const List& poster, const int& R, const bool& par, const int& p);
RcppExport SEXP _ASMK_post_draws_latent(SEXP posterSEXP, SEXP RSEXP, SEXP parSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type poster(posterSEXP);
    Rcpp::traits::input_parameter< const int& >::type R(RSEXP);
    Rcpp::traits::input_parameter< const bool& >::type par(parSEXP);
    Rcpp::traits::input_parameter< const int& >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(post_draws_latent(poster, R, par, p));
    return rcpp_result_gen;
END_RCPP
}
// BPS_post_draws_latent
List BPS_post_draws_latent(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const arma::vec& W, const int& R);
RcppExport SEXP _ASMK_BPS_post_draws_latent(SEXP dataSEXP, SEXP priorsSEXP, SEXP coordsSEXP, SEXP hyperparSEXP, SEXP WSEXP, SEXP RSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const List& >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type W(WSEXP);
    Rcpp::traits::input_parameter< const int& >::type R(RSEXP);
    rcpp_result_gen = Rcpp::wrap(BPS_post_draws_latent(data, priors, coords, hyperpar, W, R));
    return rcpp_result_gen;
END_RCPP
}
// r_pred_latent_cpp
List r_pred_latent_cpp(const List& data, const arma::mat& X_u, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster, const arma::mat& beta, const arma::mat& sigma);
RcppExport SEXP _ASMK_r_pred_latent_cpp(SEXP dataSEXP, SEXP X_uSEXP, SEXP d_uSEXP, SEXP d_usSEXP, SEXP hyperparSEXP, SEXP posterSEXP, SEXP betaSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_u(X_uSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type d_u(d_uSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type d_us(d_usSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    Rcpp::traits::input_parameter< const List& >::type poster(posterSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(r_pred_latent_cpp(data, X_u, d_u, d_us, hyperpar, poster, beta, sigma));
    return rcpp_result_gen;
END_RCPP
}
// d_pred_latent_cpp
double d_pred_latent_cpp(const List& data, const arma::mat& X_u, const arma::mat& Y_u, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster, const arma::mat& beta, const arma::mat& sigma);
RcppExport SEXP _ASMK_d_pred_latent_cpp(SEXP dataSEXP, SEXP X_uSEXP, SEXP Y_uSEXP, SEXP d_uSEXP, SEXP d_usSEXP, SEXP hyperparSEXP, SEXP posterSEXP, SEXP betaSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_u(X_uSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y_u(Y_uSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type d_u(d_uSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type d_us(d_usSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    Rcpp::traits::input_parameter< const List& >::type poster(posterSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(d_pred_latent_cpp(data, X_u, Y_u, d_u, d_us, hyperpar, poster, beta, sigma));
    return rcpp_result_gen;
END_RCPP
}
// dens_loocv_latent
arma::vec dens_loocv_latent(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar);
RcppExport SEXP _ASMK_dens_loocv_latent(SEXP dataSEXP, SEXP priorsSEXP, SEXP coordsSEXP, SEXP hyperparSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const List& >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    rcpp_result_gen = Rcpp::wrap(dens_loocv_latent(data, priors, coords, hyperpar));
    return rcpp_result_gen;
END_RCPP
}
// dens_kcv_latent
arma::vec dens_kcv_latent(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const int& K);
RcppExport SEXP _ASMK_dens_kcv_latent(SEXP dataSEXP, SEXP priorsSEXP, SEXP coordsSEXP, SEXP hyperparSEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const List& >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    Rcpp::traits::input_parameter< const int& >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(dens_kcv_latent(data, priors, coords, hyperpar, K));
    return rcpp_result_gen;
END_RCPP
}
// models_dens_latent
arma::mat models_dens_latent(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, bool useKCV, int K);
RcppExport SEXP _ASMK_models_dens_latent(SEXP dataSEXP, SEXP priorsSEXP, SEXP coordsSEXP, SEXP hyperparSEXP, SEXP useKCVSEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const List& >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    Rcpp::traits::input_parameter< bool >::type useKCV(useKCVSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(models_dens_latent(data, priors, coords, hyperpar, useKCV, K));
    return rcpp_result_gen;
END_RCPP
}
// BPS_latent_SpatialPrediction_cpp
List BPS_latent_SpatialPrediction_cpp(const List& data, const arma::mat& X_u, const List& priors, const arma::mat& coords, const arma::mat& crd_u, const List& hyperpar, const arma::vec& W, const int& R);
RcppExport SEXP _ASMK_BPS_latent_SpatialPrediction_cpp(SEXP dataSEXP, SEXP X_uSEXP, SEXP priorsSEXP, SEXP coordsSEXP, SEXP crd_uSEXP, SEXP hyperparSEXP, SEXP WSEXP, SEXP RSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_u(X_uSEXP);
    Rcpp::traits::input_parameter< const List& >::type priors(priorsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type crd_u(crd_uSEXP);
    Rcpp::traits::input_parameter< const List& >::type hyperpar(hyperparSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type W(WSEXP);
    Rcpp::traits::input_parameter< const int& >::type R(RSEXP);
    rcpp_result_gen = Rcpp::wrap(BPS_latent_SpatialPrediction_cpp(data, X_u, priors, coords, crd_u, hyperpar, W, R));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_ASMK_arma_dist", (DL_FUNC) &_ASMK_arma_dist, 1},
    {"_ASMK_expand_grid_cpp", (DL_FUNC) &_ASMK_expand_grid_cpp, 2},
    {"_ASMK_sample_index", (DL_FUNC) &_ASMK_sample_index, 3},
    {"_ASMK_fit_cpp", (DL_FUNC) &_ASMK_fit_cpp, 4},
    {"_ASMK_post_draws", (DL_FUNC) &_ASMK_post_draws, 4},
    {"_ASMK_r_pred_cpp", (DL_FUNC) &_ASMK_r_pred_cpp, 7},
    {"_ASMK_d_pred_cpp", (DL_FUNC) &_ASMK_d_pred_cpp, 8},
    {"_ASMK_dens_loocv", (DL_FUNC) &_ASMK_dens_loocv, 4},
    {"_ASMK_dens_kcv", (DL_FUNC) &_ASMK_dens_kcv, 5},
    {"_ASMK_models_dens", (DL_FUNC) &_ASMK_models_dens, 6},
    {"_ASMK_dens_kcv2", (DL_FUNC) &_ASMK_dens_kcv2, 5},
    {"_ASMK_models_dens2", (DL_FUNC) &_ASMK_models_dens2, 5},
    {"_ASMK_BPS_SpatialPrediction_cpp", (DL_FUNC) &_ASMK_BPS_SpatialPrediction_cpp, 8},
    {"_ASMK_BPS_post_draws", (DL_FUNC) &_ASMK_BPS_post_draws, 6},
    {"_ASMK_fit_latent_cpp", (DL_FUNC) &_ASMK_fit_latent_cpp, 4},
    {"_ASMK_fit_latent_cpp2", (DL_FUNC) &_ASMK_fit_latent_cpp2, 4},
    {"_ASMK_post_draws_latent", (DL_FUNC) &_ASMK_post_draws_latent, 4},
    {"_ASMK_BPS_post_draws_latent", (DL_FUNC) &_ASMK_BPS_post_draws_latent, 6},
    {"_ASMK_r_pred_latent_cpp", (DL_FUNC) &_ASMK_r_pred_latent_cpp, 8},
    {"_ASMK_d_pred_latent_cpp", (DL_FUNC) &_ASMK_d_pred_latent_cpp, 9},
    {"_ASMK_dens_loocv_latent", (DL_FUNC) &_ASMK_dens_loocv_latent, 4},
    {"_ASMK_dens_kcv_latent", (DL_FUNC) &_ASMK_dens_kcv_latent, 5},
    {"_ASMK_models_dens_latent", (DL_FUNC) &_ASMK_models_dens_latent, 6},
    {"_ASMK_BPS_latent_SpatialPrediction_cpp", (DL_FUNC) &_ASMK_BPS_latent_SpatialPrediction_cpp, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_ASMK(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
