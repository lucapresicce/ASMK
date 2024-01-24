
#ifndef CODE_H
#define CODE_H

// Function declarations

arma::mat arma_dist(const arma::mat & X);

arma::mat expand_grid_cpp(const arma::vec& x, const arma::vec& y);

arma::uvec sample_index(const int& size, const int& length, const arma::vec& p);

Rcpp::List subset_data(const Rcpp::List& data, int K);

Rcpp::List fit_cpp(const Rcpp::List& data, const Rcpp::List& priors, const arma::mat& coords, const Rcpp::List& hyperpar);

Rcpp::List post_draws(const Rcpp::List& poster, const int& R = 50, const bool& par = false, const int& p = 1);

Rcpp::List r_pred_cpp(const Rcpp::List& data, const arma::mat& X_u, const arma::mat& iRphi_s, const arma::mat& d_u, const arma::mat& d_us, const Rcpp::List& hyperpar, const Rcpp::List& poster);

arma::vec d_pred_cpp(const Rcpp::List& data, const arma::mat& X_u, const arma::vec& Y_u, const arma::mat& iRphi_s, const arma::mat& d_u, const arma::mat& d_us, const Rcpp::List& hyperpar, const Rcpp::List& poster);

arma::vec dens_loocv(const Rcpp::List& data, const Rcpp::List& priors, const arma::mat& coords, const Rcpp::List& hyperpar);

arma::vec dens_kcv(const Rcpp::List& data, const Rcpp::List& priors, const arma::mat& coords, const Rcpp::List& hyperpar, const int& K);

arma::mat models_dens(const Rcpp::List& data, const Rcpp::List& priors, const arma::mat& coords, const Rcpp::List& hyperpar, const bool& useKCV, const int& K = 10);

Rcpp::List dens_kcv2(const Rcpp::List& data, const Rcpp::List& priors, const arma::mat& coords, const Rcpp::List& hyperpar, const int& K);

Rcpp::List models_dens2(const Rcpp::List& data, const Rcpp::List& priors, const arma::mat& coords, const Rcpp::List& hyperpar, const int& K = 10);

SEXP CVXR_opt(const arma::mat& scores);

arma::mat BPSweights_cpp(const Rcpp::List& data, const Rcpp::List& priors, const arma::mat& coords, const Rcpp::List& hyperpar, int K = 5);

Rcpp::List BPSweights_cpp2(const Rcpp::List& data, const Rcpp::List& priors, const arma::mat& coords, const Rcpp::List& hyperpar, int K = 5);

Rcpp::List BPS_combine(const Rcpp::List& fit_list, const int& K, const double& rp = 1);

Rcpp::List BPS_SpatialPrediction_cpp(const Rcpp::List& data, const arma::mat& X_u, const Rcpp::List& priors, const arma::mat& coords, const arma::mat& crd_u, const Rcpp::List& hyperpar, const arma::vec& W, const int& R);

Rcpp::List fast_BPSpred(const Rcpp::List& data, const arma::mat& X_u, const Rcpp::List& priors, const arma::mat& coords, const arma::mat& crd_u, const Rcpp::List& hyperpar, const arma::vec& W, const int& R);

Rcpp::List BPS_SpatialPrediction_cpp2(const Rcpp::List& data, const arma::mat& X_u, const Rcpp::List& priors, const arma::mat& coords, const arma::mat& crd_u, const Rcpp::List& hyperpar, const arma::vec& W, const int& R);

Rcpp::List fast_BPSpred2(const Rcpp::List& data, const arma::mat& X_u, const Rcpp::List& priors, const arma::mat& coords, const arma::mat& crd_u, const Rcpp::List& hyperpar, const arma::vec& W, const int& R);

Rcpp::List fast_BPSpred3(const Rcpp::List& data, const arma::mat& X_u, const Rcpp::List& priors, const arma::mat& coords, const arma::mat& crd_u, const Rcpp::List& hyperpar, const arma::vec& W, const int& R);

Rcpp::List spPredict_ASMK(const Rcpp::List& data, const arma::mat& X_u, const Rcpp::List& priors, const arma::mat& coords, const arma::mat& crd_u, const Rcpp::List& hyperpar, const arma::vec& W, const int& R, const int& J = 1);

Rcpp::List spPredict_ASMK2(const Rcpp::List& data, const arma::mat& X_u, const Rcpp::List& priors, const arma::mat& coords, const arma::mat& crd_u, const Rcpp::List& hyperpar, const arma::vec& W, const int& R, const int& J = 1);

arma::mat BPS_post_draws(const Rcpp::List& data, const Rcpp::List& priors, const arma::mat& coords, const Rcpp::List& hyperpar, const arma::vec& W, const int& R);

Rcpp::List fit_latent_cpp(const Rcpp::List& data, const Rcpp::List& priors, const arma::mat& coords, const Rcpp::List& hyperpar);

Rcpp::List fit_latent_cpp2(const Rcpp::List& data, const Rcpp::List& priors, const arma::mat& dist, const Rcpp::List& hyperpar);

Rcpp::List post_draws_latent(const Rcpp::List& poster, const int& R = 50, const bool& par = false, const int& p = 2);

Rcpp::List BPS_post_draws_latent(const Rcpp::List& data, const Rcpp::List& priors, const arma::mat& coords, const Rcpp::List& hyperpar, const arma::vec& W, const int& R);

Rcpp::List r_pred_latent_cpp(const Rcpp::List& data, const arma::mat& X_u, const arma::mat& d_u, const arma::mat& d_us, const Rcpp::List& hyperpar, const Rcpp::List& poster, const arma::mat& beta, const arma::mat& sigma);

double d_pred_latent_cpp(const Rcpp::List& data, const arma::mat& X_u, const arma::mat& Y_u, const arma::mat& d_u, const arma::mat& d_us, const Rcpp::List& hyperpar, const Rcpp::List& poster, const arma::mat& beta, const arma::mat& sigma);

arma::vec dens_loocv_latent(const Rcpp::List& data, const Rcpp::List& priors, const arma::mat& coords, const Rcpp::List& hyperpar);

arma::vec dens_kcv_latent(const Rcpp::List& data, const Rcpp::List& priors, const arma::mat& coords, const Rcpp::List& hyperpar, const int& K);

arma::mat models_dens_latent(const Rcpp::List& data, const Rcpp::List& priors, const arma::mat& coords, const Rcpp::List& hyperpar, bool useKCV, int K = 10);

Rcpp::List BPS_latent_SpatialPrediction_cpp(const Rcpp::List& data, const arma::mat& X_u, const Rcpp::List& priors, const arma::mat& coords, const arma::mat& crd_u, const Rcpp::List& hyperpar, const arma::vec& W, const int& R);

#endif
