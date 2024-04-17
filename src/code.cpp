#include <RcppArmadillo.h>
#include "code.h"
#include "utilsC.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;


// ##################################################################################################################################################
// CONQUER-STEP FUNCTIONS ###########################################################################################################################
// ##################################################################################################################################################


//' Compute the BPS weights by convex optimization
//'
//' @param scores [matrix] \eqn{N \times K} of expected predictive density evaluations for the K models considered
//'
//' @return conv_opt [function] to perform convex optimiazion with CVXR R package
//'
// [[Rcpp::export]]
SEXP CVXR_opt(const arma::mat& scores) {

  // Evaluate R expression in the package environment
  Rcpp::Environment pkg_env = Rcpp::Environment::namespace_env("ASMK");
  Rcpp::Function conv_opt = pkg_env["conv_opt"];

  // Apply convex optimization by CVXR as an R function
  return conv_opt(Named("scores", scores));
}


//' Combine subset models wiht BPS
//'
//' @param fit_list [list] K fitted model outputs composed by two elements each: first named \eqn{epd}, second named \eqn{W}
//' @param K [integer] number of folds
//' @param rp [double] percentage of observations to take into account for optimization (\code{default=1})
//'
//' @return [matrix] posterior predictive density evaluations (each columns represent a different model)
//' @export
// [[Rcpp::export]]
List BPS_combine(const List& fit_list, const int& K, const double& rp) {

  // Weights list
  List W_list(K);
  for (int i = 0; i < K; ++i) {
    List W_i = fit_list[i];
    W_list[i] = as<arma::mat>(W_i[1]);
  }

  // Epd list
  List epd_0 = fit_list[0];
  arma::mat out3 = as<arma::mat>(epd_0[2]);
  for (int i = 1; i < K; ++i) {
    List epd_i = fit_list[i];
    out3 = join_vert(out3, as<arma::mat>(epd_i[2]));
  }

  // Calculate weighted scores
  int n = out3.n_rows;
  int N = floor(rp * n);
  arma::vec pr = arma::vec(n, arma::fill::ones) / n;
  arma::uvec red_ind = sample_index(n, N, pr);
  arma::mat out4(red_ind.n_elem, K, arma::fill::zeros);
  for (int k = 0; k < K; ++k) {
    out4.col(k) = out3.rows(red_ind) * as<arma::mat>(W_list[k]);
  }

  // Convex optimization
  arma::mat Wbps = as<arma::mat>(CVXR_opt(out4));

  // Remove small weights
  double threshold = 1.0 / (2.0 * Wbps.n_elem);
  Wbps.elem(find(Wbps < threshold)).zeros();
  Wbps /= sum(Wbps, 0).eval()(0, 0);

  // Return results as a List
  return List::create(Named("W") = Wbps,
                      Named("W_list") = W_list);
}


//' Combine subset models wiht Pseudo-BMA
//'
//' @param fit_list [list] K fitted model outputs composed by two elements each: first named \eqn{epd}, second named \eqn{W}
//'
//' @return [matrix] posterior predictive density evaluations (each columns represent a different model)
//' @export
// [[Rcpp::export]]
List BPS_PseudoBMA(const List& fit_list) {

  // Number of subsets
  int K = fit_list.size();

  // Collect model pd and weights list
  List W_list(K);
  List out3(K);
  for (int i = 0; i < K; ++i) {
    List ls = fit_list[i];
    W_list[i] = as<arma::mat>(ls[1]);
    out3[i] = as<arma::mat>(ls[2]);
  }

  int nn = as<arma::mat>(out3[0]).n_rows;
  arma::mat out4(K, K, arma::fill::zeros);

  // Compute log likelihood and sum over models
  for (int j = 0; j < K; ++j) {
    arma::mat sp(nn, K, arma::fill::zeros);

    for (int k = 0; k < K; ++k) {
      arma::mat out3_k = as<arma::mat>(out3[k]);
      arma::mat W_j = W_list[j];
      sp.col(k) = log(out3_k * W_j);
    }

    // Handle non-finite values
    sp.elem(find_nonfinite(sp)).zeros();
    out4.row(j) = sum(sp, 0);
  }


  // Compute elpd
  out4 = out4/nn;
  arma::mat out8 = trans(mean(out4, 0));


  // Compute pseudo-BMA weights
  arma::mat Waic = exp(out8);
  Waic = Waic/sum(Waic, 0).eval()(0, 0);

  // Noise threshold
  Waic.elem(find(Waic <= 1 / (2 * Waic.size()))).zeros();
  Waic /= sum(Waic, 0).eval()(0, 0);

  return List::create(Named("W") = Waic,
                      Named("W_list") = W_list);
}


// ##################################################################################################################################################
// UNIVARIATE LATENT MODELS #########################################################################################################################
// ##################################################################################################################################################


//' Compute the parameters for the posteriors distribution of \eqn{\beta} and \eqn{\Sigma} (i.e. updated parameters)
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param priors [list] priors: named \eqn{\mu_b},\eqn{V_b},\eqn{a},\eqn{b}
//' @param coords [matrix] sample coordinates for X and Y
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//'
//' @return [list] posterior update parameters
//'
// [[Rcpp::export]]
List fit_cpp(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar) {

  // Unpack data and priors
  vec Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  vec mu_b = as<vec>(priors["mu_b"]);
  arma::mat V_b = as<arma::mat>(priors["V_b"]);
  double b = as<double>(priors["b"]);
  double a = as<double>(priors["a"]);
  double delta = as<double>(hyperpar["delta"]);
  double phi = as<double>(hyperpar["phi"]);

  int n = Y.n_rows;
  int p = X.n_cols;
  // arma::mat d_s = C_dist(coords);
  arma::mat d_s = arma_dist(coords);
  arma::mat Rphi_s = exp(-phi * d_s);

  // build the aumentend linear sistem
  vec zer_n(n, arma::fill::zeros);
  vec Y_star = join_vert(Y, mu_b, zer_n);

  arma::mat Zer_np(n, p, arma::fill::zeros);
  arma::mat Zer_pn = trans(Zer_np);
  arma::mat X_1 = join_vert(X, eye<arma::mat>(p, p), Zer_np);
  arma::mat X_2 = join_vert(eye<arma::mat>(n, n), Zer_pn, eye<arma::mat>(n, n));
  arma::mat X_star = join_horiz(X_1, X_2);

  // arma::mat V_1 = join_vert(delta*eye<arma::mat>(n, n), arma::mat(p+n, n, arma::fill::zeros));
  // arma::mat V_2 = join_vert(arma::mat(n, p, arma::fill::zeros), V_b, arma::mat(n, p, arma::fill::zeros));
  // arma::mat V_3 = join_vert(arma::mat(n+p, n, arma::fill::zeros), Rphi_s);
  // arma::mat V_star = join_horiz(V_1, V_2, V_3);

  arma::mat iV_1 = join_vert((1/delta)*eye<arma::mat>(n, n), arma::mat(p+n, n, arma::fill::zeros));
  arma::mat iV_2 = join_vert(Zer_np, arma::inv(V_b), Zer_np);
  arma::mat iRphi_s = arma::inv(Rphi_s);
  arma::mat iV_3 = join_vert(arma::mat(n+p, n, arma::fill::zeros), iRphi_s);
  arma::mat iV_star = join_horiz(iV_1, iV_2, iV_3);

  // Precompute some reusable values
  arma::mat tX_star = trans(X_star);
  // arma::mat iV_star = arma::inv(V_star);

  // conjugate posterior parameters
  arma::mat iM_star = tX_star * iV_star * X_star;
  arma::mat M_star = arma::inv(iM_star);
  arma::mat tXVY = tX_star * iV_star * Y_star;
  arma::vec gamma_hat = M_star * tXVY;

  arma::mat Xgam = X_star * gamma_hat;
  arma::mat SXY = Y_star - (Xgam);
  arma::mat tSXY = trans(SXY);
  // arma::mat SXY = Y_star - (X_star * gamma_hat);
  // arma::mat tSXY = trans(Y_star - (X_star * gamma_hat));

  arma::mat bb = tSXY * iV_star * SXY;
  double b_star = b + 0.5 * as_scalar(bb);

  double a_star = a + (n/2);

  // Return results as an R list
  return List::create(Named("M_star") = M_star,
                      Named("gamma_hat") = gamma_hat,
                      Named("b_star") = b_star,
                      Named("a_star") = a_star,
                      Named("iRphi_s") = iRphi_s);
}


//' Sample R draws from the posterior distributions
//'
//' @param poster [list] output from \code{fit_cpp} function
//' @param R [integer] number of posterior samples
//' @param par [boolean] if \code{TRUE} only \eqn{\beta} and \eqn{\sigma^2} are sampled (\eqn{\omega} is omitted)
//' @param p [integer] if \code{par = TRUE}, it specifies the column number of \eqn{X}
//'
//' @return [list] posterior samples
//'
// [[Rcpp::export]]
List post_draws(const List& poster, const int& R, const bool& par, const int& p) {

  // Extract update posterior parameters
  arma::mat M_star = as<arma::mat>(poster["M_star"]);
  arma::vec gamma_hat = as<arma::vec>(poster["gamma_hat"]);
  double a_star = as<double>(poster["a_star"]);
  double b_star = as<double>(poster["b_star"]);

  if(par) {
    gamma_hat = gamma_hat.subvec(0, p-1);
    arma::uvec ind_p = arma::linspace<arma::uvec>(0, p-1, p);
    M_star = M_star.submat(ind_p, ind_p);
  }

  // Environment
  Rcpp::Environment mniw = Rcpp::Environment::namespace_env("mniw");
  Rcpp::Function rmNorm_R = mniw["rmNorm"];
  Rcpp::Function riwish_R = mniw["riwish"];

  // Initialize return objects
  arma::vec out_s(R);
  arma::mat out_b(R, gamma_hat.n_elem);

  for (int r = 0; r < R; r++) {
    double s = as<double>(riwish_R(Named("n", 1), Named("nu", a_star), Named("Psi", b_star)));
    arma::vec b = as<arma::vec>(rmNorm_R(Named("n", 1), Named("mu", trans(gamma_hat)), Named("Sigma", s * M_star)));

    out_s(r) = s;
    out_b.row(r) = trans(b);
  }

  return List::create(Named("Sigmas") = out_s,
                      Named("Betas") = out_b);
}


//' Draw from the conditional posterior predictive for a set of unobserved covariates
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param X_u [matrix] unobserved instances covariate matrix
//' @param iRphi_s [matrix] inverse of the sample correlation matrix
//' @param d_u [matrix] unobserved instances distance matrix
//' @param d_us [matrix] cross-distance between unobserved and observed instances matrix
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//' @param poster [list] output from \code{fit_cpp} function
//'
//' @return [list] posterior predictive samples
//'
// [[Rcpp::export]]
Rcpp::List r_pred_cpp(const List& data, const arma::mat& X_u, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster, const int& R) {

  // Unpack data, posterior sample and hyperparameters
  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  double delta = as<double>(hyperpar["delta"]);
  double phi = as<double>(hyperpar["phi"]);
  arma::mat iRphi_s = as<arma::mat>(poster["iRphi_s"]);
  arma::mat M_star = as<arma::mat>(poster["M_star"]);
  arma::vec gamma_hat = as<arma::vec>(poster["gamma_hat"]);
  double a_star = as<double>(poster["a_star"]);
  double b_star = as<double>(poster["b_star"]);

  // extract info from data
  int m = X_u.n_rows;
  int p = X_u.n_cols;
  int n = d_us.n_rows-m;

  // R environment
  Rcpp::Environment mvtnorm = Rcpp::Environment::namespace_env("mvtnorm");
  Rcpp::Function rmvt_R = mvtnorm["rmvt"];

  // (exponential) covariance matrices
  arma::mat Rphi_u = exp(-phi * d_u);
  arma::mat Rphi_us = exp(-phi * d_us.submat(0, m, m-1, m+n-1));

  // compute posterior predictive parameters
  arma::mat JR = Rphi_us * iRphi_s;
  arma::mat Zer_mp(m, p, arma::fill::zeros);
  arma::mat W_1 = join_vert(Zer_mp, X_u);
  arma::mat W_2 = join_vert(JR, JR);
  arma::mat W = join_horiz(W_1, W_2);
  arma::vec mu_tilde = W * gamma_hat;

  arma::mat V_z = Rphi_u - JR * trans(Rphi_us);
  arma::mat M2_1 = join_horiz(V_z, V_z);
  arma::mat M2_2 = join_horiz(V_z, V_z + (delta*eye<arma::mat>(m, m)));
  arma::mat M2 = join_vert(M2_1, M2_2);
  arma::mat M_tilde = (W * M_star * trans(W)) + M2;

  // degrees of freedom
  double t_df = 2*a_star;
  double scale_ratio = b_star/a_star;

  // posterior predictive sample
  arma::mat res = as<arma::mat>(rmvt_R(Named("n", R), Named("sigma", scale_ratio*M_tilde), Named("delta", mu_tilde), Named("df", t_df)));

  return List::create(Named("W_u") = res.cols(0, m-1),
                      Named("Y_u") = res.cols(m, (2*m)-1));

}


//' Evaluate the density of a set of unobserved response with respect to the conditional posterior predictive
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param X_u [matrix] unobserved instances covariate matrix
//' @param Y_u [matrix] unobserved instances response matrix
//' @param iRphi_s [matrix] inverse of the sample correlation matrix
//' @param d_u [matrix] unobserved instances distance matrix
//' @param d_us [matrix] cross-distance between unobserved and observed instances matrix
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//' @param poster [list] output from \code{fit_cpp} function
//'
//' @return [vector] posterior predictive density evaluations
//'
// [[Rcpp::export]]
double d_pred_cpp(const List& data, const arma::mat& X_u, const arma::vec& Y_u, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster) {

  // Unpack data, posterior sample and hyperparameters
  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  double delta = as<double>(hyperpar["delta"]);
  double phi = as<double>(hyperpar["phi"]);
  arma::mat iRphi_s = as<arma::mat>(poster["iRphi_s"]);
  arma::mat M_star = as<arma::mat>(poster["M_star"]);
  arma::vec gamma_hat = as<arma::vec>(poster["gamma_hat"]);
  double a_star = as<double>(poster["a_star"]);
  double b_star = as<double>(poster["b_star"]);

  // extract info from data
  int m = X_u.n_rows;
  int n = d_us.n_rows-m;

  // R environment
  Rcpp::Environment mvtnorm = Rcpp::Environment::namespace_env("mvtnorm");
  Rcpp::Function dmvt_R = mvtnorm["dmvt"];

  // (exponential) covariance matrices
  arma::mat Rphi_u = exp(-phi * d_u);
  arma::mat Rphi_us = exp(-phi * d_us.submat(0, m, m-1, m+n-1));

  // compute posterior predictive parameters
  arma::mat JR = Rphi_us * iRphi_s;
  arma::mat W = join_horiz(X_u, JR);
  arma::vec mu_tilde = W * gamma_hat;

  arma::mat V_z = Rphi_u - JR * trans(Rphi_us);
  arma::mat M2 = V_z + (delta*eye<arma::mat>(m, m));
  arma::mat M_tilde = (W * M_star * trans(W)) + M2;

  // degrees of freedom
  double t_df = a_star;
  double scale_ratio = b_star/a_star;

  // Rcout << "delta: " << mu_tilde << std::endl;
  // Rcout << "sigma: " << scale_ratio*M_tilde << std::endl;

  // posterior predictive density
  double P_u = as_scalar(as<arma::vec>(dmvt_R(Named("x", trans(Y_u)), Named("sigma", scale_ratio*M_tilde), Named("delta", mu_tilde), Named("df", t_df), Named("log", false))));

  return P_u;
}

//' Compute the LOOCV of the density evaluations for fixed values of the hyperparameters
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param priors [list] priors: named \eqn{\mu_b},\eqn{V_b},\eqn{a},\eqn{b}
//' @param coords [matrix] sample coordinates for X and Y
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//'
//' @return [vector] posterior predictive density evaluations
//'
// [[Rcpp::export]]
arma::vec dens_loocv(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar) {

  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  int n = Y.n_rows;
  arma::vec predictions(n);

  for (int i = 0; i < n; i++) {

    // Rcout << "i: " << i << std::endl;

    // extract the training data (excluding the i-th row) (create a mask to exclude the i-th row)
    arma::mat Ytr, Xtr, coords_i;
    if (i == 0) {
      // Exclude the 0-th row
      Ytr = Y.rows(1, n - 1);
      Xtr = X.rows(1, n - 1);
      coords_i = coords.rows(1, n - 1);
    } else if (i == 1) {
      // Exclude the 1-th row
      Ytr = join_vert(Y.row(0), Y.rows(2, n - 1));
      Xtr = join_vert(X.row(0), X.rows(2, n - 1));
      coords_i = join_vert(coords.row(0), coords.rows(2, n - 1));
    } else if (i == n - 1) {
      // Exclude the last row
      Ytr = Y.rows(0, n - 2);
      Xtr = X.rows(0, n - 2);
      coords_i = coords.rows(0, n - 2);
    } else {
      // Exclude the i-th row for i > 1
      Ytr = join_vert(Y.rows(0, i - 1), Y.rows(i + 1, n - 1));
      Xtr = join_vert(X.rows(0, i - 1), X.rows(i + 1, n - 1));
      coords_i = join_vert(coords.rows(0, i - 1), coords.rows(i + 1, n - 1));
    }
    List data_i = List::create(
      Named("Y") = Ytr,
      Named("X") = Xtr);

    // extract the test data
    arma::mat crd_i = coords.row(i);
    arma::mat X_i = X.row(i);
    arma::mat Y_i = Y.row(i);

    // Fit your model on the training data
    List poster_i = fit_cpp(data_i, priors, coords_i, hyperpar);

    // evaluate predictive density
    arma::mat d_i = arma_dist(crd_i);
    arma::mat crd_is = join_cols(crd_i, coords_i);
    arma::mat d_is = arma_dist(crd_is);

    double dens = d_pred_cpp(data_i, X_i, Y_i, d_i, d_is, hyperpar, poster_i);

    // Store the prediction
    predictions[i] = dens;
  }

  return predictions;

}


//' Compute the KCV of the density evaluations for fixed values of the hyperparameters
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param priors [list] priors: named \eqn{\mu_b},\eqn{V_b},\eqn{a},\eqn{b}
//' @param coords [matrix] sample coordinates for X and Y
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//' @param K [integer] number of folds
//'
//' @return [vector] posterior predictive density evaluations
//'
// [[Rcpp::export]]
arma::vec dens_kcv(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const int& K) {

  // unpack data
  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  int n = Y.n_rows;
  arma::vec predictions(n, arma::fill::zeros);

  // Create a random permutation of indices from 1 to K
  arma::vec p = arma::vec(K, arma::fill::ones) / K;
  arma::uvec foldIndices = sample_index(K, n, p);

  for (int k = 0; k < K; k++) {

    // Define the indices for the current fold
    arma::uvec testSet = find(foldIndices == (k + 1)); // Find indices that match the current fold number

    // Create a boolean vector to identify the training set for this fold
    arma::uvec trainSet = find(foldIndices != (k + 1)); // Find indices that do not match the current fold number

    // Extract the training data for the current fold
    arma::mat Ytr = Y.rows(trainSet);
    arma::mat Xtr = X.rows(trainSet);
    arma::mat coords_tr = coords.rows(trainSet);

    List data_tr = List::create(
      Named("Y") = Ytr,
      Named("X") = Xtr);

    // Extract the test data for the current fold
    arma::mat Y_test = Y.rows(testSet);
    arma::mat X_test = X.rows(testSet);
    arma::mat coords_test = coords.rows(testSet);

    // Fit your model on the training data
    List poster_k = fit_cpp(data_tr, priors, coords_tr, hyperpar);

    // evaluate predictive density for the test set of the current fold
    for (uword i = 0; i < testSet.n_elem; i++) {

      arma::mat crd_i = coords_test.row(i);
      arma::mat X_i = X_test.row(i);
      arma::mat Y_i = Y_test.row(i);
      arma::mat d_i = arma_dist(crd_i);

      arma::mat crd_is = join_cols(crd_i, coords_tr);
      arma::mat d_is = arma_dist(crd_is);

      double dens = d_pred_cpp(data_tr, X_i, Y_i, d_i, d_is, hyperpar, poster_k);
      predictions(testSet(i)) = dens;
    }
  }

  return predictions;

}


//' Return the CV predictive density evaluations for all the model combinations
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param priors [list] priors: named \eqn{\mu_b},\eqn{V_b},\eqn{a},\eqn{b}
//' @param coords [matrix] sample coordinates for X and Y
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//' @param useKCV [boolean] if \code{TRUE} K-fold cross validation is used instead of LOOCV (no \code{default})
//' @param K [integer] number of folds
//'
//' @return [matrix] posterior predictive density evaluations (each columns represent a different model)
//'
// [[Rcpp::export]]
arma::mat models_dens(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const bool& useKCV, const int& K) {

  // build the grid of hyperparameters
  arma::vec Delta = hyperpar["delta"];
  arma::vec Fi = hyperpar["phi"];
  arma::mat Grid = expand_grid_cpp(Delta, Fi);
  int k = Grid.n_rows;

  arma::mat out;

  for(int j = 0; j < k; j++) {

    // identify the model
    arma::rowvec hpar = Grid.row(j);
    double delt = hpar[0];
    double fi = hpar[1];
    List hmod = List::create(
      Named("delta") = delt,
      Named("phi") = fi);

    // Call the appropriate function based on the "useKCV" argument
    arma::vec out_j;
    if (useKCV) {
      out_j = dens_kcv(data, priors, coords, hmod, K);
    } else {
      out_j = dens_loocv(data, priors, coords, hmod);
    }

    out =  join_horiz(out, out_j);

  }

  return out;
}


//' Compute the BPS weights by convex optimization
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param priors [list] priors: named \eqn{\mu_b},\eqn{V_b},\eqn{a},\eqn{b}
//' @param coords [matrix] sample coordinates for X and Y
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//' @param K [integer] number of folds
//' @param g [integer] number of draws for MC density approximations
//'
//' @return [matrix] posterior predictive density evaluations (each columns represent a different model)
//' @export
// [[Rcpp::export]]
List BPS_weights(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, int K) {

  // compute predictive density evaluations
  arma::mat out = models_dens(data, priors, coords, hyperpar, true, K);

  // compute the weights
  arma::mat weights = as<arma::mat>(CVXR_opt(out));
  weights.elem(find(weights <= 0)).zeros();
  weights /= sum(weights, 0).eval()(0, 0);

  // return the list (BPS weights, Grid, and predictive density evaluations)
  arma::vec Delta = hyperpar["delta"];
  arma::vec Fi = hyperpar["phi"];
  arma::mat Grid = expand_grid_cpp(Delta, Fi);
  arma::mat res = join_horiz(weights, Grid);

  List Res = List::create(
    Named("Grid") = res,
    Named("W") = weights,
    Named("epd") = out
  );

  return Res;
}


//' Compute the BPS spatial prediction given a set of stacking weights
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param X_u [matrix] unobserved instances covariate matrix
//' @param priors [list] priors: named \eqn{\mu_b},\eqn{V_b},\eqn{a},\eqn{b}
//' @param coords [matrix] sample coordinates for X and Y
//' @param crd_u [matrix] unboserved instances coordinates
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//' @param W [matrix] set of stacking weights
//' @param R [integer] number of desired samples
//'
//' @return [list] BPS posterior predictive samples
//' @export
// [[Rcpp::export]]
List BPS_pred(const List& data, const arma::mat& X_u, const List& priors, const arma::mat& coords, const arma::mat& crd_u, const List& hyperpar, const arma::vec& W, const int& R) {

  arma::mat W_pred;
  arma::mat Y_pred;

  // compute distance matrices
  arma::mat d_u = arma_dist(crd_u);
  arma::mat crd_us = join_cols(crd_u, coords);
  arma::mat d_us = arma_dist(crd_us);

  // build the grid of hyperparameters
  arma::vec Delta = hyperpar["delta"];
  arma::vec Fi = hyperpar["phi"];
  arma::mat Grid = expand_grid_cpp(Delta, Fi);
  int k = Grid.n_rows;

  for(int r = 0; r < R; r++) {

    // sample the model
    arma::uvec kmod = sample_index(k, 1, W);
    arma::uword k_mod = kmod(0);

    // identify the k-th model
    arma::rowvec hpar = Grid.row(k_mod);
    double delt = hpar[0];
    double fi = hpar[1];
    List hmod = List::create(
      Named("delta") = delt,
      Named("phi") = fi);

    // fit your model on the training data
    List poster = fit_cpp(data, priors, coords, hmod);

    // draw from posterior predictive
    List pred_R = r_pred_cpp(data, X_u, d_u, d_us, hmod, poster, 1);

    arma::vec W_pred_r = pred_R["W_u"];
    W_pred =  join_horiz(W_pred, W_pred_r);

    arma::vec Y_pred_r = pred_R["Y_u"];
    Y_pred =  join_horiz(Y_pred, Y_pred_r);

  }

  // return pred;
  return List::create(Named("W_hat") = W_pred,
                      Named("Y_hat") = Y_pred);

}


//' Compute the BPS posterior samples given a set of stacking weights
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param priors [list] priors: named \eqn{\mu_b},\eqn{V_b},\eqn{a},\eqn{b}
//' @param coords [matrix] sample coordinates for X and Y
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//' @param W [matrix] set of stacking weights
//' @param R [integer] number of desired samples
//'
//' @return [matrix] BPS posterior samples
//' @export
// [[Rcpp::export]]
arma::mat BPS_postdraws(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const arma::vec& W, const int& R) {

  // initialize return object
  arma::mat X = as<arma::mat>(data["X"]);
  int p = X.n_cols;
  arma::mat Draws;

  for(int r = 0; r < R; r++) {

    // build the grid of hyperparameters
    arma::vec Delta = hyperpar["delta"];
    arma::vec Fi = hyperpar["phi"];
    arma::mat Grid = expand_grid_cpp(Delta, Fi);
    int k = Grid.n_rows;

    // sample the model
    arma::uvec kmod = sample_index(k, 1, W);
    arma::uword k_mod = kmod(0);

    // identify the k-th model
    arma::rowvec hpar = Grid.row(k_mod);
    double delt = hpar[0];
    double fi = hpar[1];
    List hmod = List::create(
      Named("delta") = delt,
      Named("phi") = fi);

    // fit your model on the training data
    List poster = fit_cpp(data, priors, coords, hmod);

    // posterior draws
    List post = post_draws(poster, 1, true, p);

    arma::vec s = as<arma::vec>(post["Sigmas"]);
    arma::vec b = as<arma::vec>(post["Betas"]);

    // Rcout << s << std::endl;
    // Rcout << b << std::endl;
    arma::vec smp = join_vert(s, b);

    // Rcout << Draws << std::endl;
    // Rcout << trans(smp) << std::endl;
    Draws =  join_vert(Draws, trans(smp));

  }

  // return pred;
  return Draws;

}


//' Perform prediction for ASMK models - loop over prediction set
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param X_u [matrix] unobserved instances covariate matrix
//' @param priors [list] priors: named \eqn{\mu_b},\eqn{V_b},\eqn{a},\eqn{b}
//' @param coords [matrix] sample coordinates for X and Y
//' @param crd_u [matrix] unboserved instances coordinates
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//' @param W [matrix] set of stacking weights
//' @param R [integer] number of desired samples
//' @param J [integer] number of desired partition of prediction set
//'
//' @return [list] BPS posterior predictive samples
//'
// [[Rcpp::export]]
List spPredict_ASMK(const List& data, const arma::mat& X_u, const List& priors, const arma::mat& coords, const arma::mat& crd_u, const List& hyperpar, const arma::vec& W, const int& R, const int& J) {

  // Subset number for prediction data
  int u = X_u.n_rows;
  int set_size_u = floor(u / J);
  arma::vec sets_u = arma::regspace(0, set_size_u, set_size_u * (J - 1));
  sets_u = join_vert(sets_u, arma::vec(1, arma::fill::ones) * u);

  arma::mat out_rW;  // Initialize an empty matrix for W
  arma::mat out_rY;  // Initialize an empty matrix for Y

  for (size_t l = 0; l < sets_u.n_elem - 1; ++l) {

    // Define the current set
    arma::uvec set_j = arma::regspace<arma::uvec>(sets_u(l), sets_u(l + 1) - 1);

    // Extract the current subset from X
    arma::mat sub_X_u = X_u.rows(set_j);
    arma::mat sub_crd_u = crd_u.rows(set_j);

    // Call the user-defined function FUN
    List out_j = BPS_pred(data, sub_X_u, priors, coords, sub_crd_u, hyperpar, W, R);

    // Append the results to the output list
    out_rW = join_vert(out_rW, as<arma::mat>(out_j["W_hat"]));
    out_rY = join_vert(out_rY, as<arma::mat>(out_j["Y_hat"]));
  }

  return List::create(Named("W_hat") = out_rW,
                      Named("Y_hat") = out_rY);

}


// ##################################################################################################################################################
// MULTIVARIATE LATENT MODELS #######################################################################################################################
// ##################################################################################################################################################


//' Compute the parameters for the posteriors distribution of \eqn{\beta} and \eqn{\Sigma} (i.e. updated parameters)
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param priors [list] priors: named \eqn{\mu_B},\eqn{V_r},\eqn{\Psi},\eqn{\nu}
//' @param coords [matrix] sample coordinates for X and Y
//' @param hyperpar [list] two elemets: first named \eqn{\alpha}, second named \eqn{\phi}
//'
//' @return [list] posterior update parameters
//'
// [[Rcpp::export]]
List fit_cpp_MvT(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar) {

  // Unpack data and priors
  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  arma::mat mu_B = as<arma::mat>(priors["mu_B"]);
  arma::mat V_r = as<arma::mat>(priors["V_r"]);
  arma::mat Psi = as<arma::mat>(priors["Psi"]);
  double nu = as<double>(priors["nu"]);
  double alpha = as<double>(hyperpar["alpha"]);
  double phi = as<double>(hyperpar["phi"]);

  int n = Y.n_rows;
  arma::mat d_s = arma_dist(coords);
  arma::mat Rphi_s = exp(-phi * d_s);

  // Precompute some reusable values
  double a = (alpha / (1 - alpha));
  arma::mat tX = trans(X);
  arma::mat iV_r = arma::inv(V_r);
  arma::mat iR_s = arma::inv(Rphi_s);

  // Compute posterior updating
  arma::mat V_B = a * tX * X + iV_r;
  arma::mat V_BW = a * tX;
  arma::mat V_WB = trans(V_BW);
  arma::mat V_W = iR_s + (a * eye<arma::mat>(n, n));

  arma::mat V_star1 = join_horiz(V_B, V_BW);
  arma::mat V_star2 = join_horiz(V_WB, V_W);
  arma::mat iV_star = join_vert( V_star1, V_star2);
  arma::mat V_star = arma::inv(iV_star);

  arma::mat M = join_vert( (a * tX * Y) + (iV_r * mu_B) , a * Y );
  arma::mat mu_star = V_star * M;

  arma::mat aYY = (a * trans(Y) * Y);
  arma::mat mbVrmb = (trans(mu_B) * iV_r * mu_B);
  arma::mat msVsms = (trans(mu_star) * iV_star * mu_star);
  arma::mat Psi_star = Psi + aYY  + mbVrmb - msVsms;
  double nu_star = nu + n;

  // Return results as an R list
  return List::create(Named("V_star") = V_star,
                      Named("mu_star") = mu_star,
                      Named("Psi_star") = Psi_star,
                      Named("nu_star") = nu_star,
                      Named("iRphi_s") = iR_s);
}


//' Sample R draws from the posterior distributions
//'
//' @param poster [list] output from \code{fit_cpp} function
//' @param R [integer] number of posterior samples
//' @param par [boolean] if \code{TRUE} only \eqn{\beta} and \eqn{\sigma^2} are sampled (\eqn{\omega} is omitted)
//' @param p [integer] if \code{par = TRUE}, it specifies the column number of \eqn{X}
//'
//' @return [list] posterior samples
//'
// [[Rcpp::export]]
List post_draws_MvT(const List& poster, const int& R, const bool& par, const int& p) {

  // posterior draws
  arma::mat mu_star = as<arma::mat>(poster["mu_star"]);
  arma::mat V_star = as<arma::mat>(poster["V_star"]);
  arma::mat Psi_star = as<arma::mat>(poster["Psi_star"]);
  double nu_star = as<double>(poster["nu_star"]);

  if(par) {
    mu_star = mu_star.rows(0, p-1);
    arma::uvec ind_p = arma::linspace<arma::uvec>(0, p-1, p);
    V_star = V_star.submat(ind_p, ind_p);
  }

  // Environment
  Rcpp::Environment mniw = Rcpp::Environment::namespace_env("mniw");
  Rcpp::Function riwish_R = mniw["riwish"];
  Rcpp::Function rMNorm_R = mniw["rMNorm"];

  // Initialize return objects
  List out(R);

  for (int r = 0; r < R; r++) {
    arma::mat s = as<arma::mat>(riwish_R(Named("n", 1), Named("nu", nu_star), Named("Psi", Psi_star)));
    arma::mat b = as<arma::mat>(rMNorm_R(Named("n", 1), Named("Lambda", mu_star), Named("SigmaR", V_star), Named("SigmaC", s)));

    // Rcout << "s : " << s << std::endl;
    // Rcout << "b : " << b << std::endl;

    out(r) = List::
      create(Named("beta") = b,
             Named("sigma") = s);
  }

  return out;
}


//' Draw from the conditional posterior predictive for a set of unobserved covariates
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param X_u [matrix] unobserved instances covariate matrix
//' @param d_u [matrix] unobserved instances distance matrix
//' @param d_us [matrix] cross-distance between unobserved and observed instances matrix
//' @param hyperpar [list] two elemets: first named \eqn{\alpha}, second named \eqn{\phi}
//' @param poster [list] output from \code{fit_cpp} function
//' @param R [integer] number of posterior predictive samples
//'
//' @return [list] posterior predictive samples
//'
// [[Rcpp::export]]
List r_pred_cpp_MvT(const List& data, const arma::mat& X_u, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster, const int& R) {

  // Unpack data, posterior sample and hyperparameters
  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  double alpha = as<double>(hyperpar["alpha"]);
  double phi = as<double>(hyperpar["phi"]);
  arma::mat iRphi_s = as<arma::mat>(poster["iRphi_s"]);
  arma::mat V_star = as<arma::mat>(poster["V_star"]);
  arma::mat mu_star = as<arma::mat>(poster["mu_star"]);
  arma::mat Psi_star = as<arma::mat>(poster["Psi_star"]);
  double nu_star = as<double>(poster["nu_star"]);

  // extract info from data
  int m = X_u.n_rows;
  int n = d_us.n_rows-m;
  int p = X_u.n_cols;

  // covariance matrices
  arma::mat Rphi_u = exp(-phi * d_u);
  arma::mat Rphi_us = exp(-phi * d_us.submat(0, m, m-1, m+n-1));

  // sampling environment
  Rcpp::Environment mniw = Rcpp::Environment::namespace_env("mniw");
  Rcpp::Function rMT_R = mniw["rMT"];

  // compute posterior predictive parameter
  arma::mat Mu = Rphi_us * iRphi_s;
  arma::mat Zer_mp(m, p, arma::fill::zeros);
  arma::mat Mgamma_1 = join_vert(Zer_mp, X_u);
  arma::mat Mgamma_2 = join_vert(Mu, Mu);
  arma::mat Mgamma = join_horiz(Mgamma_1, Mgamma_2);
  arma::mat mu_tilde = Mgamma * mu_star;

  arma::mat V_wu = Rphi_u - (Mu * trans(Rphi_us));
  arma::mat Ve_1 = join_horiz(V_wu, V_wu);
  arma::mat Ve_2 = join_horiz(V_wu, V_wu + (((1/alpha)-1)*eye<arma::mat>(m, m)));
  arma::mat Ve = join_vert(Ve_1, Ve_2);
  arma::mat M_tilde = (Mgamma * V_star * trans(Mgamma)) + Ve;

  // sample from posterior predictive distribution
  arma::cube smp_cube;

  if (R > 1) {
    // For R greater than 1, we directly obtain the cube
    smp_cube = as<arma::cube>(rMT_R(Named("n", R), Named("Lambda", mu_tilde), Named("SigmaR", M_tilde), Named("SigmaC", Psi_star), Named("nu", nu_star)));

  } else {
    // For R equal to 1, we obtain the matrix and reshape it into a cube
    arma::mat smp_mat = as<arma::mat>(rMT_R(Named("n", R), Named("Lambda", mu_tilde), Named("SigmaR", M_tilde), Named("SigmaC", Psi_star), Named("nu", nu_star)));
    smp_cube = arma::cube(smp_mat.memptr(), smp_mat.n_rows, smp_mat.n_cols, 1);

  }

  // Extract wu and yu as needed
  arma::cube wu = smp_cube.rows(0, m-1);
  arma::cube yu = smp_cube.rows(m, (2*m)-1);

  return List::create(Named("Wu") = wu,
                      Named("Yu") = yu);

}


//' Evaluate the density of a set of unobserved response with respect to the conditional posterior predictive
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param X_u [matrix] unobserved instances covariate matrix
//' @param Y_u [matrix] unobserved instances response matrix
//' @param d_u [matrix] unobserved instances distance matrix
//' @param d_us [matrix] cross-distance between unobserved and observed instances matrix
//' @param hyperpar [list] two elemets: first named \eqn{\alpha}, second named \eqn{\phi}
//' @param poster [list] output from \code{fit_cpp} function
//' @param beta [matrix] posterior sample for \eqn{\beta}
//' @param sigma [matrix] posterior sample for \eqn{\Sigma}
//'
//' @return [double] posterior predictive density evaluation
//'
// [[Rcpp::export]]
double d_pred_cpp_MvT(const List& data, const arma::mat& X_u, const arma::mat& Y_u, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster) {

  // Unpack data, posterior sample and hyperparameters
  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  double alpha = as<double>(hyperpar["alpha"]);
  double phi = as<double>(hyperpar["phi"]);
  arma::mat iRphi_s = as<arma::mat>(poster["iRphi_s"]);
  arma::mat V_star = as<arma::mat>(poster["V_star"]);
  arma::mat mu_star = as<arma::mat>(poster["mu_star"]);
  arma::mat Psi_star = as<arma::mat>(poster["Psi_star"]);
  double nu_star = as<double>(poster["nu_star"]);

  // extract info from data
  int m = X_u.n_rows;
  int n = d_us.n_rows-m;

  // covariance matrices
  arma::mat Rphi_u = exp(-phi * d_u);
  arma::mat Rphi_us = exp(-phi * d_us.submat(0, m, m-1, m+n-1));

  // sampling environment
  Rcpp::Environment mniw = Rcpp::Environment::namespace_env("mniw");
  Rcpp::Function dMT_R = mniw["dMT"];

  // compute posterior predictive parameter
  arma::mat Mu = Rphi_us * iRphi_s;
  arma::mat Mgamma = join_horiz(X_u, Mu);
  arma::mat mu_tilde = Mgamma * mu_star;

  arma::mat V_wu = Rphi_u - (Mu * trans(Rphi_us));
  arma::mat Ve = V_wu + (((1/alpha)-1)*eye<arma::mat>(m, m));
  arma::mat M_tilde = (Mgamma * V_star * trans(Mgamma)) + Ve;

  // posterior predictive density
  double P_u = as<double>(dMT_R(Named("X", Y_u), Named("Lambda", mu_tilde), Named("SigmaR", M_tilde), Named("SigmaC", Psi_star), Named("nu", nu_star), Named("log", false)));

  return P_u;
}

//' Compute the LOOCV of the density evaluations for fixed values of the hyperparameters
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param priors [list] priors: named \eqn{\mu_B},\eqn{V_r},\eqn{\Psi},\eqn{\nu}
//' @param coords [matrix] sample coordinates for X and Y
//' @param hyperpar [list] two elemets: first named \eqn{\alpha}, second named \eqn{\phi}
//'
//' @return [vector] posterior predictive density evaluations
//'
// [[Rcpp::export]]
arma::vec dens_loocv_MvT(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar) {

  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  int n = Y.n_rows;
  arma::vec predictions(n);

  for (int i = 0; i < n; i++) {

    // extract the training data (excluding the i-th row) (create a mask to exclude the i-th row)
    arma::mat Ytr, Xtr, coords_i;
    if (i == 0) {
      // Exclude the 0-th row
      Ytr = Y.rows(1, n - 1);
      Xtr = X.rows(1, n - 1);
      coords_i = coords.rows(1, n - 1);
    } else if (i == 1) {
      // Exclude the 1-th row
      Ytr = join_vert(Y.row(0), Y.rows(2, n - 1));
      Xtr = join_vert(X.row(0), X.rows(2, n - 1));
      coords_i = join_vert(coords.row(0), coords.rows(2, n - 1));
    } else if (i == n - 1) {
      // Exclude the last row
      Ytr = Y.rows(0, n - 2);
      Xtr = X.rows(0, n - 2);
      coords_i = coords.rows(0, n - 2);
    } else {
      // Exclude the i-th row for i > 1
      Ytr = join_vert(Y.rows(0, i - 1), Y.rows(i + 1, n - 1));
      Xtr = join_vert(X.rows(0, i - 1), X.rows(i + 1, n - 1));
      coords_i = join_vert(coords.rows(0, i - 1), coords.rows(i + 1, n - 1));
    }
    List data_i = List::create(
      Named("Y") = Ytr,
      Named("X") = Xtr);

    // extract the test data
    arma::mat crd_i = coords.row(i);
    arma::mat X_i = X.row(i);
    arma::mat Y_i = Y.row(i);

    // Fit your model on the training data
    List poster_i = fit_cpp_MvT(data_i, priors, coords_i, hyperpar);

    // evaluate predictive density
    arma::mat d_i = arma_dist(crd_i);
    arma::mat crd_is = join_cols(crd_i, coords_i);
    arma::mat d_is = arma_dist(crd_is);

    double dens = d_pred_cpp_MvT(data_i, X_i, Y_i, d_i, d_is, hyperpar, poster_i);

    // Store the prediction
    predictions[i] = dens;
  }

  return predictions;

}


//' Compute the KCV of the density evaluations for fixed values of the hyperparameters
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param priors [list] priors: named \eqn{\mu_B},\eqn{V_r},\eqn{\Psi},\eqn{\nu}
//' @param coords [matrix] sample coordinates for X and Y
//' @param hyperpar [list] two elemets: first named \eqn{\alpha}, second named \eqn{\phi}
//' @param K [integer] number of folds
//'
//' @return [vector] posterior predictive density evaluations
//'
// [[Rcpp::export]]
arma::vec dens_kcv_MvT(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const int& K) {

  // unpack data
  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  int n = Y.n_rows;
  arma::vec predictions(n, arma::fill::zeros);

  // Create a random permutation of indices from 1 to K
  arma::vec p = arma::vec(K, arma::fill::ones) / K;
  arma::uvec foldIndices = sample_index(K, n, p);

  for (int k = 0; k < K; k++) {

    // Define the indices for the current fold
    arma::uvec testSet = find(foldIndices == (k + 1)); // Find indices that match the current fold number

    // Create a boolean vector to identify the training set for this fold
    arma::uvec trainSet = find(foldIndices != (k + 1)); // Find indices that do not match the current fold number

    // Extract the training data for the current fold
    arma::mat Ytr = Y.rows(trainSet);
    arma::mat Xtr = X.rows(trainSet);
    arma::mat coords_tr = coords.rows(trainSet);

    List data_tr = List::create(
      Named("Y") = Ytr,
      Named("X") = Xtr);

    // Extract the test data for the current fold
    arma::mat Y_test = Y.rows(testSet);
    arma::mat X_test = X.rows(testSet);
    arma::mat coords_test = coords.rows(testSet);

    // Fit your model on the training data
    List poster_k = fit_cpp_MvT(data_tr, priors, coords_tr, hyperpar);

    // evaluate predictive density for the test set of the current fold
    for (uword i = 0; i < testSet.n_elem; i++) {

      arma::mat crd_i = coords_test.row(i);
      arma::mat X_i = X_test.row(i);
      arma::mat Y_i = Y_test.row(i);
      arma::mat d_i = arma_dist(crd_i);
      arma::mat crd_is = join_cols(crd_i, coords_tr);
      arma::mat d_is = arma_dist(crd_is);

      double dens = d_pred_cpp_MvT(data_tr, X_i, Y_i, d_i, d_is, hyperpar, poster_k);
      predictions(testSet(i)) = dens;
    }
  }

  return predictions;

}


//' Return the CV predictive density evaluations for all the model combinations
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param priors [list] priors: named \eqn{\mu_B},\eqn{V_r},\eqn{\Psi},\eqn{\nu}
//' @param coords [matrix] sample coordinates for X and Y
//' @param hyperpar [list] two elemets: first named \eqn{\alpha}, second named \eqn{\phi}
//' @param useKCV [boolean] if \code{TRUE} K-fold cross validation is used instead of LOOCV (no \code{default})
//' @param K [integer] number of folds
//'
//' @return [matrix] posterior predictive density evaluations (each columns represent a different model)
//'
// [[Rcpp::export]]
arma::mat models_dens_MvT(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, bool useKCV, int K) {

  // build the grid of hyperparameters
  arma::vec Alfa = hyperpar["alpha"];
  arma::vec Fi = hyperpar["phi"];
  arma::mat Grid = expand_grid_cpp(Alfa, Fi);
  int k = Grid.n_rows;

  arma::mat out;

  for(int j = 0; j < k; j++) {

    // identify the model
    arma::rowvec hpar = Grid.row(j);
    double alfa = hpar[0];
    double fi = hpar[1];
    List hmod = List::create(
      Named("alpha") = alfa,
      Named("phi") = fi);

    // Call the appropriate function based on the "useKCV" argument
    arma::vec out_j;
    if (useKCV) {
      out_j = dens_kcv_MvT(data, priors, coords, hmod, K);
    } else {
      out_j = dens_loocv_MvT(data, priors, coords, hmod);
    }

    out =  join_horiz(out, out_j);

  }

  return out;
}


//' Compute the BPS weights by convex optimization
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param priors [list] priors: named \eqn{\mu_b},\eqn{V_b},\eqn{a},\eqn{b}
//' @param coords [matrix] sample coordinates for X and Y
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//' @param K [integer] number of folds
//'
//' @return [matrix] posterior predictive density evaluations (each columns represent a different model)
//' @export
// [[Rcpp::export]]
List BPS_weights_MvT(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, int K) {

  // compute predictive density evaluations
  arma::mat out = models_dens_MvT(data, priors, coords, hyperpar, true, K);

  // compute the weights
  arma::mat weights = as<arma::mat>(CVXR_opt(out));
  weights.elem(find(weights <= 0)).zeros();
  weights /= sum(weights, 0).eval()(0, 0);

  // return the list (BPS weights, Grid, and predictive density evaluations)
  arma::vec Alfa = hyperpar["alpha"];
  arma::vec Fi = hyperpar["phi"];
  arma::mat Grid = expand_grid_cpp(Alfa, Fi);
  arma::mat res = join_horiz(weights, Grid);

  List Res = List::create(
    Named("Grid") = res,
    Named("W") = weights,
    Named("epd") = out
  );

  return Res;
}


//' Compute the BPS spatial prediction given a set of stacking weights
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param X_u [matrix] unobserved instances covariate matrix
//' @param priors [list] priors: named \eqn{\mu_b},\eqn{V_b},\eqn{a},\eqn{b}
//' @param coords [matrix] sample coordinates for X and Y
//' @param crd_u [matrix] unboserved instances coordinates
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//' @param W [matrix] set of stacking weights
//' @param R [integer] number of desired samples
//'
//' @return [list] BPS posterior predictive samples
//' @export
// [[Rcpp::export]]
List BPS_pred_MvT(const List& data, const arma::mat& X_u, const List& priors, const arma::mat& coords, const arma::mat& crd_u, const List& hyperpar, const arma::vec& W, const int& R) {

  List pred(R);

  // compute distance matrices
  arma::mat d_u = arma_dist(crd_u);
  arma::mat crd_us = join_cols(crd_u, coords);
  arma::mat d_us = arma_dist(crd_us);

  // build the grid of hyperparameters
  arma::vec Alfa = hyperpar["alpha"];
  arma::vec Fi = hyperpar["phi"];
  arma::mat Grid = expand_grid_cpp(Alfa, Fi);
  int k = Grid.n_rows;

  // sample the model
  arma::uvec kmod = sample_index(k, R, W);

  for(int r = 0; r < R; r++) {

    // identify the k-th model
    arma::uword k_mod = kmod(r);
    arma::rowvec hpar = Grid.row(k_mod);
    double alfa = hpar[0];
    double fi = hpar[1];
    List hmod = List::create(
      Named("alpha") = alfa,
      Named("phi") = fi);

    // fit your model on the training data
    List poster = fit_cpp_MvT(data, priors, coords, hmod);

    // draw from conditional posterior predictive
    pred(r) = r_pred_cpp_MvT(data, X_u, d_u, d_us, hmod, poster, 1);

  }

  return pred;

}

//' Compute the BPS posterior samples given a set of stacking weights
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param priors [list] priors: named \eqn{\mu_b},\eqn{V_b},\eqn{a},\eqn{b}
//' @param coords [matrix] sample coordinates for X and Y
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//' @param W [matrix] set of stacking weights
//' @param R [integer] number of desired samples
//'
//' @return [matrix] BPS posterior samples
//' @export
// [[Rcpp::export]]
List BPS_postdraws_MvT(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const arma::vec& W, const int& R, bool par) {

  // Unpack necessary dimensions
  arma::mat V_r = as<arma::mat>(priors["V_r"]);
  int p = V_r.n_cols;

  // initialize return object
  List Draws(R);

  for(int r = 0; r < R; r++) {

    // build the grid of hyperparameters
    arma::vec Alfa = hyperpar["alpha"];
    arma::vec Fi = hyperpar["phi"];
    arma::mat Grid = expand_grid_cpp(Alfa, Fi);
    int k = Grid.n_rows;

    // sample the model
    arma::uvec kmod = sample_index(k, 1, W);
    arma::uword k_mod = kmod(0);

    // identify the k-th model
    arma::rowvec hpar = Grid.row(k_mod);
    double alfa = hpar[0];
    double fi = hpar[1];
    List hmod = List::create(
      Named("alpha") = alfa,
      Named("phi") = fi);

    // fit your model on the training data
    List poster = fit_cpp_MvT(data, priors, coords, hmod);

    // posterior draws
    // bool par = true;
    List post = post_draws_MvT(poster, 1, par = par, p = p);

    Draws(r) =  post;

  }

  // return pred;
  return Draws;

}


