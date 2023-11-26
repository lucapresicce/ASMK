#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;


// UTILITY FUNCTIONS ------------------------------------------------------------------------------


//' Compute the Euclidean distance matrix
//'
//' @param X [matrix] (tipically of \eqn{N} coordindates on \eqn{\mathbb{R}^2} )
//'
//' @return [matrix] distance matrix of the elements of \eqn{X}
//' @export
// [[Rcpp::export(name = "arma_dist")]]
arma::mat arma_dist(const arma::mat & X){
  int n = X.n_rows;
  arma::mat D(n, n, fill::zeros); // Allocate a matrix of dimension n x n
  for (int i = 0; i < n; i++) {
    for(int k = 0; k < i; k++){
      D(i, k) = sqrt(sum(pow(X.row(i) - X.row(k), 2)));
      D(k, i) = D(i, k);
    }
  }
  return D;
}


//' Build a grid from two vector (i.e. equivalent to \code{expand.grid()} in \code{R})
//'
//' @param x [vector] first vector of numeric elements
//' @param y [vector] second vector of numeric elements
//'
//' @return [matrix] expanded grid of combinations
//'
// [[Rcpp::export]]
arma::mat expand_grid_cpp(const arma::vec& x, const arma::vec& y) {
  int n1 = x.size();
  int n2 = y.size();
  int n_combinations = n1 * n2;

  arma::mat result(n_combinations, 2);

  int k = 0;
  for (int j = 0; j < n2; j++) {
    for (int i = 0; i < n1; i++) {
      result(k, 0) = x[i];
      result(k, 1) = y[j];
      k++;
    }
  }

  return result;
}


//' Function to sample integers (index)
//'
//' @param size [integer] dimension of the set to sample
//' @param length [integer] number of elements to sample
//' @param p [vector] sampling probabilities
//'
//' @return [vector] sample of integers
//'
// [[Rcpp::export]]
arma::uvec sample_index(const int& size, const int& length, const arma::vec& p){
  arma::uvec sequence = arma::linspace<arma::uvec>(0, size-1, size);
  arma::uvec out = Rcpp::RcppArmadillo::sample(sequence, length, true, p);
  return out;
}


// UNIVARIATE LATENT MODELS -----------------------------------------------------------------------


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

  arma::mat V_1 = join_vert(delta*eye<arma::mat>(n, n), arma::mat(p+n, n, arma::fill::zeros));
  arma::mat V_2 = join_vert(arma::mat(n, p, arma::fill::zeros), V_b, arma::mat(n, p, arma::fill::zeros));
  arma::mat V_3 = join_vert(arma::mat(n+p, n, arma::fill::zeros), Rphi_s);
  arma::mat V_star = join_horiz(V_1, V_2, V_3);

  // Rcout << "X_star : " << X_star << std::endl;
  // Rcout << "V_star : " << X_star << std::endl;

  // Precompute some reusable values
  arma::mat tX_star = trans(X_star);
  arma::mat iV_star = arma::inv(V_star);

  // Rcout << "iV_star : " << iV_star << std::endl;

  // conjugate posterior parameters
  arma::mat iM_star = tX_star * iV_star * X_star;

  // Rcout << "iM_star : " << iM_star << std::endl;

  arma::mat M_star = arma::inv(iM_star);

  // Rcout << "M_star : " << M_star << std::endl;

  arma::mat tXVY = tX_star * iV_star * Y_star;
  vec gamma_hat = M_star * tXVY;

  // Rcout << "gamma_hat : " << gamma_hat << std::endl;
  // Rcout << "X_star * gamma_hat : " << X_star * gamma_hat << std::endl;
  // Rcout << "Y_star - (X_star * gamma_hat) : " << Y_star - (X_star * gamma_hat) << std::endl;
  // Rcout << "trans(Y_star - (X_star * gamma_hat)) : " << trans(Y_star - (X_star * gamma_hat)) << std::endl;


  arma::mat SXY = Y_star - (X_star * gamma_hat);
  arma::mat tSXY = trans(Y_star - (X_star * gamma_hat));

  // Rcout << "SXY : " << SXY << std::endl;
  // Rcout << "tSXY : " << tSXY << std::endl;
  // Rcout << "prod : " << tSXY * iV_star * SXY << std::endl;

  arma::mat bb = tSXY * iV_star * SXY;
  double b_star = b + 0.5 * as_scalar(bb);

  double a_star = a + (n/2);

  // Return results as an R list
  return List::create(Named("M_star") = M_star,
                      Named("gamma_hat") = gamma_hat,
                      Named("b_star") = b_star,
                      Named("a_star") = a_star,
                      Named("Rphi_s") = Rphi_s);
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
List post_draws(const List& poster, const int& R = 50, const bool& par = false, const int& p = 1) {

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
//' @param Rphi_s [matrix] correlation matrix of sample instances
//' @param d_u [matrix] unobserved instances distance matrix
//' @param d_us [matrix] cross-distance between unobserved and observed instances matrix
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//' @param poster [list] output from \code{fit_cpp} function
//'
//' @return [list] posterior predictive samples
//'
// [[Rcpp::export]]
List r_pred_cpp(const List& data, const arma::mat& X_u, const arma::mat& Rphi_s, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster) {

  // Unpack data, posterior sample and hyperparameters
  arma::mat beta = as<arma::mat>(poster["Betas"]);
  arma::vec sigma = as<arma::vec>(poster["Sigmas"]);
  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  double delta = as<double>(hyperpar["delta"]);
  double phi = as<double>(hyperpar["phi"]);

  // extract info from data
  int m = X_u.n_rows;
  int p = X_u.n_cols;
  int n = d_us.n_rows-m;
  int R = beta.n_rows;

  // covariance matrices
  arma::mat iR_s = arma::inv(Rphi_s);
  arma::mat Rphi_u = exp(-phi * d_u);
  arma::mat Rphi_us = exp(-phi * d_us.submat(0, m, m-1, m+n-1));

  // environment
  Rcpp::Environment mniw = Rcpp::Environment::namespace_env("mniw");
  Rcpp::Function rmNorm_R = mniw["rmNorm"];

  // initialize return objects
  arma::mat Z_u(m, R);
  arma::mat Y_u(m, R);
  for (int r = 0; r < R; ++r) {

    // unpack posterior sample
    arma::vec gamma_hat_r = trans(beta.row(r));
    arma::vec b = gamma_hat_r.subvec(0, p - 1);
    arma::vec gamma_r = gamma_hat_r.subvec(p, gamma_hat_r.n_elem - 1);
    double s = sigma(r);


    // predictive conjugate parameters
    arma::mat mu_z = Rphi_us * iR_s * gamma_r;
    arma::mat V_z = Rphi_u - Rphi_us * iR_s * trans(Rphi_us);
    Z_u.col(r) = as<arma::vec>(rmNorm_R(Named("n", 1), Named("mu", trans(mu_z)), Named("Sigma", s * V_z)));

    arma::mat mu_y = X_u * b + Z_u.col(r);
    arma::mat V_y = (s * delta) * eye<arma::mat>(m, m);
    Y_u.col(r) = as<arma::vec>(rmNorm_R(Named("n", 1), Named("mu", trans(mu_y)), Named("Sigma", V_y)));

  }

  return List::create(Named("Z_u") = Z_u,
                      Named("Y_u") = Y_u);
}


//' Evaluate the density of a set of unobserved response with respect to the conditional posterior predictive
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param X_u [matrix] unobserved instances covariate matrix
//' @param Y_u [matrix] unobserved instances response matrix
//' @param Rphi_s [matrix] correlation matrix of sample instances
//' @param d_u [matrix] unobserved instances distance matrix
//' @param d_us [matrix] cross-distance between unobserved and observed instances matrix
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//' @param poster [list] output from \code{fit_cpp} function
//'
//' @return [vector] posterior predictive density evaluations
//'
// [[Rcpp::export]]
arma::vec d_pred_cpp(const List& data, const arma::mat& X_u, const arma::vec& Y_u, const arma::mat& Rphi_s, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster) {

  // Unpack data, posterior sample and hyperparameters
  arma::mat beta = as<arma::mat>(poster["Betas"]);
  arma::vec sigma = as<arma::vec>(poster["Sigmas"]);
  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  double delta = as<double>(hyperpar["delta"]);
  double phi = as<double>(hyperpar["phi"]);

  // extract info from data
  int m = X_u.n_rows;
  int p = X_u.n_cols;
  int n = d_us.n_rows-m;
  int R = beta.n_rows;

  // covariance matrices
  arma::mat iR_s = arma::inv(Rphi_s);
  arma::mat Rphi_u = exp(-phi * d_u);
  arma::mat Rphi_us = exp(-phi * d_us.submat(0, m, m-1, m+n-1));

  // environment
  Rcpp::Environment mniw = Rcpp::Environment::namespace_env("mniw");
  Rcpp::Function rmNorm_R = mniw["rmNorm"];
  Rcpp::Function dmNorm_R = mniw["dmNorm"];


  // initialize return objects
  arma::vec P_u(R);
  for (int r = 0; r < R; ++r) {

    // unpack posterior sample
    arma::vec gamma_hat_r = trans(beta.row(r));
    arma::vec b = gamma_hat_r.subvec(0, p - 1);
    arma::vec gamma_r = gamma_hat_r.subvec(p, gamma_hat_r.n_elem - 1);
    double s = sigma(r);

    // predictive conjugate parameters
    arma::mat mu_z = Rphi_us * iR_s * gamma_r;
    arma::mat V_z = Rphi_u - Rphi_us * iR_s * trans(Rphi_us);
    arma::vec Z_u = as<arma::vec>(rmNorm_R(Named("n", 1), Named("mu", trans(mu_z)), Named("Sigma", s * V_z)));

    arma::mat mu_y = X_u * b + Z_u;
    arma::mat V_y = (s * delta) * eye<arma::mat>(m, m);
    P_u(r) = as<double>(dmNorm_R(Named("x", trans(Y_u)), Named("mu", trans(mu_y)), Named("Sigma", V_y)));

  }

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

    // posterior draws
    List post_i = post_draws(poster_i, 1);

    // evaluate predictive density
    arma::mat Rphi_s = as<arma::mat>(poster_i["Rphi_s"]);
    arma::mat d_i = arma_dist(crd_i);
    arma::mat crd_is = join_cols(crd_i, coords_i);
    arma::mat d_is = arma_dist(crd_is);

    double dens = as_scalar(d_pred_cpp(data_i, X_i, Y_i, Rphi_s, d_i, d_is, hyperpar, post_i));

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
arma::vec dens_kcv(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const int& K) { //, const int& seed) {

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
    arma::mat Rphi_k = as<arma::mat>(poster_k["Rphi_s"]);

    // posterior draws
    List post_k = post_draws(poster_k, 1);

    // evaluate predictive density for the test set of the current fold
    for (uword i = 0; i < testSet.n_elem; i++) {

      arma::mat crd_i = coords_test.row(i);
      arma::mat X_i = X_test.row(i);
      arma::mat Y_i = Y_test.row(i);
      arma::mat d_i = arma_dist(crd_i);

      // Rcout << crd_i << std::endl;
      // Rcout << coords_tr << std::endl;

      arma::mat crd_is = join_cols(crd_i, coords_tr);
      arma::mat d_is = arma_dist(crd_is);

      double dens = as_scalar(d_pred_cpp(data_tr, X_i, Y_i, Rphi_k, d_i, d_is, hyperpar, post_k));
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
arma::mat models_dens(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const bool& useKCV, const int& K = 10) {

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

    // Call the appropriate function based on the 'useKCV' argument
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


//' Compute the KCV of the density evaluations for fixed values of the hyperparameters
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param priors [list] priors: named \eqn{\mu_b},\eqn{V_b},\eqn{a},\eqn{b}
//' @param coords [matrix] sample coordinates for X and Y
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//' @param K [integer] number of folds
//'
//' @return [list] posterior predictive density evaluations
//'
// [[Rcpp::export]]
List dens_kcv2(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const int& K) { //, const int& seed) {

  // unpack data
  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  int n = Y.n_rows;
  int p = X.n_cols;
  arma::vec predictions(n, arma::fill::zeros);
  arma::mat all_beta(K, p, arma::fill::zeros);
  arma::vec all_sigma(K, arma::fill::zeros);

  // Create a random permutation of indices from 1 to K
  arma::vec pr = arma::vec(K, arma::fill::ones) / K;
  arma::uvec foldIndices = sample_index(K, n, pr);

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
    arma::mat Rphi_k = as<arma::mat>(poster_k["Rphi_s"]);

    // posterior draws
    List post_k = post_draws(poster_k, 1);

    // Unpack posterior sample
    arma::rowvec beta = as<arma::rowvec>(post_k["Betas"]);
    arma::vec sigma = as<arma::vec>(post_k["Sigmas"]);

    // Store the results for the current fold
    all_beta.row(k) = beta.subvec(0, p - 1);
    all_sigma(k) = sigma(0);

    // evaluate predictive density for the test set of the current fold
    for (uword i = 0; i < testSet.n_elem; i++) {

      arma::mat crd_i = coords_test.row(i);
      arma::mat X_i = X_test.row(i);
      arma::mat Y_i = Y_test.row(i);
      arma::mat d_i = arma_dist(crd_i);

      // Rcout << crd_i << std::endl;
      // Rcout << coords_tr << std::endl;

      arma::mat crd_is = join_cols(crd_i, coords_tr);
      arma::mat d_is = arma_dist(crd_is);

      double dens = as_scalar(d_pred_cpp(data_tr, X_i, Y_i, Rphi_k, d_i, d_is, hyperpar, post_k));
      predictions(testSet(i)) = dens;
    }
  }

  // Create a list to hold both predictions and post_k outputs
  List results;
  results["predictions"] = predictions;
  results["beta"] = all_beta;
  results["sigma"] = all_sigma;

  return results;
}


//' Return the CV predictive density evaluations for all the model combinations
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param priors [list] priors: named \eqn{\mu_b},\eqn{V_b},\eqn{a},\eqn{b}
//' @param coords [matrix] sample coordinates for X and Y
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//' @param K [integer] number of folds
//'
//' @return [matrix] posterior predictive density evaluations (each columns represent a different model)
//'
// [[Rcpp::export]]
List models_dens2(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const int& K = 10) {

  // build the grid of hyperparameters
  arma::vec Delta = hyperpar["delta"];
  arma::vec Fi = hyperpar["phi"];
  arma::mat Grid = expand_grid_cpp(Delta, Fi);
  int k = Grid.n_rows;

  arma::mat out;
  List beta(k);
  List sigma(k);

  for(int j = 0; j < k; j++) {

    // identify the model
    arma::rowvec hpar = Grid.row(j);
    double delt = hpar[0];
    double fi = hpar[1];
    List hmod = List::create(
      Named("delta") = delt,
      Named("phi") = fi);

    // Call the KCV function based on K folds
    List res = dens_kcv2(data, priors, coords, hmod, K);
    arma::vec out_j = res["predictions"];
    arma::mat beta_j = res["beta"];
    arma::vec sigma_j = res["sigma"];

    out =  join_horiz(out, out_j);
    beta(j) =  beta_j;
    sigma(j) =  sigma_j;

  }

  List results;
  results["out"] = out;
  results["beta"] = beta;
  results["sigma"] = sigma;

  return results;
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
List BPS_SpatialPrediction_cpp(const List& data, const arma::mat& X_u, const List& priors, const arma::mat& coords, const arma::mat& crd_u, const List& hyperpar, const arma::vec& W, const int& R) {

  arma::mat Z_pred;
  arma::mat Y_pred;

  // compute distance matrices
  arma::mat d_u = arma_dist(crd_u);
  arma::mat crd_us = join_cols(crd_u, coords);
  arma::mat d_us = arma_dist(crd_us);

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
    List post = post_draws(poster, 1);

    // draw from conditional posterior predictive
    arma::mat Rphi_s = as<arma::mat>(poster["Rphi_s"]);
    List pred_R = r_pred_cpp(data, X_u, Rphi_s, d_u, d_us, hmod, post);

    arma::vec Z_pred_r = as<arma::mat>(pred_R["Z_u"]);
    Z_pred =  join_horiz(Z_pred, Z_pred_r);

    arma::vec Y_pred_r = as<arma::mat>(pred_R["Y_u"]);
    Y_pred =  join_horiz(Y_pred, Y_pred_r);

  }

  // return pred;
  return List::create(Named("Z_hat") = Z_pred,
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
arma::mat BPS_post_draws(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const arma::vec& W, const int& R) {

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


// MULTIVARIATE LATENT MODELS ---------------------------------------------------------------------


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
List fit_latent_cpp(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar) {

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


//' The same fit_latent_cpp, but take as argument the distance matrix directly (does not compute it by itself from coords)
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param priors [list] priors: named \eqn{\mu_B},\eqn{V_r},\eqn{\Psi},\eqn{\nu}
//' @param coords [matrix] sample coordinates for X and Y
//' @param hyperpar [list] two elemets: first named \eqn{\alpha}, second named \eqn{\phi}
//'
//' @return [list] posterior update parameters
//'
// [[Rcpp::export]]
List fit_latent_cpp2(const List& data, const List& priors, const arma::mat& dist, const List& hyperpar) {

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
  arma::mat Rphi_s = exp(-phi * dist);

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
List post_draws_latent(const List& poster, const int& R = 50, const bool& par = false, const int& p = 2) {

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


//' Compute the BPS posterior samples given a set of stacking weights
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param priors [list] priors: named \eqn{\mu_B},\eqn{V_r},\eqn{\Psi},\eqn{\nu}
//' @param coords [matrix] sample coordinates for X and Y
//' @param hyperpar [list] two elemets: first named \eqn{\alpha}, second named \eqn{\phi}
//' @param W [matrix] set of stacking weights
//' @param R [integer] number of desired samples
//'
//' @return [list] BPS posterior samples
//' @export
// [[Rcpp::export]]
List BPS_post_draws_latent(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const arma::vec& W, const int& R) {

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
    List poster = fit_latent_cpp(data, priors, coords, hmod);

    // posterior draws
    bool par = true;
    List post = post_draws_latent(poster, 1, par = par, p = p);

    Draws(r) =  post;

  }

  // return pred;
  return Draws;

}


//' Draw from the conditional posterior predictive for a set of unobserved covariates
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param X_u [matrix] unobserved instances covariate matrix
//' @param d_u [matrix] unobserved instances distance matrix
//' @param d_us [matrix] cross-distance between unobserved and observed instances matrix
//' @param hyperpar [list] two elemets: first named \eqn{\alpha}, second named \eqn{\phi}
//' @param poster [list] output from \code{fit_cpp} function
//' @param beta [matrix] posterior sample for \eqn{\beta}
//' @param sigma [matrix] posterior sample for \eqn{\Sigma}
//'
//' @return [list] posterior predictive samples
//'
// [[Rcpp::export]]
List r_pred_latent_cpp(const List& data, const arma::mat& X_u, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster, const arma::mat& beta, const arma::mat& sigma) {

  // Unpack data, posterior sample and hyperparameters
  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  arma::mat iR_s = as<arma::mat>(poster["iRphi_s"]);
  double alpha = as<double>(hyperpar["alpha"]);
  double phi = as<double>(hyperpar["phi"]);

  // extract info from data
  int m = X_u.n_rows;
  int n = d_us.n_rows-m;
  int p = X_u.n_cols;

  // covariance matrices
  arma::mat Rphi_u = exp(-phi * d_u);
  arma::mat Rphi_us = exp(-phi * d_us.submat(0, m, m-1, m+n-1));

  // sampling environment
  Rcpp::Environment mniw = Rcpp::Environment::namespace_env("mniw");
  Rcpp::Function rMNorm_R = mniw["rMNorm"];

  // predictive conjugate parameters
  arma::mat b = beta.rows(0, p - 1);
  arma::mat w = beta.rows(p, (n+p) - 1);

  // prediction W_u
  arma::mat M_u = Rphi_us * iR_s;
  arma::mat mu_u = M_u * w;
  arma::mat V_u = Rphi_u - Rphi_us * iR_s * trans(Rphi_us);
  arma::mat resultW = as<arma::mat>(rMNorm_R(Named("n", 1), Named("Lambda", mu_u), Named("SigmaR", V_u), Named("SigmaC", sigma)));

  // prediction Y_u
  arma::mat mu_y = X_u * b + resultW;
  double a = alpha / (1 - alpha);
  arma::mat V_y = a * eye<arma::mat>(m, m);
  arma::mat resultY = as<arma::mat>(rMNorm_R(Named("n", 1), Named("Lambda", mu_y), Named("SigmaR", V_y), Named("SigmaC", sigma)));

  return List::create(Named("Wu") = resultW,
                      Named("Yu") = resultY);

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
double d_pred_latent_cpp(const List& data, const arma::mat& X_u, const arma::mat& Y_u, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster, const arma::mat& beta, const arma::mat& sigma) {

  // Unpack data, posterior sample and hyperparameters
  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  arma::mat iR_s = as<arma::mat>(poster["iRphi_s"]);
  double alpha = as<double>(hyperpar["alpha"]);
  double phi = as<double>(hyperpar["phi"]);

  // extract info from data
  int m = X_u.n_rows;
  int n = d_us.n_rows-m;
  int p = X_u.n_cols;

  // covariance matrices
  arma::mat Rphi_u = exp(-phi * d_u);
  arma::mat Rphi_us = exp(-phi * d_us.submat(0, m, m-1, m+n-1));

  // sampling environment
  Rcpp::Environment mniw = Rcpp::Environment::namespace_env("mniw");
  Rcpp::Function rMNorm_R = mniw["rMNorm"];
  Rcpp::Function dMNorm_R = mniw["dMNorm"];

  // predictive conjugate parameters
  arma::mat b = beta.rows(0, p - 1);
  arma::mat w = beta.rows(p, (n+p) - 1);

  // prediction W_u
  arma::mat M_u = Rphi_us * iR_s;
  arma::mat mu_u = M_u * w;
  arma::mat V_u = Rphi_u - Rphi_us * iR_s * trans(Rphi_us);
  arma::mat resultW = as<arma::mat>(rMNorm_R(Named("n", 1), Named("Lambda", mu_u), Named("SigmaR", V_u), Named("SigmaC", sigma)));

  // prediction Y_u
  arma::mat mu_y = X_u * b + resultW;
  double a = alpha / (1 - alpha);
  arma::mat V_y = a * eye<arma::mat>(m, m);
  double resultY = as<double>(dMNorm_R(Named("X", Y_u), Named("Lambda", mu_y), Named("SigmaR", V_y), Named("SigmaC", sigma)));

  return resultY;
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
arma::vec dens_loocv_latent(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar) {

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
    List poster_i = fit_latent_cpp(data_i, priors, coords_i, hyperpar);

    // evaluate predictive density
    arma::mat d_i = arma_dist(crd_i);
    arma::mat crd_is = join_cols(crd_i, coords_i);
    arma::mat d_is = arma_dist(crd_is);

    // posterior draws
    List post = post_draws_latent(poster_i, 1);
    List drw = as<List>(post(0));
    arma::mat b = as<arma::mat>(drw["beta"]);
    arma::mat s = as<arma::mat>(drw["sigma"]);

    double dens = d_pred_latent_cpp(data_i, X_i, Y_i, d_i, d_is, hyperpar, poster_i, b, s);

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
arma::vec dens_kcv_latent(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const int& K) { //, const int& seed) {

  // unpack data
  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  int n = Y.n_rows;
  arma::vec predictions(n, arma::fill::zeros);
  // Rcout << "n " << n << std::endl;

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
    List poster_k = fit_latent_cpp(data_tr, priors, coords_tr, hyperpar);

    // posterior draws
    List post = post_draws_latent(poster_k, 1);
    List drw = as<List>(post(0));
    arma::mat b = as<arma::mat>(drw["beta"]);
    arma::mat s = as<arma::mat>(drw["sigma"]);

    // evaluate predictive density for the test set of the current fold
    for (uword i = 0; i < testSet.n_elem; i++) {

      arma::mat crd_i = coords_test.row(i);
      arma::mat X_i = X_test.row(i);
      arma::mat Y_i = Y_test.row(i);
      arma::mat d_i = arma_dist(crd_i);
      arma::mat crd_is = join_cols(crd_i, coords_tr);
      arma::mat d_is = arma_dist(crd_is);

      double dens = d_pred_latent_cpp(data_tr, X_i, Y_i, d_i, d_is, hyperpar, poster_k, b, s);
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
arma::mat models_dens_latent(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, bool useKCV, int K = 10) {

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

    // Call the appropriate function based on the 'useKCV' argument
    arma::vec out_j;
    if (useKCV) {
      out_j = dens_kcv_latent(data, priors, coords, hmod, K);
    } else {
      out_j = dens_loocv_latent(data, priors, coords, hmod);
    }

    out =  join_horiz(out, out_j);

  }

  return out;
}


//' Compute the BPS spatial prediction given a set of stacking weights
//'
//' @param data [list] two elements: first named \eqn{Y}, second named \eqn{X}
//' @param X_u [matrix] unobserved instances covariate matrix
//' @param priors [list] priors: named \eqn{\mu_B},\eqn{V_r},\eqn{\Psi},\eqn{\nu}
//' @param coords [matrix] sample coordinates for X and Y
//' @param crd_u [matrix] unboserved instances coordinates
//' @param hyperpar [list] two elemets: first named \eqn{\alpha}, second named \eqn{\phi}
//' @param W [matrix] set of stacking weights
//' @param R [integer] number of desired samples
//'
//' @return [list] BPS posterior predictive samples
//' @export
// [[Rcpp::export]]
List BPS_latent_SpatialPrediction_cpp(const List& data, const arma::mat& X_u, const List& priors, const arma::mat& coords, const arma::mat& crd_u, const List& hyperpar, const arma::vec& W, const int& R) {

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
    List poster = fit_latent_cpp(data, priors, coords, hmod);

    // posterior draws
    List post = post_draws_latent(poster, 1);
    List drw = post(0);
    arma::mat b = as<arma::mat>(drw["beta"]);
    arma::mat s = as<arma::mat>(drw["sigma"]);

    // draw from conditional posterior predictive
    pred(r) = r_pred_latent_cpp(data, X_u, d_u, d_us, hmod, poster, b, s);

  }

  return pred;

}

