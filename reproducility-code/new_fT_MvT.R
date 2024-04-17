cd <-
'
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

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


// [[Rcpp::export]]
arma::uvec sample_index(const int& size, const int& length, const arma::vec& p){
  arma::uvec sequence = arma::linspace<arma::uvec>(0, size-1, size);
  arma::uvec out = Rcpp::RcppArmadillo::sample(sequence, length, true, p);
  return out;
}


// [[Rcpp::export]]
SEXP CVXR_opt(const arma::mat& scores) {

  // Evaluate R expression in the package environment
  Rcpp::Environment pkg_env = Rcpp::Environment::namespace_env("ASMK");
  Rcpp::Function conv_opt = pkg_env["conv_opt"];

  // Apply convex optimization by CVXR as an R function
  return conv_opt(Named("scores", scores));
}


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


// [[Rcpp::export]]
List r_pred_latent_cppT(const List& data, const arma::mat& X_u, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster, const int& R) {

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
  int q = mu_star.n_cols;

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

  // // sample from posterior predictive distribution
  // arma::cube smp = as<arma::cube>(rMT_R(Named("n", R), Named("Lambda", mu_tilde), Named("SigmaR", M_tilde), Named("SigmaC", Psi_star), Named("nu", nu_star)));
  //
  // arma::cube wu = smp.rows(0, m-1);
  // arma::cube yu = smp.rows(m, (2*m)-1);


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


// [[Rcpp::export]]
double d_pred_latent_cppT(const List& data, const arma::mat& X_u, const arma::mat& Y_u, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster) {

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
  // int p = X_u.n_cols;
  // int q = mu_star.n_cols;

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


// [[Rcpp::export]]
arma::vec dens_loocv_latentT(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar) {

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

    double dens = d_pred_latent_cppT(data_i, X_i, Y_i, d_i, d_is, hyperpar, poster_i);

    // Store the prediction
    predictions[i] = dens;
  }

  return predictions;

}


// [[Rcpp::export]]
arma::vec dens_kcv_latentT(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, const int& K) {

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

    // evaluate predictive density for the test set of the current fold
    for (uword i = 0; i < testSet.n_elem; i++) {

      arma::mat crd_i = coords_test.row(i);
      arma::mat X_i = X_test.row(i);
      arma::mat Y_i = Y_test.row(i);
      arma::mat d_i = arma_dist(crd_i);
      arma::mat crd_is = join_cols(crd_i, coords_tr);
      arma::mat d_is = arma_dist(crd_is);

      double dens = d_pred_latent_cppT(data_tr, X_i, Y_i, d_i, d_is, hyperpar, poster_k);
      predictions(testSet(i)) = dens;
    }
  }

  return predictions;

}


// [[Rcpp::export]]
arma::mat models_dens_latentT(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, bool useKCV, int K) {

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
      out_j = dens_kcv_latentT(data, priors, coords, hmod, K);
    } else {
      out_j = dens_loocv_latentT(data, priors, coords, hmod);
    }

    out =  join_horiz(out, out_j);

  }

  return out;
}


// [[Rcpp::export]]
List BPS_weights_MvT(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar, int K) {

  // compute predictive density evaluations
  arma::mat out = models_dens_latentT(data, priors, coords, hyperpar, true, K);

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


// [[Rcpp::export]]
List BPS_pred_MvTT(const List& data, const arma::mat& X_u, const List& priors, const arma::mat& coords, const arma::mat& crd_u, const List& hyperpar, const arma::vec& W, const int& R) {

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

    // draw from conditional posterior predictive
    pred(r) = r_pred_latent_cppT(data, X_u, d_u, d_us, hmod, poster, 1);

  }

  return pred;

}


'

sourceCpp(code = cd)
