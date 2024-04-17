

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


// [[Rcpp::export]]
arma::uvec sample_index(const int& size, const int& length, const arma::vec& p){
  arma::uvec sequence = arma::linspace<arma::uvec>(0, size-1, size);
  arma::uvec out = Rcpp::RcppArmadillo::sample(sequence, length, true, p);
  return out;
}


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
SEXP CVXR_opt(const arma::mat& scores) {

  // Evaluate R expression in the package environment
  Rcpp::Environment pkg_env = Rcpp::Environment::namespace_env("ASMK");
  Rcpp::Function conv_opt = pkg_env["conv_opt"];

  // Apply convex optimization by CVXR as an R function
  return conv_opt(Named("scores", scores));
}


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

'

Rcpp::sourceCpp(code = cd)
