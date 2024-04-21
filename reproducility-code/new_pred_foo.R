
code <-
  '
#include <RcppArmadillo.h>

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


// [[Rcpp::export]]
arma::mat forceSymmetry_cpp(const arma::mat& mat) {
  // Extract the lower triangular part of the matrix
  arma::mat lower = arma::trimatl(mat);

  // Create a symmetric matrix by copying the lower triangular part to the upper triangular part
  arma::mat symmat = arma::symmatl(lower);

  return symmat;
}

// [[Rcpp::export]]
List r_pred_MC(const List& data, const arma::mat& X_u, const arma::mat& iRphi_s, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster) {

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
  // arma::mat iR_s = arma::inv(Rphi_s);
  arma::mat Rphi_u = exp(-phi * d_u);
  arma::mat Rphi_us = exp(-phi * d_us.submat(0, m, m-1, m+n-1));

  // environment
  Rcpp::Environment mniw = Rcpp::Environment::namespace_env("mniw");
  Rcpp::Function rmNorm_R = mniw["rmNorm"];
  //
  // initialize return objects
  arma::mat Z_u(m, R);
  arma::mat Y_u(m, R);

  // compute reusable
  arma::mat RiR = Rphi_us * iRphi_s;
  arma::mat V_z = Rphi_u - RiR * trans(Rphi_us);

  for (int r = 0; r < R; ++r) {

    // unpack posterior sample
    arma::vec gamma_hat_r = trans(beta.row(r));
    arma::vec b = gamma_hat_r.subvec(0, p - 1);
    arma::vec gamma_r = gamma_hat_r.subvec(p, gamma_hat_r.n_elem - 1);
    double s = sigma(r);


    // predictive conjugate parameters
    // arma::mat mu_z = Rphi_us * iR_s * gamma_r;
    // arma::mat V_z = Rphi_u - Rphi_us * iR_s * trans(Rphi_us);
    arma::mat mu_z = RiR * gamma_r;
    Z_u.col(r) = as<arma::vec>(rmNorm_R(Named("n", 1), Named("mu", trans(mu_z)), Named("Sigma", s * V_z)));

    arma::mat mu_y = X_u * b + Z_u.col(r);
    arma::mat V_y = (s * delta) * eye<arma::mat>(m, m);
    Y_u.col(r) = as<arma::vec>(rmNorm_R(Named("n", 1), Named("mu", trans(mu_y)), Named("Sigma", V_y)));

  }

  return List::create(Named("Z_u") = Z_u,
                      Named("Y_u") = Y_u);
}


// [[Rcpp::export]]
Rcpp::List r_pred_cpp1(const List& data, const arma::mat& X_u, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster, const int& R) {

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
  Rcpp::Environment mvnfast = Rcpp::Environment::namespace_env("mvnfast");
  Rcpp::Function rmvt_R = mvnfast["rmvt"];

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
  double t_df = a_star;
  double scale_ratio = b_star/a_star;
  arma::mat scaled_M_tilde = forceSymmetry_cpp(scale_ratio * M_tilde);

  // posterior predictive sample
  arma::mat res;
  res = as<arma::mat>(rmvt_R(Named("n", R), Named("sigma", scaled_M_tilde), Named("mu", mu_tilde), Named("df", t_df)));

  return List::create(Named("W_u") = res.cols(0, m-1),
                      Named("Y_u") = res.cols(m, (2*m)-1));

}


// [[Rcpp::export]]
Rcpp::List r_pred_cppHALF(const List& data, const arma::mat& X_u, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster, const int& R) {

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
  Rcpp::Environment mvnfast = Rcpp::Environment::namespace_env("mvnfast");
  Rcpp::Function rmvt_R = mvnfast["rmvt"];

  // (exponential) covariance matrices
  arma::mat Rphi_u = exp(-phi * d_u);
  arma::mat Rphi_us = exp(-phi * d_us.submat(0, m, m-1, m+n-1));

  // compute posterior predictive parameters
  arma::mat JR = Rphi_us * iRphi_s;

  // latent process
  arma::mat Zer_mp(m, p, arma::fill::zeros);
  arma::mat W_w = join_horiz(Zer_mp, JR);
  arma::vec mu_tilde_w = W_w * gamma_hat;
  arma::mat V_w = Rphi_u - JR * trans(Rphi_us);
  arma::mat M_tilde_w = (W_w * M_star * trans(W_w)) + V_w;

  // response
  arma::mat W_y = join_horiz(X_u, JR);
  arma::vec mu_tilde_y = W_y * gamma_hat;
  arma::mat V_ey = V_w + (delta*eye<arma::mat>(m, m));
  arma::mat M_tilde_y = (W_y * M_star * trans(W_y)) + V_ey;

  // degrees of freedom
  double t_df = a_star;
  double scale_ratio = b_star/a_star;
  // arma::mat scaled_M_tilde_w = forceSymmetry_cpp(scale_ratio * M_tilde_w);
  // arma::mat scaled_M_tilde_y = forceSymmetry_cpp(scale_ratio * M_tilde_y);

  // posterior predictive sample
  arma::mat res_w;
  arma::mat res_y;
  res_w = as<arma::mat>(rmvt_R(Named("n", R), Named("sigma", scale_ratio * M_tilde_w), Named("mu", mu_tilde_w), Named("df", t_df)));
  res_y = as<arma::mat>(rmvt_R(Named("n", R), Named("sigma", scale_ratio * M_tilde_y), Named("mu", mu_tilde_y), Named("df", t_df)));

  return List::create(Named("W_u") = res_w,
                      Named("Y_u") = res_y);

}


'

Rcpp::sourceCpp(code = code)


# -------------------------------------------------------------------------

code_mvt <-
  '
#include <RcppArmadillo.h>

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


// [[Rcpp::export]]
List r_pred_cpp_MvT1(const List& data, const arma::mat& X_u, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster, const int& R) {

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


// [[Rcpp::export]]
List r_pred_latent_MC(const List& data, const arma::mat& X_u, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster, const arma::mat& beta, const arma::mat& sigma) {

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


// [[Rcpp::export]]
List r_pred_cpp_MvTHALF(const List& data, const arma::mat& X_u, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster, const int& R) {

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

  // compute posterior predictive parameters
  arma::mat Mu = Rphi_us * iRphi_s;

  // latent process
  arma::mat Zer_mp(m, p, arma::fill::zeros);
  arma::mat M_gamma_w = join_horiz(Zer_mp, Mu);
  arma::mat mu_tilde_w = M_gamma_w * mu_star;
  arma::mat V_wu = Rphi_u - (Mu * trans(Rphi_us));
  arma::mat M_tilde_w = (M_gamma_w * V_star * trans(M_gamma_w)) + V_wu;

  // response
  arma::mat M_gamma_y = join_horiz(X_u, Mu);
  arma::mat mu_tilde_y = M_gamma_y * mu_star;
  arma::mat V_ey = V_wu + (((1/alpha)-1)*eye<arma::mat>(m, m));
  arma::mat M_tilde_y = (M_gamma_y * V_star * trans(M_gamma_y)) + V_ey;

  // sample from posterior predictive distribution
  arma::cube smp_cube_w;
  arma::cube smp_cube_y;

  if (R > 1) {
    // For R greater than 1, we directly obtain the cube
    smp_cube_w = as<arma::cube>(rMT_R(Named("n", R), Named("Lambda", mu_tilde_w), Named("SigmaR", M_tilde_w), Named("SigmaC", Psi_star), Named("nu", nu_star)));
    smp_cube_y = as<arma::cube>(rMT_R(Named("n", R), Named("Lambda", mu_tilde_y), Named("SigmaR", M_tilde_y), Named("SigmaC", Psi_star), Named("nu", nu_star)));

  } else {
    // For R equal to 1, we obtain the matrix and reshape it into a cube
    arma::mat smp_mat_w = as<arma::mat>(rMT_R(Named("n", R), Named("Lambda", mu_tilde_w), Named("SigmaR", M_tilde_w), Named("SigmaC", Psi_star), Named("nu", nu_star)));
    smp_cube_w = arma::cube(smp_mat_w.memptr(), smp_mat_w.n_rows, smp_mat_w.n_cols, 1);
    arma::mat smp_mat_y = as<arma::mat>(rMT_R(Named("n", R), Named("Lambda", mu_tilde_y), Named("SigmaR", M_tilde_y), Named("SigmaC", Psi_star), Named("nu", nu_star)));
    smp_cube_y = arma::cube(smp_mat_y.memptr(), smp_mat_y.n_rows, smp_mat_y.n_cols, 1);

  }

  return List::create(Named("Wu") = smp_cube_w,
                      Named("Yu") = smp_cube_y);

}


'

Rcpp::sourceCpp(code = code_mvt)

# -------------------------------------------------------------------------


code_mvt2 <-
  '
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List r_pred_latent_MC2(const List& data, const arma::mat& X_u, const arma::mat& d_u, const arma::mat& d_us, const List& hyperpar, const List& poster, const List& post) {

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
  int q = Y.n_cols;
  int R = post.size();

  // covariance matrices
  arma::mat Rphi_u = exp(-phi * d_u);
  arma::mat Rphi_us = exp(-phi * d_us.submat(0, m, m-1, m+n-1));

  // sampling environment
  Rcpp::Environment mniw = Rcpp::Environment::namespace_env("mniw");
  Rcpp::Function rMNorm_R = mniw["rMNorm"];

  // compute reusable
  arma::mat M_u = Rphi_us * iR_s;
  arma::mat V_u = Rphi_u - (M_u * trans(Rphi_us));

  // initialize return objects
  arma::cube smp_cube_w(m, q, R);
  arma::cube smp_cube_y(m, q, R);

  for (int r = 0; r < R; ++r) {

    List smp = post(r);
    arma::mat beta = as<arma::mat>(smp["beta"]);
    arma::mat sigma = as<arma::mat>(smp["sigma"]);

    // predictive conjugate parameters
    arma::mat b = beta.rows(0, p - 1);
    arma::mat w = beta.rows(p, (n+p) - 1);

    // prediction W_u
    arma::mat mu_u = M_u * w;
    arma::mat resultW = as<arma::mat>(rMNorm_R(Named("n", 1), Named("Lambda", mu_u), Named("SigmaR", V_u), Named("SigmaC", sigma)));

    // prediction Y_u
    arma::mat mu_y = X_u * b + resultW;
    double a = alpha / (1 - alpha);
    arma::mat V_y = a * eye<arma::mat>(m, m);
    arma::mat resultY = as<arma::mat>(rMNorm_R(Named("n", 1), Named("Lambda", mu_y), Named("SigmaR", V_y), Named("SigmaC", sigma)));

    smp_cube_w.slice(r) = resultW;
    smp_cube_y.slice(r) = resultY;

  }

  return List::create(Named("Wu") = smp_cube_w,
                      Named("Yu") = smp_cube_y);

}
'

Rcpp::sourceCpp(code = code_mvt2)
