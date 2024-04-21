code <-
'
#include <RcppArmadillo.h>
#include <RcppClock.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppClock)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export(name = "arma_dist")]]
arma::mat arma_dist(const arma::mat & X){
  int n = X.n_rows;
  arma::mat D(n, n, arma::fill::zeros); // Allocate a matrix of dimension n x n
  for (int i = 0; i < n; i++) {
    for(int k = 0; k < i; k++){
      D(i, k) = sqrt(sum(pow(X.row(i) - X.row(k), 2)));
      D(k, i) = D(i, k);
    }
  }
  return D;
}


// [[Rcpp::export]]
List fit_cpp_clock(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar) {

  Rcpp::Clock clock; // mind the headers: //[[Rcpp::depends(RcppClock)]] #include <RcppClock.h> #include <thread>

  clock.tick("1");
  // Unpack data and priors
  arma::vec Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  arma::vec mu_b = as<arma::vec>(priors["mu_b"]);
  arma::mat V_b = as<arma::mat>(priors["V_b"]);
  double b = as<double>(priors["b"]);
  double a = as<double>(priors["a"]);
  double delta = as<double>(hyperpar["delta"]);
  double phi = as<double>(hyperpar["phi"]);

  int n = Y.n_rows;
  int p = X.n_cols;
  clock.tock("1");

  // arma::mat d_s = C_dist(coords);
  arma::mat d_s = arma_dist(coords);
  arma::mat Rphi_s = exp(-phi * d_s);

  clock.tick("2");
  // build the aumentend linear sistem
  arma::vec zer_n(n, arma::fill::zeros);
  arma::vec Y_star = join_vert(Y, mu_b, zer_n);

  arma::mat Zer_np(n, p, arma::fill::zeros);
  arma::mat Zer_pn = trans(Zer_np);
  arma::mat X_1 = join_vert(X, eye<arma::mat>(p, p), Zer_np);
  arma::mat X_2 = join_vert(eye<arma::mat>(n, n), Zer_pn, eye<arma::mat>(n, n));
  arma::mat X_star = join_horiz(X_1, X_2);
  clock.tock("2");

  clock.tick("3");
  arma::mat iV_1 = join_vert((1/delta)*eye<arma::mat>(n, n), arma::mat(p+n, n, arma::fill::zeros));
  arma::mat iV_2 = join_vert(Zer_np, arma::inv(V_b), Zer_np);
  arma::mat iRphi_s = arma::inv(Rphi_s);
  arma::mat iV_3 = join_vert(arma::mat(n+p, n, arma::fill::zeros), iRphi_s);
  arma::mat iV_star = join_horiz(iV_1, iV_2, iV_3);
  clock.tock("3");

  clock.tick("4");
  // Precompute some reusable values
  arma::mat tX_star = trans(X_star);
  clock.tock("4");

  clock.tick("7");
  // conjugate posterior parameters
  // arma::mat iM_star = tX_star * iV_star * X_star;
  // arma::mat M_star = arma::inv(iM_star);
  // arma::mat tXVY = tX_star * iV_star * Y_star;
  // arma::vec gamma_hat = M_star * tXVY;

  arma::mat tXiV = tX_star * iV_star;
  arma::mat M_star = arma::inv(tXiV * X_star);
  arma::mat tXVY = tXiV * Y_star;
  arma::vec gamma_hat = M_star * tXVY;
  clock.tock("7");

  clock.tick("5");
  arma::mat Xgam = X_star * gamma_hat;
  arma::mat SXY = Y_star - (Xgam);
  arma::mat tSXY = trans(SXY);

  arma::mat bb = tSXY * iV_star * SXY;
  double b_star = b + 0.5 * as_scalar(bb);
  double a_star = a + (n/2);
  clock.tock("5");

  clock.stop("Timing");

  // Return results as an R list
  return List::create(Named("M_star") = M_star,
                      Named("gamma_hat") = gamma_hat,
                      Named("b_star") = b_star,
                      Named("a_star") = a_star,
                      Named("iRphi_s") = iRphi_s);
}

// Optimized version of fit_cpp function
// [[Rcpp::export]]
List fit_cpp_optimized(const List& data, const List& priors, const arma::mat& coords, const List& hyperpar) {

  // Unpack data and priors
  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  arma::vec mu_b = as<arma::vec>(priors["mu_b"]);
  arma::mat V_b = as<arma::mat>(priors["V_b"]);
  double b = as<double>(priors["b"]);
  double a = as<double>(priors["a"]);
  double delta = as<double>(hyperpar["delta"]);
  double phi = as<double>(hyperpar["phi"]);

  int n = Y.n_rows;

  arma::mat d_s = arma_dist(coords);
  arma::mat Rphi_s = exp(-phi * d_s);

  // Precompute some reusable values
  double d = (1 / delta);
  arma::mat tX = trans(X);
  arma::mat iV_b = arma::inv(V_b);
  arma::mat iR_s = arma::inv(Rphi_s);

  // Compute posterior updating
  arma::mat iM_B = d * tX * X + iV_b;
  arma::mat iM_BW = d * tX;
  arma::mat iM_WB = trans(iM_BW);
  arma::mat iM_W = iR_s + (d * eye<arma::mat>(n, n));

  arma::mat iM_star1 = join_horiz(iM_B, iM_BW);
  arma::mat iM_star2 = join_horiz(iM_WB, iM_W);
  arma::mat iM_star = join_vert( iM_star1, iM_star2);
  arma::mat M_star = arma::inv(iM_star);

  arma::vec M = join_vert( (d * tX * Y) + (iV_b * mu_b) , d * Y );
  arma::vec gamma_hat = M_star * M;

  double dYY = as_scalar(d * trans(Y) * Y);
  double mbVbmb = as_scalar(trans(mu_b) * iV_b * mu_b);
  double bb = dYY + mbVbmb + as_scalar(trans(gamma_hat) * iM_star * gamma_hat) - as_scalar(2 * trans(gamma_hat) * M);

  double b_star = b + 0.5 * as_scalar(bb);
  double a_star = a + (n/2);

  // Return results as an R list
  return List::create(Named("M_star") = M_star,
                      Named("gamma_hat") = gamma_hat,
                      Named("b_star") = b_star,
                      Named("a_star") = a_star,
                      Named("iRphi_s") = iR_s);
}
'
Rcpp::sourceCpp(code = code)


system("gfortran -shared -fPIC -o fit_fortran.so fit_fortran.f90")



code2 <- '
#include <RcppArmadillo.h>
#include <cmath>

extern "C" {
  void matmul_(float *a, float *b, float *c, int *m, int *n, int *p);
  void sgesv_(int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, int *info);
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List fit_cpp(const arma::mat& A, const arma::mat& B) {
  int m = A.n_rows, n = A.n_cols, p = B.n_cols;
  arma::mat C(m, p);

  // Matrix multiplication using LAPACK matmul
  matmul_(A.memptr(), B.memptr(), C.memptr(), &m, &n, &p);

  // Matrix inversion using LAPACK sgesv
  arma::mat invA(n, n);
  arma::uvec ipiv(n);
  int info;
  sgesv_(&n, &n, A.memptr(), &n, ipiv.memptr(), invA.memptr(), &n, &info);

  return Rcpp::List::create(Rcpp::Named("MatrixMultiplication") = C,
                            Rcpp::Named("MatrixInversion") = invA,
                            Rcpp::Named("Info") = info);
}'


# Source the Rcpp file
Rcpp::sourceCpp(code = code2)

# Generate sample matrices
set.seed(42)
A <- matrix(rnorm(9), nrow = 3)
B <- matrix(rnorm(9), nrow = 3)

# Call the Rcpp function
result <- fit_cpp(A, B)

# Print the result
print(result)


system("gfortran -shared -fPIC -o fit_fortran.so fit_fortran.f90")
