#include <RcppArmadillo.h>
#include "code.h"

#include <omp.h>
// [[Rcpp::plugins(openmp)]]

#include <RcppClock.h>
#include <thread>
//[[Rcpp::depends(RcppClock)]]

using namespace Rcpp;
using namespace arma;

// C++ general OpenMP code structure
//
// main ()  {
//
//   int var1, var2, var3;
//
//   Serial code
//     .
//     .
//     .
//
//   Beginning of parallel section. Fork a team of threads.
//   Specify variable scoping
//
// #pragma omp parallel private(var1, var2) shared(var3)
//
// {
//
//   Parallel section executed by all threads
//   .
//   Other OpenMP directives
//   .
//   Run-time Library calls
//   .
//   All threads join master thread and disband
//
// }
//
// Resume serial code
//   .
//   .
//   .
//
// }


//' Perform Accelerated Spatial Meta Kriging (ASMK)
//'
//' @param data [list] three elements: first named \eqn{Y}, second named \eqn{X}, third named \eqn{crd}
//' @param priors [list] priors: named \eqn{\mu_b},\eqn{V_b},\eqn{a},\eqn{b}
//' @param hyperpar [list] two elemets: first named \eqn{\delta}, second named \eqn{\phi}
//' @param K [integer] number of subsets
//' @param newdata [list] two elements: second named \eqn{X}, third named \eqn{crd}
//' @param R [integer] number of poterior predictive sample
//'
//' @return [list] posterior update parameters
//' @export
// [[Rcpp::export]]
List spASMK(const List& data, const List& priors, const List& hyperpar, const int& K, const List& newdata, const int& R = 250) {

  // Profiling
  Rcpp::Clock clock; // mind the headers: //[[Rcpp::depends(RcppClock)]] #include <RcppClock.h> #include <thread>

  // Unpack data
  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  arma::mat coords = as<arma::mat>(data["crd"]);
  // int n = Y.n_rows;
  // int q = Y.n_cols;
  // int p = X.n_cols;

  // // hyperparamters grid
  // arma::vec Delta = hyperpar["delta"];
  // arma::vec Phi = hyperpar["phi"];
  // arma::mat Grid = expand_grid_cpp(Delta, Phi);
  // int K = Grid.n_rows;

  // Subset data
  List subsets = subset_data(data, K);
  List Y_list = subsets[0];
  List X_list = subsets[1];
  List crd_list = subsets[2];

  // Fit subset BPS-GP
  List fit_list(K);
  double threshold = 1.0 / (2.0 * K);

  clock.tick("Subset model fitting");
  // for loop over K
  for (int i = 0; i < K; ++i) {

    // subset data
    arma::mat Y_i = as<arma::mat>(Y_list[i]);
    arma::mat X_i = as<arma::mat>(X_list[i]);
    arma::mat crd_i = as<arma::mat>(crd_list[i]);
    List data_i = List::create(Named("Y") = Y_i,
                               Named("X") = X_i);

    // fit subset model
    List out_i = BPSweights_cpp2(data_i, priors, crd_i, hyperpar);

    // extract results
    arma::mat epd_i = as<arma::mat>(out_i["epd"]);
    arma::mat W_i = as<arma::mat>(out_i["W"]);
    W_i.elem(find(W_i < threshold)).zeros();

    // return
    List fit_i = List::create(Named("epd") = epd_i,
                              Named("W") = W_i);
    fit_list[i] = fit_i;

  }
  clock.tock("Subset model fitting");

  // Combine subset models
  clock.tick("Subset models aggregation");
  List comb_list = BPS_combine(fit_list, K);
  clock.tock("Subset models aggregation");
  arma::mat Wbps = as<arma::mat>(comb_list[0]);
  List W_list = comb_list[1];

  // Perform predictions
  // List pred_list(R); // think about two matrices instead
  arma::mat pred_Z;
  arma::mat pred_Y;
  arma::vec W_vec = arma::conv_to<arma::vec>::from(Wbps.col(0));
  arma::uvec subset_ind = sample_index(K, R, W_vec);

  clock.tick("Predictions");
  // for loop over R
  for (int r = 0; r < R; ++r) {

    // model sample
    int ind_r = subset_ind(r);
    arma::mat Y_r = as<arma::mat>(Y_list[ind_r]);
    arma::mat X_r = as<arma::mat>(X_list[ind_r]);
    arma::mat crd_r = as<arma::mat>(crd_list[ind_r]);
    arma::mat W_r = as<arma::mat>(W_list[ind_r]);
    List data_r = List::create(Named("Y") = Y_r,
                               Named("X") = X_r);

    // newdata
    arma::mat X_u = as<arma::mat>(newdata["X"]);
    arma::mat crd_u = as<arma::mat>(newdata["crd"]);

    // perform predictions
    // int jj = X_u.n_rows/500;
    List out_r = spPredict_ASMK(data_r, X_u, priors, crd_r, crd_u, hyperpar, W_r, 1, 1);
    // pred_list[r] = out_r;
    arma::mat out_rZ = out_r[0];
    pred_Z = join_horiz(pred_Z, out_rZ);
    arma::mat out_rY = out_r[1];
    pred_Y = join_horiz(pred_Y, out_rY);

  }
  clock.tock("Predictions");

  clock.stop("Timing");

  // Return results
  List pred_list = List::create(Named("Z_hat") = pred_Z,
                                Named("Y_hat") = pred_Y);
  return List::create(Named("Predictions") = pred_list,
                      Named("Comb_weights") = Wbps);

}


