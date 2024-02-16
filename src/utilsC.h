
#ifndef UTILSC_H
#define UTILSC_H

// Function declarations

arma::mat arma_dist(const arma::mat & X);

SEXP cDist(SEXP coords1_r, SEXP n1_r, SEXP coords2_r, SEXP n2_r, SEXP p_r, SEXP D_r);

Rcpp::NumericMatrix C_dist(Rcpp::NumericMatrix coords1);

arma::mat expand_grid_cpp(const arma::vec& x, const arma::vec& y);

arma::uvec sample_index(const int& size, const int& length, const arma::vec& p);

Rcpp::List subset_data(const Rcpp::List& data, int K);

#endif // UTILSC_H
