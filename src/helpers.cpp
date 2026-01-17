#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include "helpers.h"

using namespace Rcpp;
using namespace RcppParallel;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]

// Fenwick member functions
void Fenwick::add(std::size_t i) {
  for (++i; i < tree.size(); i += i & -i)
    tree[i]++;
}

int Fenwick::sum(std::size_t i) const {
  int s = 0;
  for (++i; i > 0; i -= i & -i)
    s += tree[i];
  return s;
}

// riskset_worker member functions
void riskset_worker::operator() (std::size_t begin, std::size_t end) {
  
  Fenwick fw(n);
  std::size_t k = 0;

  for (std::size_t jj = begin; jj < end; jj++) {
    
    std::size_t j = x_ord[jj];
  
    while (k < n && x[x_ord[k]] >= x[j]) {
      fw.add(y_rank[x_ord[k]]);
      k++;
    }
    
    for (std::size_t i = 0; i < n; i++) {
      N(j, i) = fw.sum(n - 1) - fw.sum(y_rank[i] - 1);
    }
  }
}

// [[Rcpp::export]]
arma::mat row_kron(const arma::mat& X, const arma::mat& Y) {
  
  int m = X.n_rows;
  int n = X.n_cols;
  int p = Y.n_cols;
  arma::mat Z(m, n * p);

  for (int i = 0; i < m; ++i) {
    // Compute Kronecker product of row i of X and row i of Y
    arma::rowvec kron_row = arma::kron(X.row(i), Y.row(i));
    Z.row(i) = kron_row;
  }
  
  return Z;
  }

// [[Rcpp::export]]
IntegerMatrix indgreater(NumericVector x) {
  int n = x.size();
  IntegerMatrix elem(n);
  for (int j=0; j<n; j++) {
    for (int i=0; i<n; i++) {
      if (x[j] >= x[i]) {
        elem(j,i) = 1;
      } else {
        elem(j,i) = 0;
      }
    }
  }
  return elem;
  }

// [[Rcpp::export]]
IntegerMatrix indless(NumericVector x) {
  int n = x.size();
  IntegerMatrix elem(n);
  for (int j=0; j<n; j++) {
    for (int i=0; i<n; i++) {
      if (x[j] <= x[i]) {
        elem(j,i) = 1;
      } else {
        elem(j,i) = 0;
      }
    }
  }
  return elem;
}

// [[Rcpp::export]]
IntegerMatrix indequal(NumericVector x) {
  int n = x.size();
  IntegerMatrix elem(n);
  for (int j=0; j<n; j++) {
    for (int i=0; i<n; i++) {
      if (x[j] == x[i]) {
        elem(j,i) = 1;
       } else {
        elem(j,i) = 0;
      }
    }
  }
  return elem;
}
// // [[Rcpp::export]]
// IntegerMatrix risksetC(NumericVector x, NumericVector y) {
//   
//   std::size_t n = x.size();
//   IntegerMatrix risksetmat(n);
//   
//   // Worker
//   riskset riskset(x,y,risksetmat);
//   
//   // Parallel loop
//   parallelFor(0, n, riskset);
//   
//   return risksetmat;
// }

// [[Rcpp::export]]
IntegerMatrix riskset_fast(NumericVector x, NumericVector y) {
  
  std::size_t n = x.size();
  IntegerMatrix N(n);

  // y ranks
  std::vector<std::size_t> y_ord(n);
  std::iota(y_ord.begin(), y_ord.end(), 0);
  std::sort(y_ord.begin(), y_ord.end(),
            [&](size_t a, size_t b){ return y[a] < y[b]; });

  std::vector<std::size_t> y_rank(n);
  for (std::size_t i = 0; i < n; i++)
    y_rank[y_ord[i]] = i;

  // x order descending
  std::vector<std::size_t> x_ord(n);
  std::iota(x_ord.begin(), x_ord.end(), 0);
  std::sort(x_ord.begin(), x_ord.end(),
            [&](size_t a, size_t b){ return x[a] > x[b]; });

  // Parallel worker
  riskset_worker w(x, y_rank, x_ord, N);
  parallelFor(0, n, w);

  return N;
}

// [[Rcpp::export]]
IntegerMatrix delta(NumericVector x, NumericVector y) {
  int n = x.size();
  IntegerMatrix delta(n);
  for (int j=0; j<n; j++) {
    for (int i=0; i<n; i++) {
      delta (j,i) = x[j]*y[i];
    }
  }
  return delta;
}