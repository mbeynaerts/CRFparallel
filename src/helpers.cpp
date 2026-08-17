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
arma::mat band_chol_cpp(arma::mat B) {
    // Armadillo's chol() can handle banded matrices if you 
    // provide the full matrix, but for memory efficiency with
    // large banded matrices, we use the 'trimatu' or 'trimatl' wrappers.
    
    // If B is already in the compact banded format (k x n):
    // It's often better to reconstruct or use a Sparse matrix if k << n.
    // However, for a direct port of your logic:
    
    arma::mat R;
    bool success = arma::chol(R, B);
    
    if (!success) {
        Rcpp::stop("Decomposition failed: matrix is not positive definite.");
    }
    
    return R;
}

// [[Rcpp::export]]
arma::vec sdiag_cpp(const arma::mat& A, int k = 0) {
    // Basic bounds checking to match R logic
    if (k >= static_cast<int>(A.n_cols) || k <= -static_cast<int>(A.n_rows)) {
        return arma::vec(); // Returns an empty vector
    }
    
    // .diag(k) handles:
    // k = 0: main diagonal
    // k > 0: super-diagonals (above main)
    // k < 0: sub-diagonals (below main)
    return A.diag(k);
}

// [[Rcpp::export]]
arma::mat compute_D1_cpp(const arma::mat& D, const arma::mat& W1, const arma::vec& h, int pord) {
    int n_h = h.n_elem;
    int n_D_cols = D.n_cols;
    int n_D_rows = D.n_rows;
    
    // Construct the banded weights matrix B (in compact storage)
    // We want a matrix of size (pord + 1) x n_D_rows
    arma::mat B(pord + 1, n_D_rows, arma::fill::zeros);
    
    // Leading diagonal logic
    arma::vec diag_W1 = W1.diag();
    arma::vec ld0 = arma::vectorise(diag_W1 * h.t()); // length: (pord+1) * n_h
    
    // Map ld0 to the global diagonal ld
    arma::vec ld = arma::zeros<arma::vec>(n_D_rows);
    for(int i = 0; i < n_h; ++i) {
        for(int j = 0; j <= pord; ++j) {
            int global_idx = i * pord + j;
            if (global_idx < n_D_rows) {
                ld(global_idx) += diag_W1(j) * h(i);
            }
        }
    }
    B.row(0) = ld.t();

    // Other diagonals
    for (int k = 1; k <= pord; ++k) {
        arma::vec diwk = W1.diag(k);
        // Fill B.row(k) based on piece-wise quadrature
        for (int i = 0; i < n_h; ++i) {
            for (int j = 0; j < diwk.n_elem; ++j) {
                int global_idx = i * pord + j;
                if (global_idx < n_D_rows - k) {
                    B(k, global_idx) = diwk(j) * h(i);
                }
            }
        }
    }

    // Banded Cholesky Factorization
    // We use the band_chol_cpp logic here (assuming your LAPACK wrapper is available)
    // Or use Armadillo's internal routines if B is converted to sparse.
    arma::mat R = band_chol_cpp(B);

    // Multiply D by the Cholesky factor (Banded matrix multiplication)
    arma::mat D1 = arma::zeros<arma::mat>(n_D_rows, n_D_cols);
    
    // Row 0 scaling
    D1 = D.each_col() % R.row(0).t();
    
    // Off-diagonal contributions
    for (int k = 1; k <= pord; ++k) {
        for (int i = 0; i < (n_D_rows - k); ++i) {
            D1.row(i) += R(k, i) * D.row(i + k);
        }
    }

    // S = D1' * D1
    // return D1.t() * D1;
    return D1;
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
arma::Mat<int> riskset_fast(arma::vec x, arma::vec y) {
  
  std::size_t n = x.n_elem;
  arma::Mat<int> N(n, n, arma::fill::zeros);

  // ponytail: O(n^3) is fine for current data sizes; replace with a tested
  // Fenwick implementation if risk-set construction becomes the bottleneck.
  for (std::size_t j = 0; j < n; j++) {
    for (std::size_t i = 0; i < n; i++) {
      for (std::size_t k = 0; k < n; k++) {
        if (x[k] >= x[j] && y[k] >= y[i]) {
          N(j, i)++;
        }
      }
    }
  }

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
