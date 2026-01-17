#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include "helpers.h"

using namespace Rcpp;
using namespace RcppParallel;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]

struct DataInput {
  
  // vectors
  const arma::colvec riskset1;
  const arma::colvec riskset2;
  const arma::colvec delta1;
  const arma::colvec delta2;
  const arma::colvec I1;
  const arma::colvec I2;
  const arma::colvec I3;
  const arma::colvec I4;
  const arma::colvec I5;
  const arma::colvec I6;
  
  // indices
  const arma::uvec idxN1;
  const arma::uvec idxN2;

  // constructor from List
  explicit DataInput(const Rcpp::List& lst)
    : riskset1(lst["riskset1"]),
      riskset2(lst["riskset2"]),
      delta1(lst["delta1"]),
      delta2(lst["delta2"]),
      I1(lst["I1"]),
      I2(lst["I2"]),
      I3(lst["I3"]),
      I4(lst["I4"]),
      I5(lst["I5"]),
      I6(lst["I6"]),
      idxN1(lst["idxN1"]),
      idxN2(lst["idxN2"]) {}
  
  // Getter functions
  inline const arma::colvec& get_riskset1() const { return riskset1; }
  inline const arma::colvec& get_riskset2() const { return riskset2; }
  inline const arma::colvec& get_I1() const { return I1; }
  inline const arma::colvec& get_I2() const { return I2; }
  inline const arma::colvec& get_I3() const { return I3; }
  inline const arma::colvec& get_I4() const { return I4; }
  inline const arma::colvec& get_I5() const { return I5; }
  inline const arma::colvec& get_I6() const { return I6; }
  inline const arma::colvec&  get_delta1() const { return delta1; }
  inline const arma::colvec&  get_delta2() const { return delta2; }
  inline const arma::uvec&  get_idxN1() const { return idxN1; }
  inline const arma::uvec&  get_idxN2() const { return idxN2; }
  
  };

struct HessianWorker : public RcppParallel::Worker {
  const arma::uvec& indices;
  const arma::colvec& weights;
  const arma::mat& X1;
  const arma::mat& X2;
  int dim;
  int n1;
  int n2;
  bool is_type_1;
  int batch_size;
  
  // Output: each thread has its own local Hessian to avoid race conditions
  arma::mat local_H;
  
  HessianWorker(const arma::uvec& indices, const arma::colvec& weights,
                const arma::mat& X1, const arma::mat& X2, 
                int dim, bool is_type_1, int batch_size)
    : indices(indices), weights(weights), X1(X1), X2(X2),
      dim(dim), is_type_1(is_type_1), batch_size(batch_size) {
    n1 = X1.n_rows;
    n2 = X2.n_rows;
    local_H.zeros(dim * dim, dim * dim);
  }
  
  // Overload for splitting the work
  HessianWorker(const HessianWorker& other, RcppParallel::Split)
    : indices(other.indices), weights(other.weights), X1(other.X1), X2(other.X2),
      dim(other.dim), n1(other.n1), n2(other.n2), 
      is_type_1(other.is_type_1), batch_size(other.batch_size) {
    local_H.zeros(dim * dim, dim * dim);
  }
  
  void operator()(std::size_t begin, std::size_t end) {
    int totalparam = dim * dim;
    
    for (std::size_t b = begin; b < end; b += batch_size) {
      int current_batch = std::min((int)batch_size, (int)(end - b));
      arma::mat D_sub(current_batch, totalparam);
      
      for (int k = 0; k < current_batch; ++k) {
        arma::uword idx = indices(b + k);
        arma::uword row_i, row_j;
        
        if (is_type_1) { // Row-major (Original D1)
          row_i = idx / n2;
          row_j = idx % n2;
        } else {         // Col-major (Original D2)
          row_i = idx % n1;
          row_j = idx / n1;
        }
        
        // Vectorized row fill
        arma::rowvec r1 = X1.row(row_i);
        arma::rowvec r2 = X2.row(row_j);
        D_sub.row(k) = arma::vectorise(r1.t() * r2).t();
      }
      
      arma::colvec w_sub = weights.subvec(b, b + current_batch - 1);
      local_H -= (D_sub.each_col() % w_sub).t() * D_sub;
    }
  }
  
  // Join the results of two threads
  void join(const HessianWorker& other) {
    local_H += other.local_H;
  }
};



// [[Rcpp::export]]
double loglikC(arma::colvec& x,
               const Rcpp::List& datalist,
               const arma::mat& X1,
               const arma::mat& X2) {

  int dim = X1.n_cols;
  
  DataInput input(datalist);
  
  const arma::colvec& riskset1 = input.get_riskset1();
  const arma::colvec& riskset2 = input.get_riskset2();
  const arma::colvec& delta1 = input.get_delta1();
  const arma::colvec& delta2 = input.get_delta2();
  const arma::colvec& I1 = input.get_I1();
  const arma::colvec& I2 = input.get_I2();
  const arma::colvec& I3 = input.get_I3();
  const arma::colvec& I4 = input.get_I4();
  const arma::colvec& I5 = input.get_I5();
  const arma::colvec& I6 = input.get_I6();
  const arma::uvec& idxN1 = input.get_idxN1();
  const arma::uvec& idxN2 = input.get_idxN2();
  
  // Calculation of logtheta
  arma::mat coefmat(x.memptr(), dim, dim, false, true);
  arma::mat logtheta = X1 * coefmat * X2.t();
  arma::colvec logtheta2 = arma::vectorise(logtheta);
  arma::colvec logtheta1 = arma::vectorise(logtheta.t());

  double sum1 = arma::accu(delta1 % I1 % (logtheta1.elem(idxN1) % I5 - arma::log(riskset1 + I2 % arma::exp(logtheta1.elem(idxN1)) - I2)));
  double sum2 = arma::accu(delta2 % I3 % (logtheta2.elem(idxN2) % I6 - arma::log(riskset2 + I4 % arma::exp(logtheta2.elem(idxN2)) - I4)));

  return(-sum1-sum2);
}


// [[Rcpp::export]]
arma::vec gradient_fast(arma::colvec& x,
                            const Rcpp::List& datalist,
                            const arma::mat &X1,
                            const arma::mat &X2) {
  
  int dim = X1.n_cols;
  int K = X1.n_rows;
  DataInput input(datalist);
  
  // 1. Map x to a matrix without copying
  arma::mat coefmat(x.memptr(), dim, dim, false, true);
  
  // 2. Calculate logtheta
  // logtheta(i, j) = X1.row(i) * coefmat * X2.row(j).t()
  arma::mat logtheta = X1 * coefmat * X2.t();
  arma::colvec logtheta2(logtheta.memptr(), logtheta.n_elem, false, true);
  arma::colvec logtheta1 = vectorise(logtheta.t());

  // 3. Compute common components efficiently
  // Instead of vectorising everything, we compute the weights into matrix form
  arma::mat W1 = arma::zeros<arma::mat>(K, K);
  arma::mat W2 = arma::zeros<arma::mat>(K, K);
  
  // Pre-extract indices and vectors to avoid repeated get calls
  const arma::uvec& idxN1 = input.get_idxN1();
  const arma::uvec& idxN2 = input.get_idxN2();
  const arma::colvec& riskset1 = input.get_riskset1();
  const arma::colvec& riskset2 = input.get_riskset2();
  const arma::colvec& delta1 = input.get_delta1();
  const arma::colvec& delta2 = input.get_delta2();
  const arma::colvec& I1 = input.get_I1();
  const arma::colvec& I2 = input.get_I2();
  const arma::colvec& I3 = input.get_I3();
  const arma::colvec& I4 = input.get_I4();
  const arma::colvec& I5 = input.get_I5();
  const arma::colvec& I6 = input.get_I6();
  
  // Vectorized weight calculation (subsetting only what's needed)
  arma::vec v_common1 = delta1 % I1 % (I5 - I2 / ((riskset1 - I2) % arma::exp(-logtheta1.elem(idxN1)) + I2));
  arma::vec v_common2 = delta2 % I3 % (I6 - I4 / ((riskset2 - I4) % arma::exp(-logtheta2.elem(idxN2)) + I4));
  
  // Map weights back to KxK structure
  W1.elem(idxN1) = v_common1; // W1 corresponds to the transposed logic in your code
  W2.elem(idxN2) = v_common2;
  
  // 4. The "Magic" Step: Replace the loop with Matrix Multiplications
  // The sum of (common * kron) is equivalent to X1.T * WeightMatrix * X2
  // We handle the indices by using the sparse-like weight matrices W1 and W2
  
  arma::mat grad_mat = -(X1.t() * W1.t() * X2) - (X1.t() * W2 * X2);
  
  // 5. Return as a vector
  return arma::vectorise(grad_mat);
}

// [[Rcpp::export]]
arma::mat hessian_fast(arma::colvec& x, // Cannot be constant due to advanced constructor on coefmat
                       const Rcpp::List& datalist,
                       const arma::mat& X1,
                       const arma::mat& X2) {
  
  DataInput input(datalist);
  
  const arma::colvec& riskset1 = input.get_riskset1();
  const arma::colvec& riskset2 = input.get_riskset2();
  const arma::colvec& delta1 = input.get_delta1();
  const arma::colvec& delta2 = input.get_delta2();
  const arma::colvec& I1 = input.get_I1();
  const arma::colvec& I2 = input.get_I2();
  const arma::colvec& I3 = input.get_I3();
  const arma::colvec& I4 = input.get_I4();
  const arma::uvec& idxN1 = input.get_idxN1();
  const arma::uvec& idxN2 = input.get_idxN2();
  
  int dim = X1.n_cols;
  int totalparam = dim * dim;
  
  // Calculate logtheta
  arma::mat coefmat(x.memptr(), dim, dim, false, true);
  arma::mat logtheta = X1 * coefmat * X2.t();
  arma::colvec logtheta2 = arma::vectorise(logtheta);
  arma::colvec logtheta1 = arma::vectorise(logtheta.t());

  // Calculate common part of hessian
  arma::colvec exp_neg1 = arma::exp(-logtheta1.elem(idxN1));
  arma::colvec exp_neg2 = arma::exp(-logtheta2.elem(idxN2));

  arma::colvec denom1 = arma::square((riskset1 - I2) % exp_neg1 + I2);
  arma::colvec denom2 = arma::square((riskset2 - I4) % exp_neg2 + I4);

  const double eps = 1e-12;
  denom1.transform([&](double x){ return std::max(x, eps); });
  denom2.transform([&](double x){ return std::max(x, eps); });

  arma::colvec common1 = -delta1 % I1 % I2 % (riskset1 - I2) % exp_neg1 / denom1;
  arma::colvec common2 = -delta2 % I3 % I4 % (riskset2 - I4) % exp_neg2 / denom2;
    
    if (arma::all(common1 == 0) && arma::all(common2 == 0))
      return arma::zeros(totalparam, totalparam);
    
    // --------------------------------------------------
    // Build indexed derivative matrices
    // --------------------------------------------------
    
    arma::mat D1(idxN1.n_elem, totalparam);
    arma::mat D2(idxN2.n_elem, totalparam);
    
    // HessianWorker worker(X1, X2, idxN1, idxN2, D1, D2);
    // parallelFor(0, totalparam, worker);
    
    for (int m = 0; m < totalparam; ++m) {
      int i = m % dim;
      int j = m / dim;

      arma::mat K = arma::kron(X1.col(i), X2.col(j).t());

      arma::colvec d1 = arma::vectorise(K.t());
      arma::colvec d2 = arma::vectorise(K);

      D1.col(m) = d1.elem(idxN1);
      D2.col(m) = d2.elem(idxN2);
    }
    
    // --------------------------------------------------
    // Hessian via weighted Gram matrices
    // --------------------------------------------------
    
    // arma::mat H = - (D1.t() * arma::diagmat(common1) * D1) - (D2.t() * arma::diagmat(common2) * D2);
    arma::mat H = - (D1.each_col() % common1).t() * D1 - (D2.each_col() % common2).t() * D2;
    
      
      return H;
}


// Memory efficient Hessian calculation
// [[Rcpp::export]]
arma::mat hessian_fast_efficient(arma::colvec& x, 
                                 const Rcpp::List& datalist,
                                 const arma::mat& X1,
                                 const arma::mat& X2) {
  
  DataInput input(datalist);
  
  const arma::colvec& riskset1 = input.get_riskset1();
  const arma::colvec& riskset2 = input.get_riskset2();
  const arma::colvec& delta1   = input.get_delta1();
  const arma::colvec& delta2   = input.get_delta2();
  const arma::colvec& I1       = input.get_I1();
  const arma::colvec& I2       = input.get_I2();
  const arma::colvec& I3       = input.get_I3();
  const arma::colvec& I4       = input.get_I4();
  const arma::uvec& idxN1      = input.get_idxN1();
  const arma::uvec& idxN2      = input.get_idxN2();
  
  int dim = X1.n_cols;
  int totalparam = dim * dim;
  
  // 1. Calculate logtheta with zero-copy view
  arma::mat coefmat(x.memptr(), dim, dim, false, true);
  arma::mat logtheta = X1 * coefmat * X2.t();
  
  // 2. Common weights (vectorise(..., 0) ensures no copy if possible)
  arma::colvec logtheta2 = arma::vectorise(logtheta);
  arma::colvec logtheta1 = arma::vectorise(logtheta.t());
  
  arma::colvec exp_neg1 = arma::exp(-logtheta1.elem(idxN1));
  arma::colvec exp_neg2 = arma::exp(-logtheta2.elem(idxN2));

  arma::colvec denom1 = arma::square((riskset1 - I2) % exp_neg1 + I2);
  arma::colvec denom2 = arma::square((riskset2 - I4) % exp_neg2 + I4);

  const double eps = 1e-12;
  denom1.transform([&](double v){ return std::max(v, eps); });
  denom2.transform([&](double v){ return std::max(v, eps); });

  arma::colvec common1 = -delta1 % I1 % I2 % (riskset1 - I2) % exp_neg1 / denom1;
  arma::colvec common2 = -delta2 % I3 % I4 % (riskset2 - I4) % exp_neg2 / denom2;

  // Pre-allocate Hessian
  arma::mat H(totalparam, totalparam, arma::fill::zeros);
  
  // We need to build D1 and D2 to perform the final H calculation.
  // To stay memory efficient, we build D1/D2 column by column.
  // This avoids the massive memory spike of 'arma::kron' matrices.
  
  arma::mat D1(idxN1.n_elem, totalparam);
  arma::mat D2(idxN2.n_elem, totalparam);

  for (int m = 0; m < totalparam; ++m) {
    int i = m % dim; // Column of X1
    int j = m / dim; // Column of X2
  
    // Original: d2 = vectorise(X1.col(i) * X2.col(j).t())
    // This is an outer product. 
    // Element (r1, r2) of the outer product is X1(r1, i) * X2(r2, j)
    
    // For D2 (Col-major vectorization of outer product):
    // The index k in the vectorized vector corresponds to:
    // row_i = k % n1, row_j = k / n1
    for (arma::uword k = 0; k < idxN2.n_elem; ++k) {
      arma::uword idx = idxN2(k);
      D2(k, m) = X1(idx % X1.n_rows, i) * X2(idx / X1.n_rows, j);
    }
  
    // For D1 (Vectorization of the TRANSPOSE of the outer product):
    // This is equivalent to Row-major vectorization of the original outer product.
    // row_i = idx / n2, row_j = idx % n2
    for (arma::uword k = 0; k < idxN1.n_elem; ++k) {
      arma::uword idx = idxN1(k);
      D1(k, m) = X1(idx / X2.n_rows, i) * X2(idx % X2.n_rows, j);
    }
  }
  
  H = - (D1.each_col() % common1).t() * D1 - (D2.each_col() % common2).t() * D2;

  return H;
}

// [[Rcpp::export]]
arma::mat hessian_fast_batched(arma::colvec& x, 
                               const Rcpp::List& datalist,
                               const arma::mat& X1,
                               const arma::mat& X2,
                               int batch_size = 1000) {
  
  DataInput input(datalist);
  const arma::colvec& riskset1 = input.get_riskset1();
  const arma::colvec& riskset2 = input.get_riskset2();
  const arma::colvec& delta1   = input.get_delta1();
  const arma::colvec& delta2   = input.get_delta2();
  const arma::colvec& I1       = input.get_I1();
  const arma::colvec& I2       = input.get_I2();
  const arma::colvec& I3       = input.get_I3();
  const arma::colvec& I4       = input.get_I4();
  const arma::uvec& idxN1      = input.get_idxN1();
  const arma::uvec& idxN2      = input.get_idxN2();
  
  int dim = X1.n_cols;
  int totalparam = dim * dim;
  int n1 = X1.n_rows;
  int n2 = X2.n_rows;
  
  // 1. Calculate common weights (as before)
  arma::mat coefmat(x.memptr(), dim, dim, false, true);
  arma::mat logtheta = X1 * coefmat * X2.t();
  
  arma::colvec logtheta1 = arma::vectorise(logtheta.t());
  arma::colvec logtheta2 = arma::vectorise(logtheta);
  
  arma::colvec exp_neg1 = arma::exp(-logtheta1.elem(idxN1));
  arma::colvec exp_neg2 = arma::exp(-logtheta2.elem(idxN2));
  
  arma::colvec denom1 = arma::square((riskset1 - I2) % exp_neg1 + I2);
  arma::colvec denom2 = arma::square((riskset2 - I4) % exp_neg2 + I4);
  
  const double eps = 1e-12;
  denom1.transform([&](double v){ return std::max(v, eps); });
  denom2.transform([&](double v){ return std::max(v, eps); });

  arma::colvec common1 = -delta1 % I1 % I2 % (riskset1 - I2) % exp_neg1 / denom1;
  arma::colvec common2 = -delta2 % I3 % I4 % (riskset2 - I4) % exp_neg2 / denom2;

  arma::mat H = arma::zeros<arma::mat>(totalparam, totalparam);

  // --- Process D1 Batches (Row-major vectorization of K) ---
  int n_idx1 = idxN1.n_elem;
  for (int b = 0; b < n_idx1; b += batch_size) {
    int current_batch = std::min(batch_size, n_idx1 - b);
    arma::mat D1_sub(current_batch, totalparam);
    
    for (int k = 0; k < current_batch; ++k) {
      arma::uword idx = idxN1(b + k);
      arma::uword row_i = idx / n2;
      arma::uword row_j = idx % n2;
      
      // Fill the row of D1_sub corresponding to the m parameters
      for (int m = 0; m < totalparam; ++m) {
        D1_sub(k, m) = X1(row_i, m % dim) * X2(row_j, m / dim);
      }
    }
  
    arma::colvec w_sub = common1.subvec(b, b + current_batch - 1);
    
    // H -= D1_sub.T * diag(w_sub) * D1_sub
    H -= (D1_sub.each_col() % w_sub).t() * D1_sub;
  }

  // --- Process D2 Batches (Col-major vectorization of K) ---
  int n_idx2 = idxN2.n_elem;
  for (int b = 0; b < n_idx2; b += batch_size) {
    int current_batch = std::min(batch_size, n_idx2 - b);
    arma::mat D2_sub(current_batch, totalparam);
  
    for (int k = 0; k < current_batch; ++k) {
      arma::uword idx = idxN2(b + k);
      arma::uword row_i = idx % n1;
      arma::uword row_j = idx / n1;
    
      for (int m = 0; m < totalparam; ++m) {
        D2_sub(k, m) = X1(row_i, m % dim) * X2(row_j, m / dim);
      }
    }
  
    arma::colvec w_sub = common2.subvec(b, b + current_batch - 1);
    H -= (D2_sub.each_col() % w_sub).t() * D2_sub;
  }
  return H;
}

// [[Rcpp::export]]
arma::mat hessian_fast_batched_parallel(arma::colvec& x, 
                                        const Rcpp::List& datalist,
                                        const arma::mat& X1,
                                        const arma::mat& X2,
                                        int batch_size = 1000) {
  
  DataInput input(datalist);
  const arma::colvec& riskset1 = input.get_riskset1();
  const arma::colvec& riskset2 = input.get_riskset2();
  const arma::colvec& delta1   = input.get_delta1();
  const arma::colvec& delta2   = input.get_delta2();
  const arma::colvec& I1       = input.get_I1();
  const arma::colvec& I2       = input.get_I2();
  const arma::colvec& I3       = input.get_I3();
  const arma::colvec& I4       = input.get_I4();
  const arma::uvec& idxN1      = input.get_idxN1();
  const arma::uvec& idxN2      = input.get_idxN2();
  
  int dim = X1.n_cols;

  // 1. Calculate common weights (as before)
  arma::mat coefmat(x.memptr(), dim, dim, false, true);
  arma::mat logtheta = X1 * coefmat * X2.t();
  
  arma::colvec logtheta1 = arma::vectorise(logtheta.t());
  arma::colvec logtheta2 = arma::vectorise(logtheta);
  
  arma::colvec exp_neg1 = arma::exp(-logtheta1.elem(idxN1));
  arma::colvec exp_neg2 = arma::exp(-logtheta2.elem(idxN2));
  
  arma::colvec denom1 = arma::square((riskset1 - I2) % exp_neg1 + I2);
  arma::colvec denom2 = arma::square((riskset2 - I4) % exp_neg2 + I4);
  
  const double eps = 1e-12;
  denom1.transform([&](double v){ return std::max(v, eps); });
  denom2.transform([&](double v){ return std::max(v, eps); });
  
  arma::colvec common1 = -delta1 % I1 % I2 % (riskset1 - I2) % exp_neg1 / denom1;
  arma::colvec common2 = -delta2 % I3 % I4 % (riskset2 - I4) % exp_neg2 / denom2;

  HessianWorker worker1(idxN1, common1, X1, X2, dim, true, batch_size);
  RcppParallel::parallelReduce(0, idxN1.n_elem, worker1);
  
  HessianWorker worker2(idxN2, common2, X1, X2, dim, false, batch_size);
  RcppParallel::parallelReduce(0, idxN2.n_elem, worker2);
  
  arma::mat H = worker1.local_H + worker2.local_H;
  
  return H;
}

arma::mat logtheta_poly(arma::vec& beta,
                        const arma::mat& X1,
                        const arma::mat& X2,
                        int poly_order = 3) {
  int dim = X1.n_cols;
  
  arma::mat beta_mat = arma::zeros<arma::mat>(dim, dim);
  int k = 0;
  for (int j = 0; j < dim; ++j) {
    for (int i = 0; i < dim; ++i) {
      // Only map coefficients where the combined power is <= poly_order
      if (i + j <= poly_order) {
        beta_mat(i, j) = beta(k++);
      }
    }
  }
  
  arma::mat logtheta = X1 * beta_mat * X2.t();
  
  return logtheta;
}

// [[Rcpp::export]]
arma::vec gradient_poly_fast(arma::colvec& x,
                             const Rcpp::List& datalist,
                             const arma::mat &X1,
                             const arma::mat &X2) {
  
  int K = X1.n_rows;
  DataInput input(datalist);

  // 2. Calculate logtheta
  // logtheta(i, j) = X1.row(i) * coefmat * X2.row(j).t()
  arma::mat logtheta = logtheta_poly(x, X1, X2);
  arma::colvec logtheta2(logtheta.memptr(), logtheta.n_elem, false, true);
  arma::colvec logtheta1 = vectorise(logtheta.t());
  
  // 3. Compute common components efficiently
  // Instead of vectorising everything, we compute the weights into matrix form
  arma::mat W1 = arma::zeros<arma::mat>(K, K);
  arma::mat W2 = arma::zeros<arma::mat>(K, K);
  
  // Pre-extract indices and vectors to avoid repeated get calls
  const arma::uvec& idxN1 = input.get_idxN1();
  const arma::uvec& idxN2 = input.get_idxN2();
  const arma::colvec& riskset1 = input.get_riskset1();
  const arma::colvec& riskset2 = input.get_riskset2();
  const arma::colvec& delta1 = input.get_delta1();
  const arma::colvec& delta2 = input.get_delta2();
  const arma::colvec& I1 = input.get_I1();
  const arma::colvec& I2 = input.get_I2();
  const arma::colvec& I3 = input.get_I3();
  const arma::colvec& I4 = input.get_I4();
  const arma::colvec& I5 = input.get_I5();
  const arma::colvec& I6 = input.get_I6();
  
  // Vectorized weight calculation (subsetting only what's needed)
  arma::vec v_common1 = delta1 % I1 % (I5 - I2 / ((riskset1 - I2) % arma::exp(-logtheta1.elem(idxN1)) + I2));
  arma::vec v_common2 = delta2 % I3 % (I6 - I4 / ((riskset2 - I4) % arma::exp(-logtheta2.elem(idxN2)) + I4));

  // Map weights back to KxK structure
  W1.elem(idxN1) = v_common1; // W1 corresponds to the transposed logic in your code
  W2.elem(idxN2) = v_common2;
  
  // 4. The "Magic" Step: Replace the loop with Matrix Multiplications
  // The sum of (common * kron) is equivalent to X1.T * WeightMatrix * X2
  // We handle the indices by using the sparse-like weight matrices W1 and W2
  
  arma::mat grad_mat = -(X1.t() * W1.t() * X2) - (X1.t() * W2 * X2);
  
  // 5. Return as a vector
  return arma::vectorise(grad_mat);
}

// [[Rcpp::export]]
arma::mat hessian_poly_batched_parallel(arma::colvec& x, 
                                        const Rcpp::List& datalist,
                                        const arma::mat& X1,
                                        const arma::mat& X2,
                                        int batch_size = 1000) {

  DataInput input(datalist);
  const arma::colvec& riskset1 = input.get_riskset1();
  const arma::colvec& riskset2 = input.get_riskset2();
  const arma::colvec& delta1   = input.get_delta1();
  const arma::colvec& delta2   = input.get_delta2();
  const arma::colvec& I1       = input.get_I1();
  const arma::colvec& I2       = input.get_I2();
  const arma::colvec& I3       = input.get_I3();
  const arma::colvec& I4       = input.get_I4();
  const arma::uvec& idxN1      = input.get_idxN1();
  const arma::uvec& idxN2      = input.get_idxN2();
  
  int dim = X1.n_cols;
  
  // 1. Calculate common weights (as before)
  arma::mat logtheta = logtheta_poly(x, X1, X2);
  arma::colvec logtheta2(logtheta.memptr(), logtheta.n_elem, false, true);
  arma::colvec logtheta1 = vectorise(logtheta.t());
  
  // arma::colvec logtheta1 = arma::vectorise(logtheta.t());
  // arma::colvec logtheta2 = arma::vectorise(logtheta);
  
  arma::colvec exp_neg1 = arma::exp(-logtheta1.elem(idxN1));
  arma::colvec exp_neg2 = arma::exp(-logtheta2.elem(idxN2));
  
  arma::colvec denom1 = arma::square((riskset1 - I2) % exp_neg1 + I2);
  arma::colvec denom2 = arma::square((riskset2 - I4) % exp_neg2 + I4);
  
  const double eps = 1e-12;
  denom1.transform([&](double v){ return std::max(v, eps); });
  denom2.transform([&](double v){ return std::max(v, eps); });
  
  arma::colvec common1 = -delta1 % I1 % I2 % (riskset1 - I2) % exp_neg1 / denom1;
  arma::colvec common2 = -delta2 % I3 % I4 % (riskset2 - I4) % exp_neg2 / denom2;

  HessianWorker worker1(idxN1, common1, X1, X2, dim, true, batch_size);
  RcppParallel::parallelReduce(0, idxN1.n_elem, worker1);
  
  HessianWorker worker2(idxN2, common2, X1, X2, dim, false, batch_size);
  RcppParallel::parallelReduce(0, idxN2.n_elem, worker2);
  
  arma::mat H = worker1.local_H + worker2.local_H;

  return H;

}
