#include <RcppArmadillo.h>
#include <RcppParallel.h>

using namespace Rcpp;
using namespace RcppParallel;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]

// Thanks chatGPT!!
struct Fenwick {
  std::vector<int> tree;
  Fenwick(std::size_t n) : tree(n+1, 0) {}
  
  void add(std::size_t i) {
    for (++i; i < tree.size(); i += i & -i)
      tree[i]++;
  }

  int sum(std::size_t i) const {
    int s = 0;
    for (++i; i > 0; i -= i & -i)
      s += tree[i];
    return s;
  }
};

struct riskset_worker : public RcppParallel::Worker {
  
  RVector<double> x;
  std::vector<std::size_t> y_rank;
  std::vector<std::size_t> x_ord;
  RMatrix<int> N;
  std::size_t n;

  riskset_worker(NumericVector x,
                 const std::vector<std::size_t>& y_rank,
                 const std::vector<std::size_t>& x_ord,
                 IntegerMatrix N)
    : x(x), y_rank(y_rank), x_ord(x_ord), N(N), n(x.size()) {}

  void operator() (std::size_t begin, std::size_t end) {
  
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
};


struct HessianWorker : public Worker {
  
  const arma::mat& X1;
  const arma::mat& X2;
  const arma::uvec& idxN1;
  const arma::uvec& idxN2;
  
  arma::mat& D1;
  arma::mat& D2;

  int dim;
  int nrow1;
  int nrow2;

  HessianWorker(const arma::mat& X1_,
              const arma::mat& X2_,
              const arma::uvec& idxN1_,
              const arma::uvec& idxN2_,
              arma::mat& D1_,
              arma::mat& D2_)
    : X1(X1_), X2(X2_),
      idxN1(idxN1_), idxN2(idxN2_),
      D1(D1_), D2(D2_) {
  
    dim   = X1.n_cols;
    nrow1 = X1.n_rows;
    nrow2 = X2.n_rows;
  }

  inline void fill_deriv(arma::colvec& out,
                         const arma::colvec& a,
                         const arma::colvec& b,
                         const arma::uvec& idx,
                         bool transpose) {
  
    for (arma::uword k = 0; k < idx.n_elem; ++k) {
      arma::uword pos = idx[k];
      arma::uword r, c;
    
      if (transpose) {
        r = pos % nrow2;
        c = pos / nrow2;
        out[k] = a[c] * b[r];
      } else {
        r = pos % nrow1;
        c = pos / nrow1;
        out[k] = a[r] * b[c];
      }
    }
  }

  void operator()(std::size_t begin, std::size_t end) {
    arma::colvec tmp1(idxN1.n_elem);
    arma::colvec tmp2(idxN2.n_elem);
  
    for (std::size_t m = begin; m < end; ++m) {
      int i = m % dim;
      int j = m / dim;
    
      fill_deriv(tmp1, X1.col(i), X2.col(j), idxN1, true);
      fill_deriv(tmp2, X1.col(i), X2.col(j), idxN2, false);
    
      D1.col(m) = tmp1;
      D2.col(m) = tmp2;
     }
  }
};



// Slow parallel implementation
// struct riskset : public Worker {
//   
//   // Input
//   RVector<double> x;
//   RVector<double> y;
//   std::size_t n;
//   
//   // Output
//   RMatrix<int> N;
//   
//   // Helper function
//   int Ind2(const double &a, const double &b) {
//     int sum = 0;
//     for (std::size_t i=0; i<n; i++) {
//       if (x[i] >= a and y[i] >= b) {
//         sum++;
//       }
//     }
//     return sum;
//   }
//   
//   // Worker
//   riskset(const NumericVector x, const NumericVector y, IntegerMatrix N)
//     : x(x), y(y), n(x.size()), N(N) {}
//   
//   // Parallel loop
//   void operator() (std::size_t begin, std::size_t end) {
//     for (std::size_t j=begin; j<end; j++) {
//       for (std::size_t i=0; i<n; i++) {
//         N(j,i) = Ind2(x[j], y[i]);
//       }
//     }
//   }
//   
// };

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
IntegerMatrix IndGreater(NumericVector &x) {
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
IntegerMatrix IndLess(NumericVector &x) {
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
IntegerMatrix IndEqual(NumericVector &x) {
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
  IntegerMatrix N(n, n);

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
IntegerMatrix DeltaC(NumericVector &x, NumericVector &y) {
  int n = x.size();
  IntegerMatrix delta(n);
  for (int j=0; j<n; j++) {
    for (int i=0; i<n; i++) {
      delta (j,i) = x[j]*y[i];
    }
  }
  return delta;
}



// [[Rcpp::export]]
double logLikC(const NumericVector &riskset1,
               const NumericVector &riskset2,
               const NumericVector &logtheta1,
               const NumericVector &logtheta2,
               const NumericVector &delta1,
               const NumericVector &delta2,
               const NumericVector &I1,
               const NumericVector &I2,
               const NumericVector &I3,
               const NumericVector &I4,
               const NumericVector &I5,
               const NumericVector &I6) {

  double sum1;
  double sum2;

  sum1 = Rcpp::sum(delta1*I1*(logtheta1*I5 - Rcpp::log(riskset1 + I2*Rcpp::exp(logtheta1) - I2)));
  sum2 = Rcpp::sum(delta2*I3*(logtheta2*I6 - Rcpp::log(riskset2 + I4*Rcpp::exp(logtheta2) - I4)));

  return(-sum1-sum2);
}

// [[Rcpp::export]]
NumericVector gradientNew(const arma::colvec &riskset1,
                          const arma::colvec &riskset2,
                          const arma::colvec &logtheta1,
                          const arma::colvec &logtheta2,
                          const arma::colvec &delta1,
                          const arma::colvec &delta2,
                          const arma::colvec &I1,
                          const arma::colvec &I2,
                          const arma::colvec &I3,
                          const arma::colvec &I4,
                          const arma::colvec &I5,
                          const arma::colvec &I6,
                          const arma::mat &X1,
                          const arma::mat &X2,
                          const arma::uvec &idxN1,
                          const arma::uvec &idxN2) {

  int K = X1.n_rows;
  int n = riskset1.size();
  int dim = X1.n_cols;
  int totalparam = pow(dim, 2);
  NumericVector result(totalparam);
  arma::colvec common1(n);
  arma::colvec common2(n);
  
  arma::vec exp_neg1 = arma::exp(-logtheta1);
  arma::vec exp_neg2 = arma::exp(-logtheta2);
  
  common1 = delta1 % I1 % ( I5 - I2 / ( (riskset1 - I2) % exp_neg1 + I2 ) );
  common2 = delta2 % I3 % ( I6 - I4 / ( (riskset2 - I4) % exp_neg2 + I4 ) );

  arma::mat deriv_mat(K,K);
  arma::mat deriv_mat_t(K,K);
  arma::colvec deriv1(deriv_mat_t.memptr(), deriv_mat_t.n_elem, false, true);
  arma::colvec deriv2(deriv_mat.memptr(), deriv_mat.n_elem, false, true);

  for (int m=0; m<totalparam; m++) {

    int idx1 = m % dim;
    int idx2 = m / dim;

    deriv_mat = arma::kron(X1.col(idx1), X2.col(idx2).as_row());
    deriv_mat_t = deriv_mat.t();

    double sum1 = arma::accu(common1 % deriv1.elem(idxN1));
    double sum2 = arma::accu(common2 % deriv2.elem(idxN2));

    result(m) = -sum1-sum2;

  }

  return(result);

}

// [[Rcpp::export]]
NumericVector gradientPoly(const NumericVector &riskset1,
                           const NumericVector &riskset2,
                           const NumericVector &logtheta1,
                           const NumericVector &logtheta2,
                           const Rcpp::List &deriv,
                           const int &df,
                           const NumericVector &delta1,
                           const NumericVector &delta2,
                           const NumericVector &I1,
                           const NumericVector &I2,
                           const NumericVector &I3,
                           const NumericVector &I4,
                           const NumericVector &I5,
                           const NumericVector &I6) {

  int n = riskset1.length();
  NumericVector result(df);
  NumericVector common1(n);
  NumericVector common2(n);

  /* Transform list of derivative matrices into vector of matrices */
  std::vector<NumericMatrix> deriv_vec(df);
  for(int k = 0; k < df; ++k) {
    NumericMatrix deriv_R = deriv[k];
    /* arma::mat derivMat(deriv_R.begin(), deriv_R.rows(), deriv_R.cols(), false, true);
     deriv_vec[k] = derivMat; */
    deriv_vec[k] = deriv_R;
  }

  common1 = delta1*I1*(I5 - I2*Rcpp::exp(logtheta1)/(riskset1 + I2*Rcpp::exp(logtheta1) - I2));
  common2 = delta2*I3*(I6 - I4*Rcpp::exp(logtheta2)/(riskset2 + I4*Rcpp::exp(logtheta2) - I4));

  for (int m=0; m<df; m++) {

    double sum1 = 0.0;
    double sum2 = 0.0;

    /* Calculation of L1 */
    for (int j=0; j<n; j++) {
      sum1 += common1(j)*deriv_vec[m](j,0);
      sum2 += common2(j)*deriv_vec[m](j,1);
    }

    result(m) = -sum1-sum2;

  }

  return(result);

}

// [[Rcpp::export]]
arma::mat hessianNew(const arma::colvec& riskset1,
                     const arma::colvec &riskset2,
                     const arma::colvec &logtheta1,
                     const arma::colvec &logtheta2,
                     const arma::colvec &delta1,
                     const arma::colvec &delta2,
                     const arma::colvec &I1,
                     const arma::colvec &I2,
                     const arma::colvec &I3,
                     const arma::colvec &I4,
                     const arma::mat& X1,
                     const arma::mat& X2,
                     const arma::uvec& idxN1,
                     const arma::uvec& idxN2) {

  int K = X1.n_rows;
  int n = riskset1.n_rows;
  int dim = X1.n_cols;
  int totalparam = dim*dim;
  arma::colvec common1(n);
  arma::colvec common2(n);

  arma::mat result(totalparam, totalparam, arma::fill::zeros);
  
  arma::colvec exp_neg1 = arma::exp(-logtheta1);
  arma::colvec exp_neg2 = arma::exp(-logtheta2);
  
  arma::colvec denom1 = arma::square((riskset1 - I2) % exp_neg1 + I2);
  arma::colvec denom2 = arma::square((riskset2 - I4) % exp_neg2 + I4);
  
  const double eps = 1e-12;
  denom1.elem(find(denom1 < eps)).fill(eps);
  denom2.elem(find(denom2 < eps)).fill(eps);
  
  common1 = -delta1 % I1 % I2 % (riskset1 - I2) % exp_neg1 / denom1;
  common2 = -delta2 % I3 % I4 % (riskset2 - I4) % exp_neg2 / denom2;
  
  if(arma::all(common1 == 0) && arma::all(common2 == 0)) 
    return result; 
  
  // if (arma::any(denom1 == 0))
  //   Rcpp::stop("Zero denominator in common1");
  // 
  // if (common1.has_nan())
  //   Rcpp::stop("NaN in common1");
    
  arma::mat deriv_mat(K,K), deriv_mat_t(K,K), deriv_mat_l(K,K), deriv_mat_l_t(K,K);
  arma::colvec deriv1m, deriv2m, deriv1l, deriv2l;
  // 
  // arma::colvec deriv1m(deriv_mat_t.memptr(), deriv_mat_t.n_elem, false, true);
  // arma::colvec deriv2m(deriv_mat.memptr(), deriv_mat.n_elem, false, true);
  // arma::colvec deriv1l(deriv_mat_l_t.memptr(), deriv_mat_l_t.n_elem, false, true);
  // arma::colvec deriv2l(deriv_mat_l.memptr(), deriv_mat_l.n_elem, false, true);
  
  

  for (int m = 0; m < totalparam; m++) {

    int idx1m = m % dim;
    int idx2m = m / dim;
    
    deriv_mat = arma::kron(X1.col(idx1m), X2.col(idx2m).as_row());
    deriv_mat_t = deriv_mat.t();
    
    deriv2m = arma::vectorise(deriv_mat);
    deriv1m = arma::vectorise(deriv_mat_t);
    
    if (idxN1.max() >= deriv1m.n_elem)
      Rcpp::stop("idxN1 out of bounds");
    
    if (idxN2.max() >= deriv2m.n_elem)
      Rcpp::stop("idxN2 out of bounds");

    for (int l = m; l < totalparam; l++) {

      int idx1l = l % dim;
      int idx2l = l / dim;

      deriv_mat_l = arma::kron(X1.col(idx1l), X2.col(idx2l).as_row());
      deriv_mat_l_t = deriv_mat_l.t();
      
      deriv2l = arma::vectorise(deriv_mat_l);
      deriv1l = arma::vectorise(deriv_mat_l_t);
      
      if (!deriv1m.is_finite() || !deriv1l.is_finite())
        Rcpp::stop("Non-finite derivative vectors");

      double sum1 = accu(common1 % deriv1m.elem(idxN1) % deriv1l.elem(idxN1));
      double sum2 = accu(common2 % deriv2m.elem(idxN2) % deriv2l.elem(idxN2));

      result(l,m) = -sum1-sum2;

    }
  }

  return arma::symmatl(result);

}

// [[Rcpp::export]]
arma::mat hessian_fast(
    const arma::colvec& riskset1,
    const arma::colvec& riskset2,
    const arma::colvec& logtheta1,
    const arma::colvec& logtheta2,
    const arma::colvec& delta1,
    const arma::colvec& delta2,
    const arma::colvec& I1,
    const arma::colvec& I2,
    const arma::colvec& I3,
    const arma::colvec& I4,
    const arma::mat& X1,
    const arma::mat& X2,
    const arma::uvec& idxN1,
    const arma::uvec& idxN2) {
  
  int dim = X1.n_cols;
  int totalparam = dim * dim;

  arma::colvec exp_neg1 = arma::exp(-logtheta1);
  arma::colvec exp_neg2 = arma::exp(-logtheta2);

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
    
      
      return arma::symmatl(H);
}


// [[Rcpp::export]]
NumericMatrix hessianPolyC(const NumericVector &riskset1,
                           const NumericVector &riskset2,
                           const NumericVector &logtheta1,
                           const NumericVector &logtheta2,
                           const Rcpp::List &deriv,
                           const int &df,
                           const NumericVector &delta1,
                           const NumericVector &delta2,
                           const NumericVector &I1,
                           const NumericVector &I2,
                           const NumericVector &I3,
                           const NumericVector &I4) {

  int n = riskset1.length();
  NumericVector common1(n);
  NumericVector common2(n);

  NumericMatrix result(df);

  /* Transform list of derivative matrices into vector of matrices */
  std::vector<NumericMatrix> deriv_vec(df);
  for(int k = 0; k < df; ++k) {
    NumericMatrix deriv_R = deriv[k];
    /* arma::mat derivMat(deriv_R.begin(), deriv_R.rows(), deriv_R.cols(), false, true);
    deriv_vec[k] = derivMat; */
    deriv_vec[k] = deriv_R;
  }

  common1 = -delta1*I1*(riskset1 - I2)*I2*Rcpp::exp(logtheta1)/Rcpp::pow(riskset1 - I2 + I2*Rcpp::exp(logtheta1),2);
  common2 = -delta2*I3*(riskset2 - I4)*I4*Rcpp::exp(logtheta2)/Rcpp::pow(riskset2 - I4 + I4*Rcpp::exp(logtheta2),2);

  for (int m = 0; m < df; m++) {
    for (int l = m; l < df; l++) {

      double sum1 = 0.0;
      double sum2 = 0.0;

      for (int j=0; j<n; j++) {
        sum1 += common1(j)*deriv_vec[m](j,0)*deriv_vec[l](j,0);
        sum2 += common2(j)*deriv_vec[m](j,1)*deriv_vec[l](j,1);
      }

      result(m,l) = -sum1-sum2;
      result(l,m) = result(m,l);

    }
  }

  return(result);

}

NumericMatrix testfunct(const List &deriv,
                        const int &df) {

  int totalparam = df*df;

  std::vector<NumericMatrix> deriv_vec(totalparam);
  for(int k = 0; k < totalparam; ++k) {
    NumericMatrix deriv_R = deriv[k];
    /* arma::mat derivMat(deriv_R.begin(), deriv_R.rows(), deriv_R.cols(), false, true);
    deriv_vec[k] = derivMat; */
    deriv_vec[k] = deriv_R;
  }
  NumericMatrix result = deriv_vec[0];
  return(result);
}

