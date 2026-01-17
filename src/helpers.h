
# ifndef HELPERS_H
# define HELPERS_H

#include <RcppArmadillo.h>
#include <RcppParallel.h>

using namespace Rcpp;
using namespace RcppParallel;

// Thanks chatGPT!!
struct Fenwick {
  std::vector<int> tree;
  Fenwick(std::size_t n) : tree(n+1, 0) {}
  
  void add(std::size_t i);
  int sum(std::size_t i) const;
  
  // void add(std::size_t i) {
  //   for (++i; i < tree.size(); i += i & -i)
  //     tree[i]++;
  // }
  // 
  // int sum(std::size_t i) const {
  //   int s = 0;
  //   for (++i; i > 0; i -= i & -i)
  //     s += tree[i];
  //   return s;
  // }
};

struct riskset_worker : public RcppParallel::Worker {
  
  RVector<double> x;
  std::vector<std::size_t> y_rank;
  std::vector<std::size_t> x_ord;
  RMatrix<int> N;
  std::size_t n;
  
  riskset_worker(NumericVector& x,
                 const std::vector<std::size_t>& y_rank,
                 const std::vector<std::size_t>& x_ord,
                 IntegerMatrix& N)
    : x(x), y_rank(y_rank), x_ord(x_ord), N(N), n(x.size()) {}
  
  
  void operator() (std::size_t begin, std::size_t end);

  // void operator() (std::size_t begin, std::size_t end) {
  // 
  //   Fenwick fw(n);
  //   std::size_t k = 0;
  // 
  //   for (std::size_t jj = begin; jj < end; jj++) {
  //   
  //     std::size_t j = x_ord[jj];
  //   
  //     while (k < n && x[x_ord[k]] >= x[j]) {
  //       fw.add(y_rank[x_ord[k]]);
  //       k++;
  //     }
  //   
  //     for (std::size_t i = 0; i < n; i++) {
  //       N(j, i) = fw.sum(n - 1) - fw.sum(y_rank[i] - 1);
  //     }
  //   }
  // }
};

// R functions
arma::mat row_kron(const arma::mat& X, const arma::mat& Y);

IntegerMatrix indgreater(NumericVector x);

IntegerMatrix indless(NumericVector x);

IntegerMatrix indequal(NumericVector x);

IntegerMatrix riskset_fast(NumericVector x, NumericVector y);

IntegerMatrix delta(NumericVector x, NumericVector y);


# endif