#include <RcppArmadillo.h>
#include <RcppParallel.h>

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]


// TODO - Add more comments to the code, especially in the copulaWorker class
struct copulaWorker : public RcppParallel::Worker {
  
  // Input
  const arma::vec& Xtilde;
  const arma::vec& Ytilde;
  const arma::vec& inv1_vec;
  const arma::vec& inv2_vec;
  const arma::vec& kmc_times;
  const arma::vec& kmc_survs;
  
  // Constants
  double max_c;
  int n; // Number of observations
  int m; // Grid size
  
  // Output
  arma::mat& copula;

  // Constructor
  copulaWorker(const arma::vec& X, const arma::vec& Y,
             const arma::vec& i1, const arma::vec& i2,
             const arma::vec& kct, const arma::vec& kcs,
             double max_c, arma::mat& out, int m_grid)
    : Xtilde(X), Ytilde(Y), inv1_vec(i1), inv2_vec(i2),
      kmc_times(kct), kmc_survs(kcs), max_c(max_c),
      n(X.n_elem), m(m_grid), copula(out) {}

  /**
   * Helper: Kaplan-Meier Survival Lookup
   * Performs an O(log N) binary search to find S(t).
   * Assumes kmc_times is sorted ascending.
   */
  double get_km_surv(double t) const {
    if (t > max_c) return 0.0;
    if (kmc_times.is_empty() || t < kmc_times[0]) return 1.0;

    // Find the first element strictly greater than t
    auto it = std::upper_bound(kmc_times.begin(), kmc_times.end(), t);
    
    // Step back one to get the survival value at or just before t
    int idx = std::distance(kmc_times.begin(), it) - 1;
    return kmc_survs[idx];
  }

  // Parallel Execution Logic
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t k = begin; k < end; ++k) {
      double t1 = inv1_vec[k];
      
      for (int l = 0; l <= m; ++l) {
        double t2 = inv2_vec[l];
        double t_max = std::max(t1, t2);

        // Count how many observations satisfy X > t1 and Y > t2.
        int num = arma::accu((Xtilde > t1) && (Ytilde > t2));

        // Denominator lookup
        double denum = get_km_surv(t_max);

        // Result calculation with safety check for zero denominator
        if (denum > 1e-10) {
          copula(k, l) = (double)num / (double(n) * denum);
        } else {
          copula(k, l) = 0.0; 
        }
      }
    }
  }
};

struct BernsteinWorker : public RcppParallel::Worker {

  // Input
  const arma::vec& s1_vec;
  const arma::vec& s2_vec;
  const arma::mat& copula;
  const int m;
  const double tau1;
  const double tau2;
  
  // Output
  arma::vec& results;

  BernsteinWorker(const arma::vec& s1, const arma::vec& s2, const arma::mat& cop,
                  int m_val, double t1, double t2, arma::vec& res)
    : s1_vec(s1), s2_vec(s2), copula(cop), m(m_val), tau1(t1), tau2(t2), results(res) {}

  // log-binomial density
  double log_dbinom(int k, int n, double p) const {
    if (p <= 0) return (k == 0) ? 0.0 : -1e100;
    if (p >= 1) return (k == n) ? 0.0 : -1e100;
    return R::lchoose(n, k) + k * std::log(p) + (n - k) * std::log(1.0 - p);
  }

  void operator()(std::size_t begin, std::size_t end) {
    arma::vec p1(m + 1);
    arma::vec p2(m + 1);
    arma::vec p1_acc(m + 1);
    arma::vec p2_acc(m + 1);
    double eps = 1e-10;

    for (std::size_t i = begin; i < end; ++i) {
      // Calculate weights
      // Note that we clamp w1 and w2 to avoid issues with log(0) in the Bernstein
      // basis calculations.
      double w1 = std::clamp((s1_vec[i] - tau1) / (1.0 - tau1), eps, 1.0 - eps);
      double w2 = std::clamp((s2_vec[i] - tau2) / (1.0 - tau2), eps, 1.0 - eps);

      double inv_w1 = 1.0 / (w1 * (1.0 - w1));
      double inv_w2 = 1.0 / (w2 * (1.0 - w2));

      for (int k = 0; k <= m; ++k) {
        double b1 = std::exp(log_dbinom(k, m, w1));
        double b2 = std::exp(log_dbinom(k, m, w2));

        // Bernstein basis
        p1(k) = b1;
        p2(k) = b2;
        
        // Derivatives of Bernstein basis
        p1_acc(k) = b1 * (k - m * w1) * inv_w1;
        p2_acc(k) = b2 * (k - m * w2) * inv_w2;
      }


      // Use matrix products instead of the original sums in Ömer's R code
      // sum1 = p1.t() * copula * p2
      // sum2 = p1_acc.t() * copula * p2
      // sum3 = p1.t() * copula * p2_acc
      // sum4 = p1_acc.t() * Chat * p2_acc

      // Pre-calculating Chat * p simplifies each matrix product from O(m^2) to O(m)
      arma::vec cop_p2 = copula * p2;
      arma::vec cop_p2_acc = copula * p2_acc;

      // Inner products: O(m)
      double sum1 = arma::dot(p1, cop_p2);
      double sum2 = arma::dot(p1_acc, cop_p2);
      double sum3 = arma::dot(p1, cop_p2_acc);
      double sum4 = arma::dot(p1_acc, cop_p2_acc);

      // log(CRF)
      results[i] = std::log(sum1) + std::log(sum4) - std::log(sum2) - std::log(sum3);
    }
  }
};

// Log of binomial density to avoid interfacing with R's dbinom
double log_dbinom(int k, int n, double p) {
  if (p <= 0)
    return (k == 0) ? 0.0 : -INFINITY;
  if (p >= 1)
    return (k == n) ? 0.0 : -INFINITY;
  return std::lgamma(n + 1) - std::lgamma(k + 1) - std::lgamma(n - k + 1) +
         k * std::log(p) + (n - k) * std::log(1.0 - p);
}


/**
 * estimate_copula
 * The main R-accessible function to compute the copula matrix.
 * * @param data    Matrix containing t1, t2, and status vectors (biSurv object)
 * @param km1s    KM-estimator S(t1) as a list with time and surv vectors(survfit object)
 * @param km2s    KM-estimator S(t2) as a list with time and surv vectors(survfit object)
 * @param kmc     KM-estimator S(status) as a list with time and surv vectors(survfit object)
 * @param m       Bernstein orders
 * @param tau1    min(S(t1))
 * @param tau2    min(S(t2))
 */
// [[Rcpp::export]]
arma::mat estimate_copula(arma::vec &t1,
                                  arma::vec &t2,
                                  double max_c,
                                  Rcpp::List &km1, // KM-estimator S(t1)
                                  Rcpp::List &km2, // KM-estimator S(t2)
                                  Rcpp::List &kmc, // KM-estimator S(status)
                                  int m, // Bernstein order
                                  double tau1, // min(S(t1))
                                  double tau2) { // min(S(t1))
  

  arma::vec k1t = km1["time"], k1s = km1["surv"];
  arma::vec k2t = km2["time"], k2s = km2["surv"];
  arma::vec kct = kmc["time"], kcs = kmc["surv"];

  // Vector of times where S(t1) <= fk and S(t2) <= fl for the grid points fk and fl.
  arma::vec inv1(m + 1);
  arma::vec inv2(m + 1);

  for (int i = 0; i <= m; ++i) {
    double fk = tau1 + (1.0 - tau1) * (double(i) / m);
    double fl = tau2 + (1.0 - tau2) * (double(i) / m);

    // Find the first element for which S(t_1) <= fk
    // Since S(t_1) is decreasing, we use std::greater
    auto it1 = std::lower_bound(k1s.begin(), k1s.end(), fk, std::greater<double>());
    inv1[i] = (it1 == k1s.end()) ? k1t.max() : k1t[std::distance(k1s.begin(), it1)];
    // Find the first element for which S(t_1) <= fk
    // Since S(t_1) is decreasing, we use std::greater
    auto it2 = std::lower_bound(k2s.begin(), k2s.end(), fl, std::greater<double>());
    inv2[i] = (it2 == k2s.end()) ? k2t.max() : k2t[std::distance(k2s.begin(), it2)];
  }

  // Initialize output matrix
  arma::mat copula(m + 1, m + 1);

  // Initialize and run the parallel worker
  copulaWorker worker(t1, t2, inv1, inv2, kct, kcs, max_c, copula, m);
  
  // Parallelize across rows of the grid
  RcppParallel::parallelFor(0, m + 1, worker);

  return copula;
}

double estimate_bernstein(const double &s1, const double &s2, const int &m,
                          const double &tau1, const double &tau2,
                          const arma::mat &copula) {

  // s1, s2: estimated survival functions in t1 and t2 (eg. from Kaplan-Meier)
  // m: Bernstein order
  // tau1, tau2: min(s1) and min(s2) (ie. the largest observed time point)
  // Chat: estimated copula function

  // Calculate weights
  // Note that we clamp w1 and w2 to avoid issues with log(0) in the Bernstein
  // basis calculations.
  double eps = 1e-10;
  double w1 = std::clamp((s1 - tau1) / (1.0 - tau1), eps, 1.0 - eps);
  double w2 = std::clamp((s2 - tau2) / (1.0 - tau2), eps, 1.0 - eps);

  // double w1 = (s1 - tau1) / (1.0 - tau1);
  // double w2 = (s2 - tau2) / (1.0 - tau2);

  // Bernstein basis
  arma::vec p1(m + 1);
  arma::vec p2(m + 1);

  // Derivatives of Bernstein basis
  arma::vec p1_acc(m + 1);
  arma::vec p2_acc(m + 1);

  // Pre-calculate constants to avoid repeated division
  double inv_w1 = (1.0 / (w1 * (1.0 - w1)));
  double inv_w2 = (1.0 / (w2 * (1.0 - w2)));

  for (int k = 0; k <= m; ++k) {
    // Original R: choose(m, k) * w^k * (1-w)^(m-k)
    // This is just dbinom(k,m,w) in R
    // double b1 = R::dbinom(k, m, w1, false);
    // double b2 = R::dbinom(k, m, w2, false);
    double b1 = std::exp(log_dbinom(k, m, w1));
    double b2 = std::exp(log_dbinom(k, m, w2));

    p1(k) = b1;
    p2(k) = b2;
    p1_acc(k) = b1 * (k - m * w1) * inv_w1;
    p2_acc(k) = b2 * (k - m * w2) * inv_w2;
  }

  // Use matrix products instead of the original sums in Ömer's R code
  // sum1 = p1.t() * copula * p2
  // sum2 = p1_acc.t() * copula * p2
  // sum3 = p1.t() * copula * p2_acc
  // sum4 = p1_acc.t() * Chat * p2_acc

  // Pre-calculating Chat * p simplifies each matrix product from O(m^2) to O(m)
  arma::vec copula_p2 = copula * p2;
  arma::vec copula_p2_acc = copula * p2_acc;

  double sum1 = arma::as_scalar(p1.t() * copula_p2);
  double sum2 = arma::as_scalar(p1_acc.t() * copula_p2);
  double sum3 = arma::as_scalar(p1.t() * copula_p2_acc);
  double sum4 = arma::as_scalar(p1_acc.t() * copula_p2_acc);

  // return (sum1 * sum4) / (sum2 * sum3);
  return std::log(sum1) + std::log(sum4) - std::log(sum2) - std::log(sum3);
}


// [[Rcpp::export]]
arma::vec estimate_bernstein_vec(const arma::vec& s1, const arma::vec& s2, 
                                 int m, double tau1, double tau2, 
                                 const arma::mat& copula) {
  int n = s1.n_elem;
  arma::vec results(n);

  BernsteinWorker worker(s1, s2, copula, m, tau1, tau2, results);
  RcppParallel::parallelFor(0, n, worker);

  return results;
}
