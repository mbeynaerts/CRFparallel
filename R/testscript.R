
library(RcppParallel)

RcppParallel::setThreadOptions(numThreads = RcppParallel::defaultNumThreads()-1)

{
mx <- copula::mixCopula(list(copula::claytonCopula(5, dim = 2),
                             copula::frankCopula(7, dim = 2),
                             copula::gumbelCopula(5, dim = 2)),
                        w = c(0,1,0))

U <- copula::rCopula(1000, mx)

T1 <- -log(U[,1])
T2 <- -log(U[,2])

C1 <- rexp(1000, 0.094)
C2 <- rexp(1000, 0.094)

X1 <- pmin(T1,C1)
X2 <- pmin(T2,C2)

X <- as.matrix(cbind(X1,X2))


N1 <- CRFparallel::risksetC(X[,1],X[,2])
N2 <- CRFpackage::risksetC(X[,1],X[,2])

}

N1[1:10,1:10]
N2[1:10,1:10]


sum(N1)
sum(N2)
