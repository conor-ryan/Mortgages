rm(list=ls())
set.seed(2)

LTV = c(0.05,0.1,0.15,0.2)
Term = c(0,1)
Adj = c(0,1) 

X = as.matrix(data.frame(LTV = rep(LTV,4),Term=rep(Term,each=8),Adj=rep(Adj,rep=2,each=4)))


sigma = rnorm(3)
beta = rnorm(3)

r = 0.5

X%*%beta - r
X%*%sigma


u = -5 


r_u =  X%*%sigma - u

r_u - X%*%beta

which(X%*%beta==min(X%*%beta))
which((r_u-X%*%beta)==max(r_u-X%*%beta))

pi = 0.01
r = 0.05
ffr = seq(0.01,0.045,length=100)
lambda = seq(0,0.05,length=100)

r = pi + (ffr + lambda[50])
r_2 = pi*(ffr+lambda[50])

plot(ffr,r)
plot(ffr,r_2)

plot(ffr,r-ffr)



