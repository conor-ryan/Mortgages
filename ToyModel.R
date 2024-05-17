rm(list=ls())
library(data.table)


##### DGP #####
N = 1000
J = 3
phi = 0.6
r_f = 0.03

## Product Segments:
LTV = c(0.8,0.85,0.9,0.95)
lambda = c(0,0.01,0.05,0.1)

products = data.table(LTV = rep(LTV,length(lambda)),
                      lambda = rep(lambda,each=length(LTV)),
                      merge=1)

products[,gse_cost:=LTV*.25 + lambda*0.75]


banks = data.table(bank= rep(1:J,N),
                   bank_cost = rep(c(0.2,.5,.1),N),
                   merge=1)

data = merge(products,banks,allow.cartesian=TRUE,by="merge")
data[,r:= runif(nrow(data),min=0.03,max=0.11)]

data[,return_to_hold:=r/(r_f + lambda) + lambda/(r_f+lambda)*(phi/LTV)]
data[,return_to_sell:=r/r_f]
data[,sell_diff:=(r/r_f - phi/LTV)*lambda/(r_f+lambda)]
data[,idio_cost:=rnorm(nrow(data),sd=0.01)]

data[,sell:=as.numeric(sell_diff>gse_cost+bank_cost+idio_cost)]
summary(data)


data = data.table(LTV = 0.8,
                  r = rnorm(N*J,mean=.08,sd=0.02),
                  phi = 0.4,
                  lambda = runif(N*J,min = 0.0,max=0.2),
                  r_f = 0.01,
                  bank = rep(1:J,N),
                  bank_rel_cost = rep(c(5,3,1),N),
                  idio_cost = rnorm(N*J,sd=0.05))
data[,sell:=as.numeric( (phi/LTV - r/r_f)*(lambda)/(r_f+lambda) > (-(bank_rel_cost+idio_cost)) )] 

data[,return_diff:=(phi/LTV-r/r_f)]
summary(data)


#### Estimate ######
data_est = as.data.table(as.data.frame(data))
data_est[,bank_rel_cost:=NULL]
data_est[,idio_cost:=NULL]
data_est[,phi:=NULL]
data_est[,r_f:=NULL]

moment_ineq <- function(par){
  # par = c(0.4,x,6,5,4)
  data_est[,phi_est:=par[1]]
  data_est[,r_f_est:=0.01]
  data_est[,bank_rel_est:=rep(par[2:4],N)]
  data_est[,sell_pred:=as.numeric( (phi_est/LTV - r/r_f_est)*(lambda)/(r_f_est+lambda) > (-(bank_rel_est))    )]
  moments = data_est[,list(res = mean(abs(sell-sell_pred)))]
  data_est[,c("bank_rel_est","sell_pred","phi_est","r_f_est"):=NULL]
  return(moments[,res])
}

# range = seq(0.001,0.5,by=0.001)
# val = lapply(range,moment_ineq)
# plot(range,val)



res = optim(c(0.5,5,5,5),moment_ineq)

print(res$par)





