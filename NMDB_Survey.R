rm(list = ls())
library(data.table)
library(ggplot2)
setwd("G:/Shared drives/Mortgages")

#### Load Raw Data ####

load("Data/NMDB Survey Data/nsmo_v41_1320_puf.rdata")
data = nsmo_v41_1320_puf
rm(nsmo_v41_1320_puf)
data = as.data.table(data)

## Limit to Purchases 
data = data[x33==1]

#### General Variables ####
data[,monthDate:=open_month + (open_year-2013)*12]
data[,rate:=pmms+rate_spread]
data[,gse_sold:=as.numeric(gse!=-2)]

#### Search Variables ####
data[,consider2plus:=as.numeric(x11>=2)]
data[,apply2plus:=as.numeric(x12>=2)]

data[,consider3plus:=as.numeric(x11>=3)]
data[,apply3plus:=as.numeric(x12>=3)]

### Applications per origination
data[x12>1,mean(x12)] # With Cancelation
data[x13d!=1,mean(x12)] # Exclude turn-downs
data[x13d!=1&x13b!=1,mean(x12)] # Exclude turn-downs and credit concerns



#### Search Trends ####
df_plot = data[,list(consider2plus=sum(consider2plus*analysis_weight)/sum(analysis_weight),
                     apply2plus=sum(apply2plus*analysis_weight)/sum(analysis_weight),
                     consider3plus=sum(consider3plus*analysis_weight)/sum(analysis_weight),
                     apply3plus=sum(apply3plus*analysis_weight)/sum(analysis_weight),
                     prime_rate=sum(pmms*analysis_weight)/sum(analysis_weight)),by=c("open_month","open_year","monthDate")]

ggplot(df_plot) + aes(x=monthDate,y=prime_rate) + geom_line()
ggplot(df_plot) + 
  geom_line(aes(x=monthDate,y=consider2plus,color="consider2plus")) + 
  geom_line(aes(x=monthDate,y=apply2plus,color="apply2plus")) + 
  geom_line(aes(x=monthDate,y=consider3plus,color="consider3plus")) + 
  geom_line(aes(x=monthDate,y=apply3plus,color="apply3plus")) 


df_plot = data[,list(consider2plus=sum(consider2plus*analysis_weight)/sum(analysis_weight),
                     apply2plus=sum(apply2plus*analysis_weight)/sum(analysis_weight),
                     consider3plus=sum(consider3plus*analysis_weight)/sum(analysis_weight),
                     apply3plus=sum(apply3plus*analysis_weight)/sum(analysis_weight),
                     prime_rate=sum(pmms*analysis_weight)/sum(analysis_weight),
                     discount=sum(rate_spread*analysis_weight)/sum(analysis_weight)),by=c("open_month","open_year","monthDate","metro_lmi")]

ggplot(df_plot) + aes(x=monthDate,y=prime_rate,color=as.factor(metro_lmi)) + geom_line()
ggplot(df_plot) + aes(x=monthDate,y=discount,color=as.factor(metro_lmi)) + geom_line()
ggplot(df_plot) + aes(x=monthDate,y=consider2plus,color=as.factor(metro_lmi)) + geom_line()



df_plot = data[,list(consider2plus=sum(consider2plus*analysis_weight)/sum(analysis_weight),
                     apply2plus=sum(apply2plus*analysis_weight)/sum(analysis_weight),
                     consider3plus=sum(consider3plus*analysis_weight)/sum(analysis_weight),
                     apply3plus=sum(apply3plus*analysis_weight)/sum(analysis_weight),
                     prime_rate=sum(pmms*analysis_weight)/sum(analysis_weight),
                     discount=sum(rate_spread*analysis_weight)/sum(analysis_weight)),by=c("metro_lmi")]

res = glm(rate*100~as.factor(x11) + score_orig_r + as.factor(loan_amount_cat) + as.factor(jumbo) +  ltv + dti + 
                     as.factor(loan_type) + as.factor(monthDate) + as.factor(first_mort_r),data=data,weight=analysis_weight)
summary(res)

## Looking for better rate
res = glm(rate*100~as.factor(x11) + score_orig_r + as.factor(loan_amount_cat) + as.factor(jumbo) +  ltv + dti + 
            as.factor(loan_type) + as.factor(monthDate) + as.factor(first_mort_r),data=data,weight=analysis_weight)
summary(res)

## Not about credit risk
res = glm(rate*100~as.factor(x11) + score_orig_r + as.factor(loan_amount_cat) + as.factor(jumbo) +  ltv + dti + 
            as.factor(loan_type) + as.factor(monthDate) + as.factor(first_mort_r),data=data[x13d!=1&x13b!=1],weight=analysis_weight)
summary(res)

## search due to credit risk
res = glm(rate*100~as.factor(x11) + score_orig_r + as.factor(loan_amount_cat) + as.factor(jumbo) +  ltv + dti + 
            as.factor(loan_type) + as.factor(monthDate) + as.factor(first_mort_r),data=data[x13d==1&x13b==1],weight=analysis_weight)
summary(res)

## search by demographic
res = glm(x12~ score_orig_r + as.factor(loan_amount_cat) + as.factor(jumbo) +  ltv + dti + 
            as.factor(loan_type) + as.factor(open_year) + as.factor(first_mort_r),data=data[x13d!=1&x13b!=1],weight=analysis_weight)
summary(res)


## Loan Performance
data[,performance:=as.numeric(perf_status_0916%in%c("C","P","Q","S","A","K"))]
data[perf_status_0916%in%c("Q","S","A","K"),performance:=NA]
res = glm(rate*100~as.factor(x11) + score_orig_r + as.factor(loan_amount_cat) + as.factor(jumbo) + ltv + dti + 
            as.factor(loan_type) + as.factor(monthDate) + as.factor(first_mort_r),data=data[x13d!=1&x13b!=1],weight=analysis_weight)
summary(res)

res = glm(performance~gse_sold+ score_orig_r + ltv + dti + as.factor(monthDate),data=data,weight=analysis_weight)
summary(res)
