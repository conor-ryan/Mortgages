rm(list = ls())
library(data.table)
library(ggplot2)
library(fixest)
setwd("G:/Shared drives/Mortgages")

##### Read and Combine Data ####
firm_panel = NULL
tracts=NULL
for (year in 2007:2017){
print(year)
  df = fread("Data/Public HMDA/2022_public_lar_csv.csv")
# df = fread("Data/Public HMDA/hmda_2014_nationwide_all-records_labels.csv")
#### Filter
df = df[action_take==1]
df = df[loan_type_name=="Conventional"]
df = df[loan_purpose_name=="Home purchase"]
df = df[conforming_loan_limit=="c"&loan_amount_000s>=40 ]
df = df[credit_score>=620]
df = df[combined_loan_to_value_ratio >=30 & combined_loan_to_value_ratio <=95]
df = df[loan_term==360]

df[,sold:=as.numeric(purchaser_type>0)]
df[,count:=1]

firm_avg = df[,list(sold_value=sum(sold*conforming_amount)/sum(conforming_amount),
                    sold_loan=mean(sold),
                    total_loans=sum(count),
                    total_value=sum(loan_amount_000s),
                    total_conforming_value=sum(conforming_amount)),
   by=c("respondent_id","agency_code","as_of_year","state_abbr","county_code")]

### Tract Info ### 
tracts_temp = unique(df[,c("as_of_year","state_abbr","state_code","county_code","census_tract_number","population","minority_population","hud_median_family_income","tract_to_msamd_income","number_of_owner_occupied_units","number_of_1_to_4_family_units")])

tracts = rbind(tracts,tracts_temp)

firm = as.data.table(read.csv(paste("Data/Public HMDA/Institutions/hmda_",year,"_panel.csv",sep="")))

if (year<2010){
  firm[,c("Respondent.Address","Respondent.Zip.Code","Tax.ID"):=NULL]
  firm[,Top.Holder.RSSD.ID:=NA]
  firm[,Top.Holder.Name:=NA]
  firm[,Respondent.RSSD.ID:=NA]
  firm[,Parent.RSSD.ID:=NA]
  firm_names = names(firm)
}else{
  firm[,c("Top.Holder.City","Top.Holder.State","Top.Holder.Country","Respondent.FIPS.State.Number" ):=NULL]
  names(firm) = firm_names
}


firm = merge(firm,firm_avg,by.x=c("Respondent.Identification.Number","Agency.Code"),by.y=c("respondent_id","agency_code"),all.x=TRUE)

firm_panel = rbind(firm_panel,firm)
rm(df,firm)
gc()
}
