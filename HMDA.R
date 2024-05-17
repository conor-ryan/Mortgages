rm(list = ls())
library(data.table)
library(ggplot2)
library(fixest)
setwd("G:/Shared drives/Mortgages")

df = fread("Data/Public HMDA/2022_public_lar_csv.csv")

### Test Dropped Applications
df[,dropped:=action_taken%in%c(2,4)]
df[,dropped_denial:=action_taken%in%c(2,3,4)]
df[,originated:=action_taken==1]

df[,sum(originated+dropped)]/df[,sum(originated)]
df[,sum(originated + dropped_denial)]/df[,sum(originated)]



##### Read and Combine Data ####
firm_panel = NULL
tracts=NULL
for (year in 2007:2017){
print(year)
df = fread(paste("Data/Public HMDA/hmda_",year,"_nationwide_first-lien-owner-occupied-1-4-family-records_labels.csv",sep=""))
# df = fread("Data/Public HMDA/hmda_2014_nationwide_all-records_labels.csv")
#### Filter

df = df[loan_type_name=="Conventional"]
df = df[loan_purpose_name=="Home purchase"]

df[,conforming_amount:=loan_amount_000s*as.numeric(loan_amount_000s<500)]
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

save(firm_panel,file="firm_panel.rData")

load("firm_panel.rData")

#### State Entry Patterns ####
firm_panel = firm_panel[!is.na(total_loans)]
firm_panel = firm_panel[!is.na(county_code)]
firm_panel = firm_panel[!state_abbr%in%c("VI","","PR")]


firm_panel[,count:=1]
firm_panel[,banks_per_year:=sum(count),by=c("Activity.Year","state_abbr","county_code")]

firm_panel[,mortgage_per_firm:=sum(total_loans,na.rm=TRUE),by=c("Respondent.Identification.Number","Activity.Year","Agency.Code")]

firm_panel[,lending_years:=sum(count),by=c("Respondent.Identification.Number","state_abbr","county_code","Agency.Code")]
firm_panel[,operating_years:=1 + max(as_of_year) - min(as_of_year),by=c("Respondent.Identification.Number","state_abbr","county_code","Agency.Code")]
firm_panel[,entry_year:=min(as_of_year),by=c("Respondent.Identification.Number","state_abbr","county_code","Agency.Code")]
firm_panel[,exit_year:=max(as_of_year),by=c("Respondent.Identification.Number","state_abbr","county_code","Agency.Code")]


firm_panel[,most_years:=max(operating_years),by=c("Respondent.Identification.Number","Agency.Code")]
firm_panel[,least_years:=min(operating_years),by=c("Respondent.Identification.Number","Agency.Code")]

firm_panel[,operating_counties:=sum(count),by=c("Respondent.Identification.Number","Activity.Year","Agency.Code")]

firm_panel[total_loans>10,hist(sold_loan)]

firm_panel[,avg_loans:=sum(total_loans)/11,by=c("state_abbr","county_code")]



firm_panel[,county_conforming_value:=sum(total_conforming_value),by=c("as_of_year","state_abbr","county_code")]
hhi = firm_panel[,list(hhi = sum( (100*total_conforming_value/county_conforming_value)^2),total=sum(total_conforming_value),num_loans=sum(total_loans)),by=c("as_of_year","state_abbr","county_code","avg_loans")]


ggplot(hhi[avg_loans>100]) + aes(x=log(total),y=hhi,size=total) + geom_point(alpha=0.5) + scale_x_continuous(breaks=c(5,10,15),labels=c(exp(5)/1e3,exp(10)/1e3,exp(15)/1e3))

trend = firm_panel[,list(loans=sum(total_loans),sold=sum(sold_value*total_conforming_value,na.rm=TRUE)/sum(total_conforming_value,na.rm=TRUE)),by=c("as_of_year","Other.Lender.Code")]

ggplot(trend) + aes(x=as_of_year,y=loans,group=Other.Lender.Code,color=Other.Lender.Code) + geom_line(linewidth=2)
ggplot(trend) + aes(x=as_of_year,y=sold,group=Other.Lender.Code,color=Other.Lender.Code) + geom_line(linewidth=2)


firm_panel[,annual_value:=sum(total_value,na.rm=TRUE),by=c("Respondent.Identification.Number","Agency.Code","Activity.Year")]
firm_panel[,oos_value:= (annual_value - total_value)*1.0]
firm_panel[,operating:=lending_years==operating_years]


trend = firm_panel[,list(banks=sum(operating),loans=sum(total_loans),value=sum(total_value*operating),oos_value=sum(oos_value*operating),sold=sum(operating*sold_value*total_conforming_value,na.rm=TRUE)/sum(operating*total_conforming_value,na.rm=TRUE)),by=c("as_of_year","Other.Lender.Code","state_abbr","county_code","avg_loans")]
trend = trend[avg_loans>100]

trend[,county:=paste(state_abbr,county_code)]
trend[,yearFactor:=as.factor(as_of_year)]


trend0 = trend[Other.Lender.Code==0]
trend3 = trend[Other.Lender.Code==3]
names(trend3) = paste(names(trend3),"nb",sep="_")
trend = merge(trend0,trend3,by.x=c("as_of_year","county"),by.y=c("as_of_year_nb","county_nb"))

trend[!is.na(banks_nb),banks_nb_sd:=banks_nb/sd(banks_nb)]
trend[!is.na(value_nb),value_nb_sd:=value_nb/sd(value_nb)]
trend[!is.na(oos_value_nb),oos_value_nb_sd:=oos_value_nb/sd(oos_value_nb)]


summary(feols(sold~banks_nb_sd|county + yearFactor,data=trend))
summary(feols(sold~value_nb_sd|county + yearFactor,data=trend))
summary(feols(sold~oos_value_nb_sd|county + yearFactor,data=trend))



trend[,summary(feols(sold~value_nb_sd|county + yearFactor))]
trend[,summary(feols(sold~oos_value_nb_sd|county + yearFactor))]

# df = df[property_type_name=="One-to-four family dwelling (other than manufactured housing)"]
# df = df[lien_status_name=="Secured by a first lien"]
# 
# df[,count:=1]
# df[,count:=sum(count),by="sequence_number"]
# 
# 
# orig = df[action_taken_name=="Loan originated"]
# 
# 
# nrow(unique(df[,c("applicant_income_000s","state_code","county_code", "census_tract_number","loan_amount_000s")]))
