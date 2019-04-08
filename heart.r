

heart.data <- read.csv("C:/Users/ssn/Desktop/heakthcare/heartdetection.csv")

head(heart.data) #to check if data got loaded

names(heart.data) <- c( "age", "sex", "cp", "trestbps", "chol","fbs", "restecg",
                        "mhr","eia", "oldpeak","slope", "ca", "thal", "num")


avg_age=mean(heart.data[["age"]])  
avg_bp= mean(heart.data[["trestbps"]]) 
avg_chol=mean(heart.data[["chol"]])
avg_thal=mean(heart.data[["thal"]])
avg_old=mean(heart.data[["oldpeak"]])

m_age=median(heart.data[["age"]])
m_bp=median(heart.data[["trestbps"]])
m_chol=median(heart.data[["chol"]])
m_thal=median(heart.data[["thal"]])
m_old=median(heart.data[["oldpeak"]])

sd(heart.data[["age"]])
sd(heart.data[["trestbps"]])
sd(heart.data[["chol"]])
sd(heart.data[["thal"]])
sd(heart.data[["oldpeak"]])

range(heart.data[["age"]])
range(heart.data[["trestbps"]])
range(heart.data[["chol"]])
range(heart.data[["thal"]])
range(heart.data[["oldpeak"]])


boxplot(heart.data[["age"]],main = "Age")
boxplot(heart.data[["trestbps"]],main = "BP")
boxplot(heart.data[["chol"]],main = "cholostrol")
boxplot(heart.data[["thal"]],main='maximum heart rate')
boxplot(heart.data[["oldpeak"]],main='old peak')


cor(heart.data[["age"]], heart.data["num"]) 
cor(heart.data[["trestbps"]], heart.data["num"])
cor(heart.data[["chol"]], heart.data["num"])
cor(heart.data[["thal"]], heart.data["num"])
cor(heart.data[["oldpeak"]], heart.data["num"])



quantile(heart.data[["age"]])
quantile(heart.data[["trestbps"]])
quantile(heart.data[["chol"]])
quantile(heart.data[["thal"]])
quantile(heart.data[["oldpeak"]])



