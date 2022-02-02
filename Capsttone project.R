

# Create edx set, validation set (final hold-out test set)- (provided by the edx team)

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")


set.seed(1) 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]


validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")


removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#Check if the data has been loaded correctly

dim(edx)
dim(validation)

#Data Exploration 
#Split the edx data into train and test sets: 

set.seed(1)
index<-createDataPartition(validation$rating,p=0.1,times = 1,list = FALSE)
test<-edx[index,]
train<-edx[-index,]


#Explore the train data set

str(train)
summary(train)

#Analysis of movieId

mu<-mean(train$rating)
train%>%group_by(movieId)%>%summarize(n=n(),mean_rating=mean(rating))%>%arrange(desc(mean_rating))%>%slice(1:20)
train%>%group_by(movieId)%>%summarize(n=n(),mean_rating=mean(rating))%>%arrange(mean_rating)%>%slice(1:20)
library(ggplot2)
train%>%group_by(movieId)%>%summarize(n=n(),mean_rating=mean(rating))%>%ggplot(aes(mean_rating,n)) + geom_point()



#Analysis of userId

train%>%group_by(userId)%>%summarize(n=n(),mean_rating=mean(rating))%>%arrange(desc(mean_rating))%>%slice(1:20)
train%>%group_by(userId)%>%summarize(n=n(),mean_rating=mean(rating))%>%arrange(mean_rating)%>%slice(1:20)
train%>%group_by(userId)%>%summarize(n=n(),mean_rating=mean(rating))%>%ggplot(aes(mean_rating,n)) + geom_point()





#Analysis of Genres

length(unique(train$genres))
train%>%mutate(genres=factor(genres))%>%group_by(genres)%>%summarize(n=n(),mean_rating=mean(rating))%>%arrange(desc(mean_rating))
train%>%mutate(genres=factor(genres))%>%group_by(genres)%>%summarize(n=n()/1000,mean_rating=mean(rating))%>%ggplot(aes(mean_rating,n)) + geom_point()


#Timestamp :Split the timestamp to date

library(lubridate)
train%>%mutate(date=as_datetime(timestamp))%>%mutate(date=round_date(date,unit="week"))%>%group_by(date)%>%summarize(rating=mean(rating))%>%ggplot(aes(date,rating)) + geom_point() + geom_smooth()

# Modelling and Results

# First Model: Just the mean

mu<-mean(train$rating)
rmse_1<-RMSE(mu,test$rating)
Results<-data.frame(Model="Just the Mean",RMSE=rmse_1)
Results


#Second Model:The movie Effect

movie_effect<-train%>%group_by(movieId)%>%summarize(b_i=mean(rating-mu))
pred_2<-test%>%left_join(movie_effect,by="movieId")%>%mutate(pred=mu+b_i)%>%.$pred
rmse_2<-RMSE(pred_2,test$rating,na.rm = TRUE)
Results<-rbind(Results,data.frame(Model="Movie Effect",RMSE=rmse_2))
Results

#The Third model: The movie + user Effect  

user_effect<-train%>%left_join(movie_effect,by="movieId")%>%group_by(userId)%>%summarize(b_u=mean(rating-mu-b_i))
pred_3<-test%>%left_join(movie_effect,by="movieId")%>%left_join(user_effect,by="userId")%>%mutate(pred=mu+b_i+b_u)%>%.$pred
rmse_3<-RMSE(pred_3,test$rating,na.rm = TRUE)
Results<-rbind(Results,data.frame(Model="Movie + User Effect",RMSE=rmse_3))
Results

#The Forth model: The movie + user  + genre Effect  

genre_effect<-train%>%left_join(movie_effect,by="movieId")%>%left_join(user_effect,by="userId")%>%group_by(genres)%>%summarize(b_g=mean(rating-mu-b_i-b_u))
test<-test%>%mutate(genres=factor(genres))
pred_4<-test%>%left_join(movie_effect,by="movieId")%>%left_join(user_effect,by="userId")%>%left_join(genre_effect,by="genres")%>%mutate(pred=mu+b_i+b_u+b_g)%>%.$pred
rmse_4<-RMSE(pred_4,test$rating,na.rm = TRUE)
Results<-rbind(Results,data.frame(Model="Movie + User + Genre Effect",RMSE=rmse_4))
Results

# The fifth Model: Regularization of the variables

 
lambda<-seq(0,50,1)
test<-test%>%mutate(genres=factor(genres))
rmse<-function(lambda){
  movie_reg<-train%>%group_by(movieId)%>%summarize(b_i=sum(rating-mu)/(n() +lambda))
  user_reg<-train%>%left_join(movie_reg,by="movieId")%>%group_by(userId)%>%summarize(b_u=sum(rating-mu-b_i)/(n()+lambda))
  genre_reg<-train%>%left_join(movie_reg,by="movieId")%>%left_join(user_reg,by="userId")%>%group_by(genres)%>%summarize(b_g=sum(rating-mu-b_i-b_u)/(n()+lambda))
  pred_5<-test%>%left_join(movie_reg,by="movieId")%>%left_join(user_reg,by="userId")%>%left_join(genre_reg,by="genres")%>%mutate(pred=mu+b_i+b_u+b_g)
  rmse_5<-RMSE(pred_5$pred,test$rating,na.rm = TRUE)
}
rmses<-data.frame(lambda,sapply(lambda, rmse))
optimal<- rmses[which.min(rmses[,2]),2]#This code may take around 5 minutes
optimal

Results<-rbind(Results,data.frame(Model="Regularization Effect",RMSE=optimal))
Results

#Test the final model on the Validation data

optimal_lambda <-rmses[which.min(rmses[,2]),1]
movie_reg<-train%>%group_by(movieId)%>%summarize(b_i=sum(rating-mu)/(n() +optimal_lambda))
user_reg<-train%>%left_join(movie_reg,by="movieId")%>%group_by(userId)%>%summarize(b_u=sum(rating-mu-b_i)/(n()+optimal_lambda))
genre_reg<-train%>%left_join(movie_reg,by="movieId")%>%left_join(user_reg,by="userId")%>%group_by(genres)%>%summarize(b_g=sum(rating-mu-b_i-b_u)/(n()+optimal_lambda))
final_predictions<-validation%>%left_join(movie_reg,by="movieId")%>%left_join(user_reg,by="userId")%>%left_join(genre_reg,by="genres")%>%mutate(pred=mu+b_i+b_u+b_g)
Final_RMSE<-RMSE(final_predictions$pred,validation$rating,na.rm = TRUE)
Final_RMSE

