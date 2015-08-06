# machine learning from disaster -- datacamp ML in R tutorial
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)


train <- read.csv(url("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"))
test <- read.csv(url("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"))
head(train)

table(train$survived)
table(train$Sex, train$Survived)
prop.table(table(train$Survived)) #overall survival rate
prop.table(table(train$Sex, train$Survived),1)   #m vs f survival rates


#Child survival rate
train$Child <- NULL
train$Child[train$Age == NA] <- NA
train$Child[train$Age < 18] <- 1
train$Child[train$Age >= 18] <- 0
test$Child <- NULL
test$Child[test$Age < 18] <- 1
test$Child[test$Age >= 18] <- 0

# Two-way comparison
prop.table(table(train$Child, train$Survived), 1)

#simple prediction: males die, females survive
test_one <- test
test_one$Survived <- NULL
test_one$Survived[test_one$Sex == 'female'] <- 1
test_one$Survived[test_one$Sex == 'male'] <- 0


#Build Decision Tree:
decisionTree <- rpart(formula = Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train, method = "class")
fancyRpartPlot(decisionTree)

#make prediction, put in submission format
my_prediction <- predict(decisionTree, test, "class")
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = predict(decisionTree, test, "class"))

nrow(my_solution)
#write.csv(my_solution, file = "my_solution.csv", row.names = FALSE)

#new var: family size 
train$family_size <- train$SibSp + train$Parch + 1
test$family_size <- test$SibSp + test$Parch + 1

# Create a new decision tree my_tree_three
tree_two<- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + family_size, 
                      data = train, method ="class")
fancyRpartPlot(tree_two)


####### RANDOM FORESTS #########
# All data, both training and test set
test$Survived <- 0

#combine, create new var 'title'
combi <- rbind(train, test)
combi$Name <- as.character(combi$Name)
strsplit(combi$Name[1], split='[,.]')
strsplit(combi$Name[1], split='[,.]')[[1]]
strsplit(combi$Name[1], split='[,.]')[[1]][2]
combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combi$Title <- sub(' ', '', combi$Title)
combi$Title[combi$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
combi$Title <- factor(combi$Title)


combi$Embarked[c(62,830)] = "S"
combi$Embarked <- factor(combi$Embarked)

# Passenger on row 1044 has an NA Fare value - replace it with median fare value.
combi$Fare[1044] <- median(combi$Fare, na.rm=TRUE)

#Predict passengers age using the other variables and a decision tree model
predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + family_size,
                       data=combi[!is.na(combi$Age),], method="anova")
combi$Age[is.na(combi$Age)] <- predict(predicted_age, combi[is.na(combi$Age),])

# Split the data back into a train set and a test set
train <- combi[1:891,]
test <- combi[892:1309,]

#Apply Random Forest 
set.seed(111)
my_forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title, data = train, importance = TRUE, ntree=1000)
my_prediction <- predict(my_forest, test)

my_solution <- data.frame(PassengerId = test$PassengerId, Survived = predict(my_forest, test))
write.csv(my_solution,file = "my_solution.csv", row.names = FALSE)


varImpPlot(my_forest)
prop.table(table(train$Survived))
prop.table(table(my_solution$Survived))




