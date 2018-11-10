# source packages
library(ggplot2)
library(data.table)

# load rtrain data
trainSet = as.data.table(read.csv("initial_data/train.csv"))

# get summary
summary(trainSet)

# Gender - survived
ggplot(data=trainSet, aes(x=Sex, y=Survived)) +
        geom_bar(stat="identity")

# boxplot
trainSet$Survived = as.character(trainSet$Survived)
ggplot(data=trainSet, aes(x=Survived, y=Age)) +
  ggplot2::geom_boxplot()


