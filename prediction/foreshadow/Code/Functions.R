library(dplyr)
library(lme4)
library(e1071)
library(lmtest)
library(lubridate)
library(knitr)
library(BaylorEdPsych)
library(pscl)
#source('rsquaredglmm.R')
# source('nbclass.R')


downsampleData <- function(df, var_name, replace=FALSE){
  #' Downsample dataset to have equal numbers of var_name==1 and var_name == 2
  #' @df dataset to be resampled
  #' @var_name which variable to resample on
  #' @replace whether to use sampling by replacement (default FALSE)
  
  # Determine the number of values used in downsampling. 
  var_values = unlist(sort(unique(as.numeric(unlist(df[var_name])))))
  
  if(length(var_values)<2) {
    stop("Trying to downsample based on a variable with fewer than two levels!!")
  } else if(length(var_values) > 2){
    stop("Trying to downsample based on a variable with more than two levels!!")
  }
  
  print("Before Downsample Balance:")
  print(table(as.numeric(unlist(df[var_name]))))
  
  ind0 <- which(as.numeric(unlist(df[var_name]))==unlist(var_values)[1])
  ind1 <- which(as.numeric(unlist(df[var_name]))==unlist(var_values)[2])
  
  # downsample results. to upsample, use max and replace = TRUE
  sampsize <- min(length(ind1), length(ind0))
  sampind1 <- sample(ind1, sampsize, replace = replace)
  sampind0 <- sample(ind0, sampsize, replace = replace)
  sampind <- c(sampind1,sampind0)
  balanced.df <- df[sampind,]
  print("After Downsample Balance:")
  print(table(as.numeric(unlist(balanced.df[var_name]))))
  balanced.df
}

#' ModelAccuracy
#' Function to evaluate the accuracy of a model. Can also return model
#' predictions given an input dataset. If verbose=T, show confusion matrix
#' and display accuracy confidence intervals and p value based on a binomial
#' test assuming 50% accuracy in the test set (i.e. binomial test assumes
#' downsampled test data.)
#' 
#' @param mdl a trained model
#' @param d.test the training data
#' @param y the variable containing the result
#' @param type which model type (currently glm, svm, or naivebayes)
#' @param predictions whether to return a list l(accuracy, predictions)
#' @param verbose whether to print out a number of model statistics
#' @param complete_result whether to return a list l(accuracy, 95 confidence interval, )
#'
#' @return the model accuracy (or l(accuracy, predictions))
ModelAccuracy <- function(mdl, d.test, y="Result", type='glm', predictions=FALSE, verbose=F, complete_result=F){
  mdl.vars <- all.vars(formula(mdl))
  if(verbose) print(type)
  pred.bin <- NULL
  if (type == 'glm') {
    pred = predict(mdl,
                   newdata = d.test,
                   type = 'response',
                   allow.new.levels = TRUE)
    pred.bin = factor(ifelse(pred >= 0.5, 1, 0), levels = c(0, 1))
  }
  
  if (type == 'svm' || type == "rf") {
    pred = predict(mdl, d.test)
    pred.bin <-
      factor (as.numeric(pred),
              levels = c(1, 2),
              labels = c(0, 1))
  }
  
  if (type == 'naivebayes') {
    pred = predict(mdl, d.test, type = 'raw')[, 2]
    pred.bin = factor(ifelse(pred >= 0.5, 1, 0), levels = c(0, 1))
  }
  confusion.matrix = table(REAL = unlist(d.test[y]), MODEL = unlist(pred.bin))
  if(verbose){
    print("Confusion matrix:")
    print(confusion.matrix)
  }
  n.true = confusion.matrix[1, 1]
  
  try(n.true <- n.true + confusion.matrix[2, 2])
  #sometimes you only get one value in the result with sparse data
  n.false <- sum(confusion.matrix) - n.true
  
  acc <- n.true / sum(confusion.matrix)
  # calculate precision and recall which assume a different table format
  precision <- "undefined"
  recall <- "undefined"
  tryCatch({
    cm2 <<- caret::confusionMatrix(
      unlist(pred.bin),
      unlist(factor(as.numeric(unlist(d.test[y]))-1, levels=c(0,1))),
      positive="1"
      
    )
    if(verbose) print(cm2)
    precision <- caret::precision(cm2$table, relevant="1")
    recall <- caret::recall(cm2$table, relevant="1")
  }, error = function(e){
  })

  if(verbose){
    print("Precision")
    print(precision)
    print("Recall")
    print(recall)
  }
  
  #### GLOSSY RESULTS for verbose=T
  ### Construct a hypothetical n.true, n.false model for a null model
  ### if n is even, we add the leftover to the random model's true count
  n.data <- floor(sum(confusion.matrix) / 2) * 2 # ensure even
  result.table <-
    data.frame(NULL = c(n.data / 2 + sum(confusion.matrix) %% 2, n.data / 2),
               Model = c(n.true, n.false))
  
  n = nrow(d.test)
  r = confusion.matrix[1, 2] + confusion.matrix[2, 1]
  e.s = r / n
  
  error = 1.96 * (sqrt(e.s * (1 - e.s) / n))
  if (verbose) {
    print("95% Confidence interval classifier accuracy = +/-:")
    print(error)
    print(
      "Binomial test: testing whether classifier accuracy is higher than chance
      (compare results model against coin flipping)"
    )
    print(result.table)
    try(print(binom.test(n.true, n.data)))
  }
  
  if (predictions == TRUE) {
    l <-
      list(
        "accuracy" = acc,
        "predictions" = data.frame(
          pred = pred.bin,
          ticker = d.test$ticker,
          date = d.test$date
        )
      )
    return(l)
  }
  if (complete_result == TRUE) {
    l <- list(
      "accuracy" = acc,
      "error" = error,
      "precision"=precision,
      "recall"  = recall
    )
    return(l)
  }
  acc
}

fixLevels <- function(df, y){
  if(all(sort(unique(as.numeric(unlist(df[y])))) == c(1,2))){
    df[y] <- factor(as.numeric(unlist(df[y]))-1, levels=c(0,1))
  }
  df
}

trainTestSplit <- function(df, split.method='average', y="Result", downsample.var=NULL, days=5, first_day=1){ 
  # Function that takes df and splits it into a train set and a test set
  # and returns them as a list l<- list(train, test). Uses splitting by average
  # to make two datasets that have n_days in the train set and 1 test in the 
  # test set
  set.seed(0) # this means that if you run your function more than once, you 
              # should get the same result
  
  #Change response levels to 0,1 from 1, 2
  df <- fixLevels(df, y)
  
  if(split.method=='average'){
    d.train = df[(df$Trial %in%  first_day:(first_day + days - 1)),]
    if(!is.null(downsample.var)){
      d.train <- downsampleData(d.train, var_name=downsample.var)
    }
    d.test = df[df$Trial == (first_day + days),]
    # print(table(d.train[y]))
    # print(table(d.test[y]))
    l <- list("train" = d.train, "test" = d.test)
  } else {
    print('trainTestSplit called with split.method != average')
  }

  l
}

fitModel <- function(df, d.train, formula, type='glm', verbose=FALSE){
  # This is the function that takes a dataset, a training set (a subset of df),
  # and a formula, and a type, and trains a model of type 'type' using the
  #formula and returns the model. If verbose is true, then it prints out
  # the results for the model summary/anova/r2/lrtest against null model.
  # type can be glm/svm/or naivebayes
  needs.multilevel <- '|' %in% unlist(strsplit(gsub("[^[:alnum:]|'_' ]", "", as.character(formula)), c(' ')))
  if(type=='glm'){
    if(needs.multilevel){
      null_formula <- as.formula(paste(as.character(formula)[2], '~', '(1 | Subject)'))
      model <- glmer(formula, data = df, family = binomial(link = logit), control=glmerControl(optimizer="bobyqa"))
      m2 <- glmer(null_formula, data = df, family = binomial(link = logit), control=glmerControl(optimizer="bobyqa"))

      if(verbose){
        print(summary(model), correlation=T)
        if(needs.multilevel){
          print(rsquared.glmm(model))
        }
        try(print(lrtest(model, m2)))
        print(anova(model))
      }
      model <- glmer(formula, data = d.train, family = binomial(link = logit), control=glmerControl(optimizer="bobyqa"))

    } else{
      null_formula <- as.formula(paste(as.character(formula)[2], '~', '1'))
      model <- glm(formula, data=df, family="binomial")
      m2 <- glm(null_formula, data=df, family="binomial")

      if(verbose){
        print(summary(model))
        print(PseudoR2(model))
        print("Compare model against baseline model with no regressors: is model significant by itself?")
        try(print(lrtest(model, m2)))
        print(anova(model))
      }
      #print(summary(d.train))
      model <- glm(formula, data=d.train, family="binomial")
    }
  }
  if(type=='svm'){
     model <- svm(formula, data=d.train)
  }
  if(type=='rf'){
    model <- randomForest::randomForest(formula, data=d.train)
  }
  if(type=='naivebayes'){
    if(needs.interaction){
      print('naiveBayes cannot handle interaction terms')
    } else{
      model <-  naiveBayes(formula, data=d.train)
    }
  }
  model
}

.eval_model <- function(df, formula, swap=FALSE, type='glm', split.method='lastday', 
                        verbose=FALSE, days=5, days_train=5, 
                        predictions=FALSE, test.df = F, downsample.var=NULL, 
                        complete_result=F){
  # This is the function that actually calls everything that you need in order to 
  # first split df in to train and test and then get the accuracy. 
  if(split.method=='average'){
    trainacc <- c()
    testacc <- c()
    first_day<-1
    
    if(!(test.df==F)){
      days_train <- 10
    }

    y<-as.character(formula[2])
    
    if(test.df==F){
      train.test <<- trainTestSplit(df, split.method, y=y, days=days_train, first_day = first_day, downsample.var=downsample.var)
    } else {
      df <- fixLevels(df, y)
      if(is.null(downsample.var)){
        train.test <- list("train" = df, "test" = test.df)
      } else {
        train.test <- list("train" = downsampleData(df, var_name = downsample.var), "test" = test.df)
      }
    }
    m <- fitModel(df, train.test$train, formula, type, verbose)
    if(complete_result){
      train_accuracy_results <- ModelAccuracy(m, train.test$train, y=y, type, complete_result=T)
      test_accuracy_results <- ModelAccuracy(m, train.test$test, y=y, type, verbose=verbose, complete_result=T)
      result = list(
        "train_accuracy" = train_accuracy_results$accuracy,
        "train_error" = train_accuracy_results$error,
        "train_precision" = train_accuracy_results$precision,
        "train_recall" = train_accuracy_results$recall,
        "test_accuracy" = test_accuracy_results$accuracy,
        "test_error" = test_accuracy_results$error,
        "test_precision" = test_accuracy_results$precision,
        "test_recall" = test_accuracy_results$recall
      )
      if(verbose) print(unlist(result))
      return(result)
    }
    trainacc <- c(trainacc, ModelAccuracy(m, train.test$train, y=y, type))
    testacc <- c(testacc, ModelAccuracy(m, train.test$test, y=y, type, verbose=verbose))
    if(verbose) print("train accuracy", trainacc)
    train.accuracy = mean(trainacc)
    test.accuracy = mean(testacc)
  }
  list('train.accuracy'=train.accuracy, 'test.accuracy'=test.accuracy)
}

checkClassifier <- function(df, formula, type='glm', downsample.var= NULL, 
                            split.method='lastday', sep.stocks=F, sep.subjects=F,
                            days=5, days_train=3, predictions=F, d.test=F,
                            complete_result=FALSE, verbose=T){
  # This function is used to first make sure we have no missing data
  # for the variables in formula, then depending on whether we're doing separate
  # stocks/separate subjects/neither calls .eval_classifier, which will
  # return an accuracy number for each model you train. If you do
  #separate subjecst/stocks then this function will average them. 
  # mostly it just determines how many models we need to run. 
  
  # if downsample.var is NULL, then WE DO NOT Downsample. You MUST specify downsample.var
  # in order for results to be downsampled. 
  formula.vars <- unlist(strsplit(gsub("[^[:alnum:]|'_' ]", "", as.character(formula)), c(' ')))
  for(v in formula.vars){
    if(v %in% colnames(df)) {
      df <- df[!is.na(df[v]), ]
    }
  }

  swap <- !is.na(pmatch('Is_Inflection',formula))
  d.res <- data.frame()
  if(!sep.stocks){
    if(sep.subjects){
      for(subj in unique(df$Subject)){
        accuracies <- .eval_model(filter(df, Subject==subj), formula, swap, type, split.method, days=days, days_train=days_train, downsample.var=downsample.var)
        d.res<- rbind(d.res, data.frame(Subject=subj, accuracies))
      }
      d.res <- d.res[order(-d.res$test.accuracy),]
      d.res<- rbind(d.res, data.frame(Subject='Average', t(colMeans(d.res[,-1]))))
    } else {
      accuracies <- .eval_model(df, formula, swap, type, split.method, verbose=verbose,
                                days=days, predictions, days_train=days_train, 
                                test.df=d.test, downsample.var=downsample.var,
                                complete_result=complete_result)
      if((split.method=='future' & predictions==TRUE)) {#split.method=='n_days_future' | 
        the.acc <<- accuracies
        d.res <- accuracies
      } else {
        if(verbose) print(accuracies)
        d.res <- rbind(d.res, data.frame(stock='All', accuracies))
      }
    }
  } else {
    for(stock in unique(df$ticker)){
      accuracies <- .eval_model(filter(df, ticker==stock), formula, swap, type, split.method, days_train=days_train, downsample.var=downsample.var)
      d.res<- rbind(d.res, data.frame(stock=stock, accuracies))
    }
    d.res<- rbind(d.res, data.frame(stock='Average', t(colMeans(d.res[,-1]))))
  }

  if(is.null(downsample.var)){
    warning("Results for models without downsampling. Accuracy/precision/recall will not be as reliable.\n")
  } else {
    if(verbose) print(paste("Results for models downsampled using variable", downsample.var))
  }
  if(verbose){
    print("Final Accuracy results:")
    print(kable(d.res))
  }
  d.res
}
