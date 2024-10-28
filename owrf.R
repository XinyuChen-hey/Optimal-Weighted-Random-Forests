library(Metrics)
library(scorecard)
library(randomForest)
library(tidymodels)
library(lattice)
library(e1071)
library(caret)
library(rpart)  
library(ipred)
library(adabag)
library(ggplot2)
library(foreach)
library(doParallel)
library(iterators)
library(missForest)
library(parallel)
library(glmnet)
library(doParallel)
library(scorecard)
library(pracma) 
library(resample)
library(stats) 
library(purrr) 
library(MLmetrics)
library(Rsolnp)
library(doSNOW)
library(MASS)
library(dplyr)


get_diversity <- function(y_test, pred_Test_matrix){
  err <-  -(pred_Test_matrix - as.vector(y_test))
  n_test <- length(y_test)
  product_data_matrix <- t(err) %*% err / n_test
  lower_tri_indices <- lower.tri(product_data_matrix, diag = FALSE) # use lower.tri to choose (i<j) comninations
  numerators <- product_data_matrix[lower_tri_indices] # len=ntree*(ntree-1)/2
  numerator <- mean(numerators)
  
  sd_data <- sqrt(diag(product_data_matrix))                                              
  denominator <- mean(sd_data)
  
  rho_bar <- numerator / denominator**2
  return(rho_bar)
}


get_scores <- function(ypred,ytest){
  ypred = as.vector(ypred) 
  ytest = as.vector(ytest) 
  score1 = Metrics::mse(ytest,ypred)
  score2 = Metrics::mae(ytest,ypred)
  return(c(score1,score2))
}


process <- function(random_seed){
  set.seed(random_seed)
  ntree = 100
  
  filename = "./dataset/housing.csv"
  dataset_csv <- data.table::fread(filename,header = FALSE,sep = "")
  dataset <- data.matrix(dataset_csv)
  n <- dim(dataset)[1]
  p <- dim(dataset)[2]-1
  
  Xdata <- model.matrix(V14 ~ ., dataset_csv)[, -1]
  ydata <- dataset_csv$V14
  
  test_size = 0.3
  idx_test <- sample.int(n, size=floor(n*test_size), replace=F)
  idx_train <- setdiff(1:n, idx_test)
  X_train <- Xdata[idx_train,]
  y_train <- ydata[idx_train]
  X_test <- Xdata[idx_test,]
  y_test <- ydata[idx_test]
  n_train <-  length(y_train)
  n_test <- length(y_test)
  
  
  rf_reg <- randomForest::randomForest(x=X_train,y=y_train,mtry=floor(p/3),ntree = ntree,
                                       replace=TRUE,nodesize = floor(sqrt(n_train)),keep.inbag=TRUE,keep.forest = TRUE)

  pred_Train <- as.matrix(predict(rf_reg, X_train))
  pred_Test <- as.matrix(predict(rf_reg, X_test))
  pred_Train_matrix <- predict(rf_reg, X_train, predict.all=TRUE)$individual #dim=(n_train,ntree)
  pred_Test_matrix <- predict(rf_reg, X_test, predict.all=TRUE)$individual
  
  #---------------v get weight v-----------------------------------------------------#
  num_resample <- rf_reg$inbag
  nodeMatrix <- matrix(attr(predict(rf_reg, X_train, nodes = TRUE), "nodes"),
                       nrow = n_train, ncol = ntree)
  count_in_same_node <- sapply(1:ntree, function(j){
    column <- nodeMatrix[,j]
    unique_elements <- unique(column)
    
    count_in_same_node_thisCol <- sapply(unique_elements, function(x){
      this_idx <- which(column==x)
      this_count <- sum(num_resample[this_idx,j])
      return(this_count)
    })
    
    sapply(column, function(x){
      count_in_same_node_thisCol[which(x==unique_elements)]
    })
  })
  
  m <- num_resample / count_in_same_node
  
  w0 <- rep(1/ntree,ntree) 
  g <- function(w) sum(w)-1
  fun2 <- function(w){
    n_tree = ntree
    n_train = length(y_train)
    p_tt  <- m %*% w
    error <- y_train - pred_Train_matrix %*% w
    norm_error <- norm(error,"2")
    c = norm_error**2 + 2 * sum(error ** 2 * p_tt)
    return(c)
  }
  
  w_star_1step <- matrix(Rsolnp::solnp(w0,fun2,eqfun = g,eqB = 0,LB = rep(0,ntree),
                                       UB = rep(1,ntree))$pars)
  
  ## 2 steps OWRF:
  Y.hat <- t(pred_Train_matrix) #(n_tree,n_train)
  C0 <- 2*Y.hat %*% t(Y.hat) + diag(10^-6, ntree, ntree)
  sc0 <- norm(C0,"2")
  C0 <- C0/sc0
  y_pred_train <- predict(rf_reg, X_train)
  sigma.square.hat.C1 <- ((norm(y_train - pred_Train,"2"))**2)/n_train
  d0 <- as.vector((-2*Y.hat %*% y_train+ t(m) %*% matrix(sigma.square.hat.C1,nrow=n_train,ncol=1))/sc0)
  lb0 <- matrix(0,nrow=ntree,ncol=1)
  ub0 <- matrix(1,nrow=ntree,ncol=1)
  Aeq0 <- matrix(1,nrow=1,ncol=ntree)
  beq0 <- matrix(1,nrow=1,ncol=1)
  
  w0_star <- matrix(pracma::quadprog(C=C0,d=d0,Aeq=Aeq0,beq=beq0,lb=lb0,ub=ub0)$xmin)
  pred_train_C1_step1 <- pred_Train_matrix %*% w0_star
  residual_step1 <- as.matrix(y_train - pred_train_C1_step1) 
  ##step2:
  C0_ <- 2*Y.hat %*% t(Y.hat) + diag(10^-6, ntree, ntree)
  sc0_ <- norm(C0_,"2")
  C0_ <- C0_/sc0_
  d0_ <- as.vector((-2*Y.hat %*% y_train+t(m) %*% residual_step1**2)/sc0_)
  lb0_ <- matrix(0,nrow=ntree,ncol=1)
  ub0_ <- matrix(1,nrow=ntree,ncol=1)
  Aeq0_ <- matrix(1,nrow=1,ncol=ntree)
  beq0_ <- matrix(1,nrow=1,ncol=1)
  
  w_star_2steps <- matrix(pracma::quadprog(C=C0_,d=d0_,Aeq=Aeq0_,beq=beq0_,lb=lb0_,ub=ub0_)$xmin)
  #---------------^ get weight ^-----------------------------------------------------#
  
  y_pred_optimized1 <- pred_Test_matrix %*% w_star_1step
  y_pred_optimized2 <- pred_Test_matrix %*% w_star_2steps
  
  RF_score <- get_scores(pred_Test, y_test)
  OWRF1_score <- get_scores(y_pred_optimized1, y_test) 
  OWRF2_score <- get_scores(y_pred_optimized2, y_test) 
  
  #---------------v competitors v-----------------------------------------------------#
  to1 <- function(x) x/sum(x)
  
  y_train.broadcast <- matrix(rep(y_train,ntree),n_train,ntree)
  y_test.broadcast <- matrix(rep(y_test,ntree),n_test,ntree)
  
  indicator_OOB <- matrix(as.numeric(rf_reg$inbag==0),nrow=n_train,ncol=ntree)
  pred_eachtree_train <- predict(rf_reg, X_train, predict.all=TRUE)$individual 
  pred_eachtree_test <- predict(rf_reg, X_test, predict.all=TRUE)$individual 
  
  wRF <- function(){
    tPE = colSums(abs(pred_eachtree_train-y_train.broadcast)*indicator_OOB)/colSums(indicator_OOB)
    w_vector1 = to1((1/tPE)**5)
    w_vector2 = to1((1/tPE)**10)
    w_vector3 = to1((1+ntree)-(rank(1/tPE)))
    w_vector4 = to1(exp(1/tPE))
    pred_oneahead_wRF.list <- list("f1"=w_vector1%*%t(pred_eachtree_test),
                                   "f2"=w_vector2%*%t(pred_eachtree_test),
                                   "f3"=w_vector3%*%t(pred_eachtree_test),
                                   "f4"=w_vector4%*%t(pred_eachtree_test))
    return(pred_oneahead_wRF.list)
  }
  
  R.wRF <- function(){
    tPE = colMeans(abs(pred_eachtree_train-y_train.broadcast))
    w_vector1 = to1((1/tPE)**5)
    w_vector2 = to1((1/tPE)**10)
    w_vector3 = to1((1+ntree)-(rank(1/tPE)))
    w_vector4 = to1(exp(1/tPE))
    pred_oneahead_R.wRF.list <- list("f1"=w_vector1%*%t(pred_eachtree_test),
                                     "f2"=w_vector2%*%t(pred_eachtree_test),
                                     "f3"=w_vector3%*%t(pred_eachtree_test),
                                     "f4"=w_vector4%*%t(pred_eachtree_test))
    return(pred_oneahead_R.wRF.list)
  }
  
  CRF <- function(){
    error_OOB_eachtree <- colSums(((pred_eachtree_train-y_train.broadcast)^2)*indicator_OOB)/colSums(indicator_OOB)
    rank <- rank(error_OOB_eachtree)
    
    compute_weight <- function(r){
      mylist <- list()
      for (i in r:ntree){
        mylist <- c(mylist,1/i)
      }
      return(sum(as.vector(unlist(mylist))))
    }
    w <- numeric(ntree)
    for (j in 1:ntree){
      w[j] <- compute_weight(rank[j])
    }
    w <- to1(w)
    pred_CRF_test <- w %*% t(pred_eachtree_test) 
    return(pred_CRF_test)
  }
  
  pred_rf_test <- matrix(predict(rf_reg,X_test))
  pred_wRF_test <- wRF()
  pred_R.wRF_test <- R.wRF()
  pred_CRF_test <- CRF()
  
  wRF_scores <- c(get_scores(pred_wRF_test$f1,y_test),get_scores(pred_wRF_test$f2,y_test),get_scores(pred_wRF_test$f3,y_test),get_scores(pred_wRF_test$f4,y_test))
  R.wRF_scores <- c(get_scores(pred_R.wRF_test$f1,y_test),get_scores(pred_R.wRF_test$f2,y_test),get_scores(pred_R.wRF_test$f3,y_test),get_scores(pred_R.wRF_test$f4,y_test))
  CRF_scores <- get_scores(pred_CRF_test,y_test)
  
  tmp_len <- length(wRF_scores)
  wRF_scores <- c(min(c(wRF_scores[seq(1, tmp_len, 2)], R.wRF_scores[seq(1, tmp_len, 2)]))
                  , min(c(wRF_scores[seq(2, tmp_len, 2)], R.wRF_scores[seq(2, tmp_len, 2)])))
  #---------------^ competitors ^-----------------------------------------------------#
  
  rho_bar <- get_diversity(y_test, pred_Test_matrix)
  
  IR_OWRF1_score <- RF_score/OWRF1_score - 1
  IR_OWRF2_score <- RF_score/OWRF2_score - 1
  IR_wRF_score <- RF_score/wRF_scores - 1
  IR_CRF_score <- RF_score/CRF_scores - 1
  
  return(c(RF_score, OWRF1_score, OWRF2_score, wRF_scores, CRF_scores
           ,IR_OWRF1_score, IR_OWRF2_score, IR_wRF_score, IR_CRF_score
           , rho_bar)) 
}


set.seed(2021)
n_size <- 1000
seed <- runif(n_size,min=0,max=1e4)
cl <- makeCluster(50)
registerDoSNOW(cl)
pb <- txtProgressBar(max=n_size, style=3)
progress <- function(n) setTxtProgressBar(pb,n)
opts <- list(progress=progress)
out <- foreach(x=seed,.combine='rbind',.options.snow = opts) %dopar% process(x)
close(pb)
stopCluster(cl)


save(out)

avg_out <- matrix(colMeans(out), nrow=1)
colnames(avg_out) <- c("RF_mse", "RF_mae"
                                    ,"OWRF1_mse", "OWRF1_mae"
                                    , "OWRF2_mse", "OWRF2_mae"
                                    , "wRF_mse", "wRF_mae"
                                    , "CRF_mse", "CRF_mae"
                                    , "IR_OWRF1_mse", "IR_OWRF1_mae"
                                    , "IR_OWRF2_mse", "IR_OWRF2_mae"
                                    , "IR_wRF_mse", "IR_wRF_mae"
                                    , "IR_CRF_mse", "IR_CRF_mae"
                                    , "rhoBar")
save(avg_out)


















