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
library(doSNOW)
library(tcltk)


process <- function(random_seed){
  buildTree <- function(X_train,y_train,boot,seed,nodesize,mtry,varImp){
    set.seed(seed)
    n_train <- dim(X_train)[1]
    p <- dim(X_train)[2]
    if (boot){
      indices <- as.vector(resample::samp.bootstrap(n=n_train,R=1,size=n_train))
    }else{
      indices <- 1:n_train
    }
    rawTreeInfo <- splitProcess(indices,nodesize,seed,n_train,varImp)
    
    leaf.indicator <- unname(unlist(sapply(rawTreeInfo,tail,1)))
    leafPred <- rep(NA,length(rawTreeInfo))
    leafPred[leaf.indicator] <- unlist(lapply(which(leaf.indicator),
                                              function(x) mean(y_train[rawTreeInfo[[x]]$indices])))
    TreeInfo <- lapply(1:length(rawTreeInfo),
                       function(x) c(rawTreeInfo[[x]],list("pred"=leafPred[x])))
    
    return(TreeInfo)
  }
  
  
  
  splitProcess <- function(thisIndices,nodesize,seed,n_train,varImp){
    thisTreeInfoList <- list()
    iter <- 0
    iteration <- function(thisNodeIndices,n_train){
      if (length(thisNodeIndices)>nodesize & iter<=100){
        iter <<- iter + 1
        isLeaf <- FALSE
        tmp_list <- findBestSplit(thisNodeIndices,p,seed,varImp)
        thisVar <- tmp_list$Var
        thisValue <- tmp_list$Value
        left.daughter.indices <- thisNodeIndices[unname(which(X_train[thisNodeIndices,thisVar]<=thisValue))]
        right.daughter.indices <- thisNodeIndices[unname(which(X_train[thisNodeIndices,thisVar]>thisValue))]
        
        if (length(left.daughter.indices)<5 || length(right.daughter.indices)<5){
          tmp <- list("indices"=as.vector(thisNodeIndices),"splitVar"=NA,"splitValue"=NA,"leftIndices"=NA,"rightIndices"=NA,"isLeaf"=TRUE)
          thisTreeInfoList[[(length(thisTreeInfoList) + 1)]] <<- tmp
        }else{
          tmp <- list("indices"=as.vector(thisNodeIndices),"splitVar"=thisVar,"splitValue"=thisValue,"leftIndices"=as.vector(left.daughter.indices),"rightIndices"=as.vector(right.daughter.indices),"isLeaf"=isLeaf)
          thisTreeInfoList[[(length(thisTreeInfoList) + 1)]] <<- tmp
          iteration(left.daughter.indices,n_train)
          iteration(right.daughter.indices,n_train)
        }
      }
      else{
        iter <<- iter + 1
        tmp <- list("indices"=as.vector(thisNodeIndices),"splitVar"=NA,"splitValue"=NA,"leftIndices"=NA,"rightIndices"=NA,"isLeaf"=TRUE)
        thisTreeInfoList[[(length(thisTreeInfoList) + 1)]] <<- tmp
      }
    }
    
    iteration(thisIndices,n_train=n_train)
    return(thisTreeInfoList)
  }  
  
  
  get_var_importance <- function(X_valid,y_valid,seed0,p,ntree,nodesize){
    set.seed(seed0)
    rf_reg <- randomForest::randomForest(x=X_valid,y=y_valid,mtry=floor(p/3),ntree = ntree,
                                         replace=TRUE,nodesize = nodesize,importance=TRUE)
    varImp <- randomForest::importance(x=rf_reg,type=2)
    varImp_to1 <- as.vector(to1(varImp))
    return(varImp_to1)
  }
  
  
  findBestSplit <- function(thisIndices,p,seed,varImp){
    set.seed(seed)
    var.idx <- sample(x=1:p,size=floor(p/3),replace=FALSE,prob=varImp)
    df <- data.frame("var"=var.idx)
    df$var.min <- lapply(var.idx, function(x) min(X_train[thisIndices,x]))
    df$var.max <- lapply(var.idx, function(x) max(X_train[thisIndices,x]))
    df$cut <- lapply(var.idx, function(x) median(X_train[thisIndices,x]))
    
    score <- function(rowInfo){
      thisVar <- rowInfo$var
      thisValue <- rowInfo$cut[[1]]
      if (rowInfo$var.max==rowInfo$cut[[1]]){
        left.indices <- thisIndices[X_train[thisIndices,thisVar]<thisValue]
        right.indices <- thisIndices[X_train[thisIndices,thisVar]>=thisValue]
      }else{
        left.indices <- thisIndices[X_train[thisIndices,thisVar]<=thisValue]
        right.indices <- thisIndices[X_train[thisIndices,thisVar]>thisValue]
      }
      n_parent <- length(thisIndices)
      n_left <- length(left.indices)
      n_right <- length(right.indices)
      
      colMedian <- apply(X_train,MARGIN = 2,median)
      colScale <- apply(X_train,MARGIN = 2,function(x) max(x)-min(x))
      
      compactness <- function(thisIndices_){
        n_this_dim <- length(thisIndices_)
        X.normed <- sweep(X_train[thisIndices_,],2,colMedian,FUN="-")
        X.normed <- sweep(X.normed,2,colScale,FUN="/")
        cps <- apply(X.normed, MARGIN = 2, norm,type="2")
        cp <- mean(cps)
        return(cp)
      } 
      if (n_left<=2 || n_right<=2){
        increase <- 0
      }else{
        parent.cp <- compactness(thisIndices)
        left.cp <- compactness(left.indices)
        right.cp <- compactness(right.indices)
        increase <- (parent.cp - left.cp*n_left/n_parent - right.cp*n_right/n_parent) / parent.cp
      }
      return(increase)
    }
    
    df$var.score <- apply(df,1, score)
    best.Var <- unlist(df$var)[which.max(unlist(df$var.score))]
    best.Value <- unlist(df$cut)[which.max(unlist(df$var.score))]
    return(list("Var"=best.Var, "Value"=best.Value))
  }
  
  
  predTree <- function(x_singlePoint,thisTreeInfo,isTrain){
    indicesList <- lapply(1:length(thisTreeInfo),function(x) thisTreeInfo[[x]]$indices)
    i <- 1
    while(!thisTreeInfo[[i]]$isLeaf){
      toLeft <- unname(x_singlePoint[thisTreeInfo[[i]]$splitVar] <= thisTreeInfo[[i]]$splitValue)
      if(toLeft){
        next.indices <- thisTreeInfo[[i]]$leftIndices
      }else{
        next.indices <- thisTreeInfo[[i]]$rightIndices
      }
      i <- which(unlist(lapply(1:length(indicesList),function(x) identical(next.indices,indicesList[[x]]))))
    }
    pred <- thisTreeInfo[[i]]$pred
    if (isTrain){
      thisNodeIndices <- thisTreeInfo[[i]]$indices
      x_idx <- which(unlist(lapply(1:dim(X_train)[1], function(x) identical(x_singlePoint,X_train[x,]))))
      m_percent <- sum(x_idx==thisNodeIndices) / length(thisNodeIndices)
      inBag <- x_idx %in% as.vector(indicesList[[1]]) 
      oob <- !inBag
    }else{
      m_percent <- NA
      oob <- NA
    }
    return(list("pred"=pred,"m_ij"=m_percent,"oob"=oob))
  }
  
  
  predsTree <- function(Xs,thisTreeInfo,isTrain){
    tmp <- apply(Xs,1,predTree,thisTreeInfo=thisTreeInfo,isTrain=isTrain)
    preds_this_single_tree <- unname(unlist(sapply(tmp,head,1)))
    m_this_single_tree <- unname(unlist(sapply(tmp,function(x) x[2])))
    oob_this_single_tree <- unname(unlist(sapply(tmp,tail,1)))
    return(list("pred"=preds_this_single_tree,"m"=m_this_single_tree,"oob"=oob_this_single_tree))
  }
  
  
  get_pred_matrix <- function(X_train,y_train,X_test,X_valid,boot,seed0,nodesize,mtry,ntree,varImp){
    set.seed(seed0)
    seeds <- runif(ntree,min=0,max=1e4)
    
    fun_pred <- function(j){
      tmp_treeInfo <- buildTree(X_train,y_train,boot,seeds[j],nodesize,mtry,varImp)
      pred_Test_matrix_j <- predsTree(X_test,tmp_treeInfo,isTrain=FALSE)$pred
      pred_Valid_matrix_j <- predsTree(X_valid,tmp_treeInfo,isTrain=FALSE)$pred
      train_tmp <- predsTree(X_train,tmp_treeInfo,isTrain=TRUE)
      pred_Train_matrix_j <- train_tmp$pred
      m_j <- train_tmp$m
      oob_j <- train_tmp$oob
      rm(tmp_treeInfo)
      gc()
      list_j <- list("train"=pred_Train_matrix_j, "test"=pred_Test_matrix_j, "valid"=pred_Test_matrix_j, "m"=m_j, "oob"=oob_j)
      return(list_j)
    }
    
    out_tree <- lapply(1:ntree, fun_pred)
    
    pred_Train_matrix <- matrix(unname(unlist(sapply(out_tree,head,1))), nrow=dim(X_train)[1], ncol=ntree, byrow=FALSE)
    pred_Test_matrix <- matrix(unname(unlist(sapply(out_tree,function(x) x[2]))), nrow=dim(X_test)[1], ncol=ntree, byrow=FALSE)
    pred_Valid_matrix <- matrix(unname(unlist(sapply(out_tree,function(x) x[3]))), nrow=dim(X_valid)[1], ncol=ntree, byrow=FALSE)
    m <- matrix(unname(unlist(sapply(out_tree,function(x) x[4]))), nrow=dim(X_train)[1], ncol=ntree, byrow=FALSE)
    oob <- matrix(unname(unlist(sapply(out_tree,tail,1))), nrow=dim(X_train)[1], ncol=ntree, byrow=FALSE)
    
    myList <- list("oob"=oob,"m"=m,"pred_Train"=pred_Train_matrix,"pred_Test"=pred_Test_matrix,"pred_Valid"=pred_Valid_matrix)
    save(myList,file = "chenxinyu/realdata/mma/mma5/result5/BH_tmp.RData")
    return(myList)
  }
  
  
  to1 <- function(x) x/sum(x)
  
  get_scores <- function(ypred,ytest){
    ypred = as.vector(ypred) 
    ytest = as.vector(ytest) 
    score1 = Metrics::mse(ytest,ypred)
    score2 = Metrics::mae(ytest,ypred)
    return(c(score1,score2))
  }
  
  
  get_1and2step_scores <- function(X_train,y_train,X_test,y_test,X_valid,y_valid,boot,seed0,nodesize,mtry,ntree,varImp){
    n_train <- dim(X_train)[1]
    tmp_list <- get_pred_matrix(X_train,y_train,X_test,X_valid,boot,seed0,nodesize,mtry,ntree,varImp)
    m <- tmp_list$m
    oob <- tmp_list$oob
    pred_Train_matrix <- tmp_list$pred_Train
    pred_Test_matrix <- tmp_list$pred_Test
    pred_Valid_matrix <- tmp_list$pred_Valid
    
    w0 <- rep(1/ntree,ntree) # matrix
    g <- function(w){sum(w)-1}
    
    ################ equal weight rf #################
    score0 <- get_scores(rowMeans(pred_Test_matrix),y_test)
    ################ equal weight rf end #################
    
    ################ 1 step ##########################
    fun2 <- function(w){
      p_tt  <- m %*% w #m=(n_train,n_tree),p_tt=(n_train,1)
      error <- y_train - pred_Train_matrix %*% w
      norm_error <- norm(error,"2")
      c = norm_error**2 + 2 * sum(error ** 2 * p_tt)
      return(c)
    } 
    
    w_star1 <- matrix(Rsolnp::solnp(w0,fun2,eqfun = g,eqB = 0,LB = rep(0,ntree),
                                    UB = rep(1,ntree))$pars)
    
    y_pred_optimized1 <- pred_Test_matrix %*% w_star1
    score1 <- get_scores(y_pred_optimized1,y_test)
    ################ 1 step end #######################
    
    ################ 2 step ##########################
    Y.hat <- t(pred_Train_matrix) #(n_tree,n_train)
    C0 <- 2*Y.hat %*% t(Y.hat) + diag(ntree)*(10**(-8))
    y_pred_train <- rowMeans(pred_Train_matrix) 
    sigma.square.hat.C1 <- ((norm(y_train - y_pred_train,"2"))**2)/n_train
    d0 <- as.vector(-2*Y.hat %*% y_train + t(m) %*% matrix(sigma.square.hat.C1,nrow=n_train,ncol=1))
    lb0 <- matrix(0,nrow=ntree,ncol=1)
    ub0 <- matrix(1,nrow=ntree,ncol=1)
    Aeq0 <- matrix(1,nrow=1,ncol=ntree)
    beq0 <- matrix(1,nrow=1,ncol=1)
    
    w0_star <- matrix(pracma::quadprog(C=C0,d=d0,Aeq=Aeq0,beq=beq0,lb=lb0,ub=ub0)$xmin)
    pred_train_C1_step1 <- pred_Train_matrix %*% w0_star
    residual_step1 <- as.matrix(y_train - pred_train_C1_step1) #(n_train,1)
    ##step2:
    C0_ <- 2*Y.hat %*% t(Y.hat) + diag(ntree)*(10**(-8))
    d0_ <- as.vector(-2*Y.hat %*% y_train+t(m) %*% residual_step1**2)
    lb0_ <- matrix(0,nrow=ntree,ncol=1)
    ub0_ <- matrix(1,nrow=ntree,ncol=1)
    Aeq0_ <- matrix(1,nrow=1,ncol=ntree)
    beq0_ <- matrix(1,nrow=1,ncol=1)
    
    w_star2 <- matrix(pracma::quadprog(C=C0_,d=d0_,Aeq=Aeq0_,beq=beq0_,lb=lb0_,ub=ub0_)$xmin)
    y_pred_optimized2 <- pred_Test_matrix %*% w_star2
    score2 <- get_scores(y_pred_optimized2,y_test)
    ################ 2 step end ##########################
    
   
    return(c(score0,score1,score2))
  }
  
  
  filename = "./dataset/housing.csv"
  dataset_csv <- read.csv(filename,header = FALSE,sep = "")
  dataset <- data.matrix(dataset_csv)
  p <- dim(dataset)[2]-1
  #test_size = 0.3
  data_list <- scorecard::split_df(dataset_csv, ratio = c(0.5, 0.3, 0.2),name_dfs = c('train', 'test', 'valid'),seed = 2021)
  train <- data_list$train
  test <- data_list$test
  valid <- data_list$valid
  X_train <- model.matrix(V14 ~ ., train)[, -1]
  y_train <- train$V14
  X_test <- model.matrix(V14~ ., test)[, -1]
  y_test <- test$V14
  X_valid <- model.matrix(V14~ ., valid)[, -1]
  y_valid <- valid$V14
  
  n_train <-  length(y_train)
  n_test <- length(y_test)
  n_valid <- length(y_valid)
  
  n_tree <- 100
  node_size <- 5
  varImp_ <- get_var_importance(X_valid,y_valid,seed0=random_seed,p,n_tree,nodesize=node_size)
  
  scores <- get_1and2step_scores(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,X_valid=X_valid,y_valid=y_valid,boot=TRUE,seed0=random_seed,nodesize=node_size,mtry=floor(p/3),ntree=n_tree,varImp=varImp_)
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





