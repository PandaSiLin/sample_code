## XGBoost in Predict Fail/Pass
## This part encounters model training and tuning codes

library(tidyverse)
library(caret)
library(lime)
library(ggpubr)
library(keras)
library(lubridate)

############    Data Cleanup    ##################

source("i2_data_prep_augmented.R")
source("i4_data_prep.R")

input4 <- readxl::read_xlsx("input/input4.xlsx")

#imp_cols = intersect(colnames(df_final), colnames(df_final_old))
imp_cols <- c('mounted_high_frequency_fa_fr',	'mounted_high_frequency_impedance',	
              'mounted_high_frequency_resonant_fr',	'mounted_low_frequency_fa_fr',	
              'mounted_low_frequency_impedance',	'mounted_low_frequency_resonant_fr',	
              'initial_wedge_height',	'vout',	'wedge_stroke',	'c_fa',	'c_fr',	'c_imp', 
              'stb_high_freq_combined_value')
day_cnt = 30


# classifier
upbd = 1.20 #1.2636
lwbd = 1.10 #1.0270   
raw_df <- input4 %>% 
  mutate(
    fail_up = ifelse(stb_high_freq_combined_value >= upbd, "Fail", "Pass"),
    fail_lw = ifelse(stb_high_freq_combined_value <= lwbd, "Fail", "Pass")
  )


###########   Adding Time series model as a feature ----
if(T){
  # Train New LSTM model ----
  
  dat <- raw_df %>%
    group_by(batch_no) %>%
    summarise(
      mu = mean(stb_high_freq_combined_value)
    ) %>%
    ungroup()
  
  d=0.8
  N=nrow(dat)    # total rows
  m=floor(N*d)   # train size ind
  a = dat$mu  %>% exp() %>% exp()  # time series
  step = 5      # time step
  a = c(replicate(step, head(a, 1)), a)   # padding
  
  x = NULL
  y = NULL
  for(i in 1:N)
  {
    s = i-1+step
    x = rbind(x,a[i:s])
    y = rbind(y,a[s+1])
  }
  
  x.train = x[1:m, 1:2]
  y.train = y[1:m,]
  
  x.train = array(x.train, dim=c(m, step, 1)) # c(sample_size, time_step, features)
  x = array(x, dim=c(N, step,1))
  y.train = array(y.train, dim=c(m, 1, 1))
  
  ts_model = keras_model_sequential() %>%   
    layer_lstm(units=128, input_shape=c(step, 1), activation="relu") %>%  
    layer_dense(units=64, activation = "relu") %>%  
    layer_dense(units=16) %>%  
    layer_dense(units=1, activation = "linear")
  
  ts_model %>% compile(loss = 'mse',
                    #optimizer = 'RMSprop',
                    optimizer = 'adam',
                    metrics = list("mean_absolute_error")
  )
  
  ts_model %>% fit(x.train, y.train, 
                epochs=10, batch_size=100, shuffle = FALSE)
  
  ts_model %>% fit(
    x.train, y.train, 
    epochs=200, batch_size=100, 
    validation_split = 0.2,
    shuffle = FALSE,
    callbacks = callback_early_stopping(
      monitor = "val_mean_absolute_error",
      min_delta = 0.01,
      patience = 100,
      verbose = 0,
      mode = "min",
      baseline = NULL,
      restore_best_weights = FALSE
    )
  )
  
    
}else{
  # load old LSTM model
  ts_model <- load_model_hdf5("models/ts_model")
}

day_mean_obpf = cbind(dat[,c("batch_no")], "day_mean_obpf"=ts_model %>% predict(x) %>% log() %>% log())

raw_df <- raw_df %>% left_join(day_mean_obpf, by=c("batch_no"))


model_data <- raw_df %>% 
  mutate(
    fail_up = ifelse(stb_high_freq_combined_value >= upbd, "Fail", "Pass"),
    fail_lw = ifelse(stb_high_freq_combined_value <= lwbd, "Fail", "Pass")
  ) %>%
  select(all_of(imp_cols), fail_up, fail_lw)

# split train/test by 80/20 randome select
if(F){
  set.seed(123)
  tridx <- sample(1:nrow(raw_df), floor(0.8*nrow(raw_df)), replace = F)
}
# split train/test by time line
tridx <- 1:floor(0.8*nrow(model_data))


train <- model_data[tridx,]
test <- model_data[-tridx,]

train_x <- train %>% select(-fail_up,-fail_lw) %>% as.matrix()
train_y_up <- train$fail_up %>% as.factor()
train_y_lw <- train$fail_lw %>% as.factor()

test_x <- test %>% select(-fail_up, -fail_lw) %>% as.matrix()
test_y_up <- test$fail_up %>% as.factor()
test_y_lw <- test$fail_lw %>% as.factor()

input_x <- model_data %>% select(-fail_up,-fail_lw) %>% as.matrix()

############    Model Tuning    ##################
### Benchmark model (raw model) ----
if(F){
train_control <- caret::trainControl(
  method = "repeatedcv",
  number = 2,
  repeats = 4,
)

model_grid_up <- expand.grid(
  nrounds = 100,
  max_depth = 3,    
  eta = 0.02,
  gamma = 0.2,
  colsample_bytree = 0.7,
  min_child_weight = 4,
  subsample = 0.4
)

xgb_base_up_benchmark <- caret::train(
  x = train_x,
  y = train_y_up,
  trControl = train_control,
  tuneGrid = model_grid_up,
  method = "xgbTree",
  verbose = TRUE
  #metric = "CPNU",
  #maximize=FALSE
)


model_grid_lw <- expand.grid(
  nrounds = 100,
  max_depth = 3,    
  eta = 0.01,
  gamma = 0.1,
  colsample_bytree = 1,
  min_child_weight = 3,
  subsample = 0.7
)


xgb_base_lw_benchmark <- caret::train(
  x = train_x,
  y = train_y_lw,
  trControl = train_control,
  tuneGrid = model_grid_lw,
  method = "xgbTree",
  verbose = TRUE
)

if(F){
  ### Tune hyperparameters for fail_up ----
  
  tuneplot <- function(x, probs = .90, ylim = NA) {
    if (is.na(ylim)) {
      ggplot(x) +
        coord_cartesian(ylim = c(min(x$results$Accuracy), max(x$results$Accuracy))) +#c(quantile(x$results$CPNU, probs = probs), max(x$results$CPNU))) +
        theme_bw()
    } else{
      ggplot(x) +
        coord_cartesian(ylim = ylim) +#c(quantile(x$results$CPNU, probs = probs), max(x$results$CPNU))) +
        theme_bw()
    }
  }
  
  
  xgb1 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(300,1000,100),
      max_depth = seq(3,6,1),
      eta = 0.1,
      gamma = 0.01,
      colsample_bytree = 1,
      min_child_weight = 1,
      subsample = 1
    ),
    method = "xgbTree",
    verbose = TRUE
    #metric = "CPNU",
    #maximize=FALSE
  )
  tuneplot(xgb1)
  
  xgb2 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(100,500,50),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = 0,
      colsample_bytree = seq(0.5,1,0.1),
      min_child_weight = 2:5,
      subsample = 1
    ),
    method = "xgbTree",
    verbose = TRUE
    #metric = "CPNU",
    #maximize=FALSE
  )
  tuneplot(xgb2)
  
  xgb3 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(50,200,25),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = c(0, 0.05, 0.1, 0.2, 0.5),
      #colsample_bytree = 0.4,
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = 1
    ),
    method = "xgbTree",
    verbose = TRUE
    #metric = "CPNU",
    #maximize=FALSE
  )
  tuneplot(xgb3)
  
  xgb4 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(50,500,25),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = xgb3$bestTune$gamma,
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = seq(0.2,1,0.2)
    ),
    method = "xgbTree",
    verbose = TRUE
    #metric = "CPNU",
    #maximize=FALSE
  )
  tuneplot(xgb4)
  
  xgb5 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(100,1000,100),
      max_depth = xgb1$bestTune$max_depth,
      eta = c(0.01, 0.05, 0.1, 0.2, 0.5),
      gamma = xgb3$bestTune$gamma,
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = xgb4$bestTune$subsample
    ),
    method = "xgbTree",
    verbose = TRUE
    #metric = "CPNU",
    #maximize=FALSE
  )
  tuneplot(xgb5)
  
  xgb5$bestTune
  
  xgb6 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(100,500,50),
      max_depth = 3,    
      eta = c(0.02, 0.04, 0.06, 0.1, 0.2),
      gamma = 0.2,
      colsample_bytree = 0.7,
      min_child_weight = 4,
      subsample = 0.4
    ),
    method = "xgbTree",
    verbose = TRUE
    #metric = "CPNU",
    #maximize=FALSE
  )
  tuneplot(xgb6)
  
  xgb6$bestTune
  
  ### Tune hyperparameters for fail_lw ----
  xgb1 <- caret::train(
    x = train_x,
    y = train_y_lw,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(300,1000,100),
      max_depth = seq(3,6,1),
      eta = 0.1,
      gamma = 0.01,
      colsample_bytree = 0.6,
      min_child_weight = 1,
      subsample = 0.5
    ),
    method = "xgbTree",
    verbose = TRUE
    #metric = "CPNU",
    #maximize=FALSE
  )
  tuneplot(xgb1)
  
  
  xgb2 <- caret::train(
    x = train_x,
    y = train_y_lw,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(100,500,100),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = 0,
      colsample_bytree = seq(0.2,1,0.2),
      min_child_weight = 2:5,
      subsample = 1
    ),
    method = "xgbTree",
    verbose = TRUE
    #metric = "CPNU",
    #maximize=FALSE
  )
  tuneplot(xgb2)
  
  xgb3 <- caret::train(
    x = train_x,
    y = train_y_lw,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(100,500,100),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = c(0, 0.05, 0.1, 0.2, 0.5),
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = 1
    ),
    method = "xgbTree",
    verbose = TRUE
    #metric = "CPNU",
    #maximize=FALSE
  )
  tuneplot(xgb3)
  
  xgb4 <- caret::train(
    x = train_x,
    y = train_y_lw,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(100,100,200),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = xgb3$bestTune$gamma,
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = seq(0.1,1,0.1)
    ),
    method = "xgbTree",
    verbose = TRUE
    #metric = "CPNU",
    #maximize=FALSE
  )
  tuneplot(xgb4)
  
  xgb5 <- caret::train(
    x = train_x,
    y = train_y_lw,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(100,1000,100),
      max_depth = xgb1$bestTune$max_depth,
      eta = c(0.01, 0.05, 0.1, 0.2, 0.5),
      gamma = xgb3$bestTune$gamma,
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = xgb4$bestTune$subsample
    ),
    method = "xgbTree",
    verbose = TRUE
    #metric = "CPNU",
    #maximize=FALSE
  )
  tuneplot(xgb5)
  xgb5$bestTune
}
}

### Models I and II (higer cost on the False Positive) ----
costSummary <- function(data, lev = NULL, model = NULL)  {
  
  if(length(levels(data$obs)) > 2) {
    stop(paste("Your outcome has", length(levels(data$obs)), 
               "levels. `costSummary` function isn't appropriate.", 
               call. = FALSE))
  }
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) {
    stop("Levels of observed and predicted data do not match.", 
         call. = FALSE)
  }
  
  cutoff_vec <- seq(0.05,0.95,0.01)
  len <- length(cutoff_vec)
  pred <- data[, lev[1]]
  ref <- ifelse(data$obs == lev[1], 1, 0)
  cpnu <- sapply(
    1:len,
    function(i){
      y_hat <- (pred > cutoff_vec[i])*1
      return( (13*sum(ref*(1-y_hat)))/(sum(1-y_hat)+1) )
    }
  )
  c( CPNU = min(cpnu) )
}

train_control <- caret::trainControl(
  method = "repeatedcv",
  number = 2,
  repeats = 4,
  #index = cv_f2r4,
  summaryFunction = costSummary, ### Look here
  sampling = "smote",
  verboseIter = TRUE, # no training log
  allowParallel = TRUE,
  classProbs = TRUE
)

model_grid_up <- expand.grid(
  nrounds = 300,
  max_depth = 3,    
  eta = 0.02,
  gamma = 0.5,
  colsample_bytree = 0.7,
  min_child_weight = 5,
  subsample = 0.6
)

xgb_base_up <- caret::train(
  x = train_x,
  y = train_y_up,
  trControl = train_control,
  tuneGrid = model_grid_up,
  method = "xgbTree",
  verbose = TRUE,
  metric = "CPNU",
  maximize=FALSE
)


model_grid_lw <- expand.grid(
  nrounds = 300,
  max_depth = 3,    
  eta = 0.01,
  gamma = 0.1,
  colsample_bytree = 0.4,
  min_child_weight = 5,
  subsample = 0.2
)


xgb_base_lw <- caret::train(
  x = train_x,
  y = train_y_lw,
  trControl = train_control,
  tuneGrid = model_grid_lw,
  method = "xgbTree",
  verbose = TRUE,
  metric = "CPNU",
  maximize=FALSE
)

if(F){
  ### Tune hyperparameters for fail_up ----
  
  tuneplot <- function(x, probs = .90, ylim = NA) {
    if (is.na(ylim)) {
      ggplot(x) +
        coord_cartesian(ylim = c(min(x$results$CPNU), max(x$results$CPNU))) +#c(quantile(x$results$CPNU, probs = probs), max(x$results$CPNU))) +
        theme_bw()
    } else{
      ggplot(x) +
        coord_cartesian(ylim = ylim) +#c(quantile(x$results$CPNU, probs = probs), max(x$results$CPNU))) +
        theme_bw()
    }
  }
  
  
  xgb1 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(300,1000,100),
      max_depth = seq(3,6,1),
      eta = 0.1,
      gamma = 0.01,
      colsample_bytree = 0.6,
      min_child_weight = 1,
      subsample = 0.5
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb1)
  
  xgb2 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(100,500,50),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = 0,
      colsample_bytree = seq(0.5,1,0.1),
      min_child_weight = 2:5,
      subsample = 1
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb2)
  
  xgb3 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(50,200,25),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = c(0, 0.05, 0.1, 0.2, 0.5),
      colsample_bytree = 0.4,
      #colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = 1
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb3)
  
  xgb4 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(50,500,25),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = xgb3$bestTune$gamma,
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = seq(0.2,1,0.2)
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb4)
  
  xgb5 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(100,1000,100),
      max_depth = xgb1$bestTune$max_depth,
      eta = c(0.01, 0.05, 0.1, 0.2, 0.5),
      gamma = xgb3$bestTune$gamma,
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = xgb4$bestTune$subsample
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb5)
  
  xgb5$bestTune
  
  xgb6 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(100,1000,200),
      max_depth = 3,    
      eta = c(0.02, 0.04, 0.06, 0.1, 0.2),
      gamma = 0.5,
      colsample_bytree = 0.7,
      min_child_weight = 5,
      subsample = 0.6
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb6)
  
  xgb6$bestTune
  
  ### Tune hyperparameters for fail_lw ----
  xgb1 <- caret::train(
    x = train_x,
    y = train_y_lw,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(300,1000,100),
      max_depth = seq(3,6,1),
      eta = 0.1,
      gamma = 0.01,
      colsample_bytree = 0.6,
      min_child_weight = 1,
      subsample = 0.5
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb1)
  
  
  xgb2 <- caret::train(
    x = train_x,
    y = train_y_lw,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(100,500,100),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = 0,
      colsample_bytree = seq(0.2,1,0.2),
      min_child_weight = 2:5,
      subsample = 1
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb2)
  
  xgb3 <- caret::train(
    x = train_x,
    y = train_y_lw,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(100,500,100),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = c(0, 0.05, 0.1, 0.2, 0.5),
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = 1
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb3)
  
  xgb4 <- caret::train(
    x = train_x,
    y = train_y_lw,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(100,100,200),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = xgb3$bestTune$gamma,
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = seq(0.1,1,0.1)
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb4)
  
  xgb5 <- caret::train(
    x = train_x,
    y = train_y_lw,
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = seq(100,1000,100),
      max_depth = xgb1$bestTune$max_depth,
      eta = c(0.01, 0.05, 0.1, 0.2, 0.5),
      gamma = xgb3$bestTune$gamma,
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = xgb4$bestTune$subsample
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb5)
  xgb5$bestTune
}

### Models III and IV (higher cost on the False Negative) ----
costSummary_FN <- function(data, lev = NULL, model = NULL)  {
  
  if(length(levels(data$obs)) > 2) {
    stop(paste("Your outcome has", length(levels(data$obs)), 
               "levels. `costSummary` function isn't appropriate.", 
               call. = FALSE))
  }
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) {
    stop("Levels of observed and predicted data do not match.", 
         call. = FALSE)
  }
  
  cutoff_vec <- seq(0.05,0.95,0.01)
  len <- length(cutoff_vec)
  pred <- data[, lev[1]]
  ref <- ifelse(data$obs == lev[1], 1, 0)
  cpnu <- sapply(
    1:len,
    function(i){
      y_hat <- (pred > cutoff_vec[i])*1
      return( (1/13*sum(ref*(1-y_hat)))/(sum(1-y_hat)+1) )
    }
  )
  c( CPNU = min(cpnu) )
}

train_control2 <- caret::trainControl(
  method = "repeatedcv",
  number = 2,
  repeats = 4,
  #index = cv_f2r4,
  summaryFunction = costSummary_FN, ### Look here
  sampling = "smote",
  verboseIter = TRUE, # no training log
  allowParallel = TRUE,
  classProbs = TRUE
)

model_grid_up2 <- expand.grid(
  nrounds = 200,
  max_depth = 3,    
  eta = 0.01,
  gamma = 0,
  colsample_bytree = 0.2,
  min_child_weight = 2,
  subsample = 0.2
)

xgb_base_up2 <- caret::train(
  x = train_x,
  y = train_y_up,
  trControl = train_control2,
  tuneGrid = model_grid_up2,
  method = "xgbTree",
  verbose = TRUE,
  metric = "CPNU",
  maximize=FALSE
)

model_grid_lw2 <- expand.grid(
  nrounds = 700,
  max_depth = 3,    
  eta = 0.01,
  gamma = 0,
  colsample_bytree = 0.4,
  min_child_weight = 2,
  subsample = 0.6
)


xgb_base_lw2 <- caret::train(
  x = train_x,
  y = train_y_lw,
  trControl = train_control2,
  tuneGrid = model_grid_lw2,
  method = "xgbTree",
  verbose = TRUE,
  metric = "CPNU",
  maximize=FALSE
)


if(F){
  ### Tune hyperparameters for fail_up ----
  
  tuneplot <- function(x, probs = .90, ylim = NA) {
    if (is.na(ylim)) {
      ggplot(x) +
        coord_cartesian(ylim = c(min(x$results$CPNU), max(x$results$CPNU))) +#c(quantile(x$results$CPNU, probs = probs), max(x$results$CPNU))) +
        theme_bw()
    } else{
      ggplot(x) +
        coord_cartesian(ylim = ylim) +#c(quantile(x$results$CPNU, probs = probs), max(x$results$CPNU))) +
        theme_bw()
    }
  }
  
  
  xgb1 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control2,
    tuneGrid = expand.grid(
      nrounds = seq(300,1000,100),
      max_depth = seq(3,6,1),
      eta = 0.1,
      gamma = 0.01,
      colsample_bytree = 0.6,
      min_child_weight = 1,
      subsample = 0.5
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb1)
  
  xgb2 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control2,
    tuneGrid = expand.grid(
      nrounds = seq(100,500,50),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = 0,
      colsample_bytree = seq(0.2,1,0.1),
      min_child_weight = 2:5,
      subsample = 1
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb2)
  
  xgb3 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control2,
    tuneGrid = expand.grid(
      nrounds = seq(50,200,25),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = c(0, 0.01, 0.05, 0.1, 0.2, 0.5),
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = 1
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb3)
  
  xgb4 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control2,
    tuneGrid = expand.grid(
      nrounds = seq(50,500,25),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = xgb3$bestTune$gamma,
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = seq(0.2,1,0.2)
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb4)
  
  xgb5 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control2,
    tuneGrid = expand.grid(
      nrounds = seq(100,1000,100),
      max_depth = xgb1$bestTune$max_depth,
      eta = c(0.01, 0.05, 0.1, 0.2, 0.5),
      gamma = xgb3$bestTune$gamma,
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = xgb4$bestTune$subsample
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb5)
  
  xgb5$bestTune
  
  xgb6 <- caret::train(
    x = train_x,
    y = train_y_up,
    trControl = train_control2,
    tuneGrid = expand.grid(
      nrounds = seq(100,1000,200),
      max_depth = 3,    
      eta = c(0.02, 0.04, 0.06, 0.1, 0.2),
      gamma = 0,
      colsample_bytree = 0.5,
      min_child_weight = 2,
      subsample = 1
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb6)
  
  xgb6$bestTune
  
  ### Tune hyperparameters for fail_lw ----
  xgb1 <- caret::train(
    x = train_x,
    y = train_y_lw,
    trControl = train_control2,
    tuneGrid = expand.grid(
      nrounds = seq(300,1000,100),
      max_depth = seq(3,6,1),
      eta = 0.1,
      gamma = 0.01,
      colsample_bytree = 0.6,
      min_child_weight = 1,
      subsample = 0.5
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb1)
  
  
  xgb2 <- caret::train(
    x = train_x,
    y = train_y_lw,
    trControl = train_control2,
    tuneGrid = expand.grid(
      nrounds = seq(100,500,100),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = 0,
      colsample_bytree = seq(0.2,1,0.2),
      min_child_weight = 2:5,
      subsample = 1
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb2)
  
  xgb3 <- caret::train(
    x = train_x,
    y = train_y_lw,
    trControl = train_control2,
    tuneGrid = expand.grid(
      nrounds = seq(100,500,100),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = c(0, 0.05, 0.1, 0.2, 0.5),
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = 1
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb3)
  
  xgb4 <- caret::train(
    x = train_x,
    y = train_y_lw,
    trControl = train_control2,
    tuneGrid = expand.grid(
      nrounds = seq(100,100,200),
      max_depth = xgb1$bestTune$max_depth,
      eta = 0.1,
      gamma = xgb3$bestTune$gamma,
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = seq(0.1,1,0.1)
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb4)
  
  xgb5 <- caret::train(
    x = train_x,
    y = train_y_lw,
    trControl = train_control2,
    tuneGrid = expand.grid(
      nrounds = seq(100,1000,100),
      max_depth = xgb1$bestTune$max_depth,
      eta = c(0.01, 0.05, 0.1, 0.2, 0.5),
      gamma = xgb3$bestTune$gamma,
      colsample_bytree = xgb2$bestTune$colsample_bytree,
      min_child_weight = xgb2$bestTune$min_child_weight,
      subsample = xgb4$bestTune$subsample
    ),
    method = "xgbTree",
    verbose = TRUE,
    metric = "CPNU",
    maximize=FALSE
  )
  tuneplot(xgb5)
  xgb5$bestTune
}
if(F){
  # Tuning CPNU leverage
  acc_df <- NULL
  for(cost in 10:15){
    costSummary2 <- function(data, lev = NULL, model = NULL)  {
      
      if(length(levels(data$obs)) > 2) {
        stop(paste("Your outcome has", length(levels(data$obs)), 
                   "levels. `costSummary` function isn't appropriate.", 
                   call. = FALSE))
      }
      if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) {
        stop("Levels of observed and predicted data do not match.", 
             call. = FALSE)
      }
      
      cutoff_vec <- seq(0.05,0.95,0.01)
      len <- length(cutoff_vec)
      pred <- data[, lev[2]]
      ref <- ifelse(data$obs == lev[2], 1, 0)
      cpnu <- sapply(
        1:len,
        function(i){
          y_hat <- (pred > cutoff_vec[i])*1
          return( (cost*sum(ref*(1-y_hat)))/(sum(1-y_hat)+1) )
        }
      )
      c( CPNU = min(cpnu) )
    }
    
    train_control <- caret::trainControl(
      method = "repeatedcv",
      number = 2,
      repeats = 4,
      index = cv_f2r4,
      summaryFunction = costSummary_FN, ### Look here
      sampling = "smote",
      verboseIter = TRUE, # no training log
      allowParallel = TRUE,
      classProbs = TRUE
    )
    
    xgb_base_up <- caret::train(
      x = train_x,
      y = train_y_up,
      trControl = train_control2,
      tuneGrid = model_grid_up,
      method = "xgbTree",
      verbose = TRUE,
      metric = "CPNU",
      maximize=FALSE
    )
    
    xgb_base_lw <- caret::train(
      x = train_x,
      y = train_y_lw,
      trControl = train_control2,
      tuneGrid = model_grid_lw,
      method = "xgbTree",
      verbose = TRUE,
      metric = "CPNU",
      maximize=FALSE
    )
    
    acc_df <- rbind(
      acc_df, 
      data.frame(
        acc_up = sum(train_y_up == predict(xgb_base_up, train_x))/nrow(train_x),
        acc_lw = sum(train_y_lw == predict(xgb_base_lw, train_x))/nrow(train_x),
        cost = cost)
    )
    
  }
}



############    Save Model    ##################

if(F){
saveRDS(raw_df, "raw_df.rds")
keras::save_model_hdf5(ts_model,"deployment/models/ts_model")
xgboost::xgb.save(xgb_base_lw$finalModel, "deployment/models/xgb_base_lw.model")
xgboost::xgb.save(xgb_base_up$finalModel, "deployment/models/xgb_base_up.model")
xgboost::xgb.save(xgb_base_lw2$finalModel, "deployment/models/xgb_base_lw2.model")
xgboost::xgb.save(xgb_base_up2$finalModel, "deployment/models/xgb_base_up2.model")
}

save.image("deployment/models/consolidated.RData")
keras::save_model_tf(ts_model,"deployment/models/ts_model/")
saveRDS(xgb_base_up,"deployment/models/xgb_base_up.rds")
saveRDS(xgb_base_lw,"deployment/models/xgb_base_lw.rds")
saveRDS(xgb_base_up2,"deployment/models/xgb_base_up2.rds")
saveRDS(xgb_base_lw2,"deployment/models/xgb_base_lw2.rds")
