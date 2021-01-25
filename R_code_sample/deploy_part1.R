## XGBoost in Classification 
## Part 1 is gener

library(tidyverse)
library(caret)
library(lime)
library(ggpubr)
library(keras)
library(lubridate)
library(SHAPforxgboost)
library(png)

#load("models/consolidated.RData")

raw_df <- readRDS("models/raw_df.rds")   # bench data

input1 <- readxl::read_xlsx("input/input1.xlsx") # New batch to predict (batch of 30) 
input4 <- readxl::read_xlsx("input/input4.xlsx") # Historical data


ts_model <- keras::load_model_tf("models/ts_model/")   

xgb_base_up <- readRDS("models/xgb_base_up.rds")
xgb_base_lw <- readRDS("models/xgb_base_lw.rds")
#xgb_base_up2 <- readRDS("models/xgb_base_up2.rds")
#xgb_base_lw2 <- readRDS("models/xgb_base_lw2.rds")

upbd = 1.20 #1.2636
lwbd = 1.10 #1.0270  


###############         Data Clean Up     #######################

if(F){ # first time assign batch no to old data.
  
input4 <- input4 %>%
  mutate(batch_no = head(rep(0:floor(nrow(input4)/30+1),each=30), nrow(input4)))
write.csv(input4, "input4.csv", row.names = F)

}

input_data <- input1 %>%
  mutate(
    c_fa = (c1_fa + c2_fa + c3_fa + c4_fa)/4,
    c_fr = (c1_fr + c2_fr + c3_fr + c4_fr)/4,
    c_imp = (c1_imp + c2_imp + c3_imp + c4_imp)/4
  ) %>%
  select(-matches("^c[1-4].*")) %>%
  mutate(batch_no = max(input4$batch_no)+1)

#raw_df is data feed in [batch_no, stb_high_freq_combined_value]

###############         LSTM feature prediction     #######################

lstm_tf <- function(data, timestep=5){
  
  dat <- data %>%
    group_by(batch_no) %>%
    summarise(
      mu = mean(stb_high_freq_combined_value)
    ) %>%
    ungroup()
  
  N=nrow(dat)    # total rows
  a = dat$mu  %>% exp() %>%exp()
  
  step = timestep      # time step
  a = c(replicate(step, head(a, 1)), a)   # padding
  
  x = NULL
  #y = NULL
  for(i in 1:(N+1))
  {
    s = i-1+step
    x = rbind(x,a[i:s])
    #y = rbind(y,a[s+1])
  }
  
  return(list(
    "batch_no" = dat[,c("batch_no")]+1,
    "data" = array(x, dim=c(N, step, 1))   # (rows, steps, features)
  ))
  
}

day_mean_obpf = cbind(
  "day_mean_obpf" = log(log(predict(ts_model, lstm_tf(input4, 5)$data))),
  "batch_no"=lstm_tf(input4, 5)$batch_no)

# backfill 1 with mean
day_mean_obpf <- rbind(c(day_mean_obpf[1,1], 1), day_mean_obpf)

###############         XGBoost Model Data prep     #######################

model_input <- input_data %>%
  left_join(day_mean_obpf, by="batch_no") %>%
  mutate(
    fail_up = ifelse(stb_high_freq_combined_value >= upbd, "Fail", "Pass"),
    fail_lw = ifelse(stb_high_freq_combined_value <= lwbd, "Fail", "Pass")
  )



imp_cols <- c("mounted_high_frequency_fa_fr", "mounted_high_frequency_impedance",
              "mounted_high_frequency_resonant_fr",
              "mounted_low_frequency_fa_fr", "mounted_low_frequency_impedance",
              "mounted_low_frequency_resonant_fr" ,
              "initial_wedge_height",  "vout" ,                        
              "wedge_stroke", "c_fa", "c_fr", "c_imp",
              "day_mean_obpf"    # exclusively for UC5 i2
              )

xgb_data <- model_input %>% 
  select(all_of(imp_cols)) %>% 
  as.matrix()

###############         XGBoost Model Pred    #######################

output <-  cbind(
  model_input,
  "pred_up" = predict(xgb_base_up, as.matrix(xgb_data)),
  "pred_lw" = predict(xgb_base_lw, as.matrix(xgb_data))
) %>%
  mutate(
    pred_result = case_when(
      pred_up == "Fail" & pred_lw == "Fail" ~ "Error, Fail both",
      pred_up == "Fail" & pred_lw == "Pass" ~ "Fail Up",
      pred_up == "Pass" & pred_lw == "Fail" ~ "Fail Low",
      pred_up == "Pass" & pred_lw == "Pass" ~ "Pass"
    ),
    actual_result = case_when(
      fail_up == "Fail" & fail_lw == "Fail" ~ "Error, Fail both",
      fail_up == "Fail" & fail_lw == "Pass" ~ "Fail Up",
      fail_up == "Pass" & fail_lw == "Fail" ~ "Fail Low",
      fail_up == "Pass" & fail_lw == "Pass" ~ "Pass"
    ))
  

shap_values_up = predict(xgb_base_up$finalModel, xgb_data, 
                      predcontrib = TRUE, approxcontrib = FALSE) %>%
  as.data.frame()

shap_values_lw = predict(xgb_base_lw$finalModel, xgb_data, 
                         predcontrib = TRUE, approxcontrib = FALSE) %>%
  as.data.frame()


output1 <- shap_values_up %>%
  cbind(
    model_input %>%
      select(station_complete_time, serial_longurl, rt_bom_serial, 
             mounted_station_complete_time, batch_no)
  ) %>%
  mutate(
    predicted_result = output$pred_result,
    actual_result = output$actual_result
  )

output2 <- shap_values_lw %>%
  cbind(
    model_input %>%
      select(station_complete_time, serial_longurl, rt_bom_serial, 
             mounted_station_complete_time, batch_no)
  ) %>%
  mutate(
    predicted_result = output$pred_result,
    actual_result = output$actual_result
  )


write.csv(
  rbind(read.csv("output/output1.csv"), output1),
  "output/output1.csv", row.names = F
)

write.csv(
  rbind(read.csv("output/output2.csv"), output2),
  "output/output2.csv", row.names = F
)


# update historical data set
input4 <- rbind(
  input4,
  input_data
  )

writexl::write_xlsx(input4, "input4.xlsx")


# individual SHAP Plot (archived)----
if(F){
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
new_generate_shap_indv <- function(case=1, bxplt_window=5){ 
  #Other variables obtained from Global Env. : 
      # output, xgb_base_up, xgb_base_lw, Input1, Input4

  if(case> nrow(output)){
    stop("Error: case is beyond output range.")
  }
  
  grp = output$batch_no[1]
  act_up = output[case,]$fail_up
  act_lw = output[case,]$fail_lw
  pred_up = output[case,]$pred_up
  pred_lw = output[case,]$pred_lw

  shap_input_x <- output[case,] %>%
    select(all_of(imp_cols)) %>% 
    as.matrix()
  
  
  if(act_up == "Pass" & act_lw == "Pass"){
    return(print("Case does not fail on either side. No graph being displayed"))
    
  }else if(act_up == "Fail" & act_lw =="Fail"){
    return(print("Case failed on both side. Data issue."))
    
  }else if(act_up == "Fail"){
    shap_model = xgb_base_up$finalModel
    
  }else if(act_lw == "Fail"){
    shap_model = xgb_base_lw$finalModel
  }
  
  # get shap value
  #shap_values = shap.values(xgb_model = shap_model, X_train = input_data)
  
  shap_values = predict(shap_model, shap_input_x, 
                        predcontrib = TRUE, approxcontrib = FALSE) %>%
    as.data.frame() 
  
  
  # data prep: shap value plot

  shap_value_indv <- shap_values %>%
    #filter(row.names(.) == case) %>%
    select(-BIAS) %>%
    gather(feature, value) %>%
    mutate(
      feature = factor(feature, ordered = is.ordered(value)),
      col_ind = ifelse(value<=0,"blue","red")
    ) %>%
    filter(feature != "day_mean_obpf") 
  
  # sort the features by values

  lvls = shap_value_indv[order(abs(shap_value_indv$value)),]$feature
  
  shap_plot_dat <- shap_value_indv %>%
    mutate(x.label = paste("<span style = 'color: ",
                           ifelse(feature %in% as.character(lvls[8:12]), "black", "grey"),
                           ";'>",
                           feature,
                           "</span>", sep = "")
    ) %>%
    mutate(
      col_ind = ifelse(grepl("grey",x.label), paste0("light ",col_ind, sep=""), col_ind)
    )
  
  lvls2 = shap_plot_dat[order(abs(shap_plot_dat$value)),]$x.label
  shap_plot_dat$x.label <- factor(shap_plot_dat$x.label, levels=lvls2, ordered=TRUE)
  
  if(act_up == "Fail"){
    # shap value plot (up)
    shap_plt <- ggplot(shap_plot_dat) +
      geom_bar(aes(x=x.label, y=value, fill = col_ind), stat = "identity") +
      coord_flip() +
      theme_classic() +
      theme(legend.position = "none") +
      ggtitle(paste0("Machine ID: ", raw_df[case,c("machine_id")], "\n", "Case ",case," \npredict up: ", pred_up, "\nactual up: ",act_up,  sep="")) +
      scale_fill_manual(breaks = c("blue", "red", "light blue", "light red"), 
                        values = c(rgb(red=30,blue=228,green=136,max=255),
                                   rgb(red=255,blue=87,green=13,max=255),
                                   "grey96","grey96")) +
      theme(axis.title.x=element_blank(),
            axis.text.x=element_blank(),
            axis.ticks.x=element_blank(),
            axis.text.y=ggtext::element_markdown()) +
      labs(x="feature")
      
  }else if(act_lw == "Fail"){
    # shap value plot (lw) 
    shap_plt <- ggplot(data=shap_plot_dat) +
      geom_bar(aes(x=x.label, y=value, fill = col_ind), stat = "identity") +
      coord_flip() +
      theme_classic() +
      theme(legend.position = "none") +
      ggtitle(paste0("Machine ID: ", raw_df[case,c("machine_id")], "\n","Case ",case," \npredict lw: ", pred_lw, "\nactual lw: ",act_lw,  sep="")) +
      scale_fill_manual(breaks = c("blue", "red", "light blue", "light red"), 
                        values = c(rgb(red=30,blue=228,green=136,max=255),
                                   rgb(red=255,blue=87,green=13,max=255),
                                   "grey96","grey96")) +
      theme(axis.title.x=element_blank(),
            axis.text.x=element_blank(),
            axis.ticks.x=element_blank(),
            axis.text.y=ggtext::element_markdown()) +
      labs(x="feature")
  }
  
  # Data prep: box plot 
  get_desc <- function(x){
    map(x, ~list(
      min = min(.x),
      max = max(.x),
      mean = mean(.x),
      sd = sd(.x)
    ))
  }
  trimmer <- function(x, lw=0.05, up=0.95){
    if(ncol(x) != length(levels(lvls))){
      add_cols = x[,setdiff(colnames(x), levels(lvls))]
      x = x[,levels(lvls)]
    }
    
    x = as.data.frame(
      sapply(x, function(x, a=0.005, b=0.995) pmax(quantile(x,a), pmin(x, quantile(x,b))))
      )
    
    
    if(exists("add_cols")){
      x = cbind(x, add_cols)
    }
    return(x)
    
  }
  minmaxscaler2 <- function(x, desc) {
    if(ncol(x) != length(names(desc))){
      add_cols = x %>% select(setdiff(colnames(x), names(desc)))
      x = x[,names(desc)]
    }
    x = map2_dfc(x, desc, ~(.x - .y$min)/(.y$max - .y$min))
    #x = map2_dfc(x, desc, ~(.x - .y$mean)/(.y$sd))
    if(exists("add_cols")){
      x = cbind(x, add_cols)
    }
    return(x)
  }
  
  
  input4_w_day_mean_obpf <- input4 %>% 
    left_join(day_mean_obpf, by="batch_no")
  
  bplt_grp_dat <- trimmer(input4_w_day_mean_obpf) %>%
    filter(batch_no %in% c(grp-bxplt_window:1+1)) %>%
    select(all_of(lvls))
  
  #desc <- get_desc(bplt_grp_dat)
  desc <- get_desc(select(trimmer(input4_w_day_mean_obpf), all_of(lvls)))
  
  # red points on case
  norm_test_x <- trimmer(output) %>%
    filter(row_number() == case) %>%
    minmaxscaler2(desc) %>%
    select(all_of(lvls)) %>%
    gather(feature, norm_value) %>%
    mutate(
      cat = "case_value",
      feature = factor(feature, levels = lvls, ordered = TRUE)
    ) %>%
    rbind({
      trimmer(output) %>%
        minmaxscaler2(desc) %>%
        select(all_of(lvls)) %>%
        #map_dfc(.,~c(mean(.x), median(.x))) %>%
        #mutate(cat = c("hist_mean","hist_median")) %>%
        map_dfc(.,~c(median(.x))) %>%
        mutate(cat = c("hist_median")) %>%
        gather(feature, norm_value, -cat)
    }) %>%
    mutate(norm_value = round(norm_value,2))
  
  # text label
  text_label <- trimmer(output) %>%
    filter(row_number() == case) %>%
    select(all_of(lvls)) %>%
    gather(feature, actual_value) %>%
    mutate(
      cat = "case_value",
      feature = factor(feature, levels = lvls, ordered = TRUE)
    ) %>%
    rbind({
      trimmer(output) %>%
        select(all_of(lvls)) %>%
        # map_dfc(.,~c(mean(.x), median(.x))) %>%
        # mutate(cat = c("hist_mean","hist_median")) %>%
        map_dfc(.,~c(median(.x))) %>%
        mutate(cat = c("hist_median")) %>%
        gather(feature, actual_value, -cat)
    }) %>%
    mutate(actual_value = round(actual_value,1)) %>%
    mutate(col_ind = ifelse(feature %in% as.character(lvls[8:12]), cat, "light grey"))
  
  norm_test_x <- left_join(norm_test_x, text_label,by=c("feature", "cat")) %>%
    mutate(feature = factor(feature, levels=lvls, ordered = TRUE))
  
  # box plot with group data
  bplt_dat <- bplt_grp_dat %>%
    minmaxscaler2(desc) %>% # scaled
    gather(feature, value) %>%
    mutate(
      feature = factor(feature, levels = lvls, ordered = TRUE),
      value = round(value,1)
      ) %>%
    mutate(col_ind = ifelse(feature %in% as.character(lvls[8:12]), "grey", "light grey"))
  
  # box plot
  box_plt <- ggplot(data = bplt_dat) +
    #geom_boxplot(aes(x=feature, y=value), color="grey") +
    geom_boxplot(aes(x=feature, y=value, color=col_ind)) +
    geom_text(aes(x=feature, y=norm_value, label=actual_value, color=col_ind), data=norm_test_x, size=4.5, position = position_dodge(width = .75), hjust=-0.1) + 
    geom_point(aes(x=feature, y=norm_value, color=col_ind, shape=cat), data = norm_test_x) +
    scale_colour_manual(
      breaks = c("case_value", "hist_median", "grey", "light grey"), 
      values = c("#FF6700", "#696969", "grey", "grey96")
    ) +
    coord_flip() +
    theme_classic() +
    theme(axis.text.x = element_text(angle = 90)) +
    ggtitle(paste0("\n\nQuantiles in ",bxplt_window," group(s)\nOBPF: ",input4[case,]$stb_high_freq_combined_value))+
    theme(
      legend.position = "none",
      #legend.box = "horizontal",
      legend.title = element_blank(),
      axis.title.x=element_blank(),
      axis.text.x=element_blank(),
      axis.ticks.x=element_blank(),
      axis.title.y=element_blank(),
      axis.text.y=element_blank()
      #axis.ticks.y=element_blank()
    )

  if(F){
  # include shap bar -- WIP
  write.csv(shap_values %>% 
              filter(row.names(.) == case) %>%
              mutate(caseid = case),
            "deployment/shap_value.csv", row.names = F)
  write.csv(trimmer(output) %>% 
              filter(caseid == case) %>%
              select(all_of(lvls)), 
            "deployment/model_data.csv", row.names = F)
  
  reticulate::py_run_file('deployment/shap_plt_generate.py')
  
  img <- png::readPNG("deployment/case.png")
  shap_bar <- ggplot() + background_image(img)
  }
  
  
  # 3-in-1 plot
  figure <- ggpubr::ggarrange(shap_plt, box_plt, 
                             ncol = 2, nrow = 1)
  
  # layout <- matrix(c(rep(1,5), 2, 2, 3, 3, 3, 2, 2, 3, 3, 3), nrow = 3, byrow = TRUE)
  # figure <- multiplot(plotlist = list(shap_bar, shap_plt, box_plt), layout = layout)


  
  return(figure)
}

new_generate_shap_indv(6)

batch_shap_trend_plot <- function(case=1,  bxplt_window=20){
  
  grp = floor(case/30) + 1
  act_up = output[output$caseid==case,]$fail_up
  act_lw = output[output$caseid==case,]$fail_lw
  
  if(act_up == "Pass" & act_lw == "Pass"){
    return(print("Case does not fail on either side. No graph being displayed"))
    
  }else if(act_up == "Fail" & act_lw =="Fail"){
    return(print("Case failed on both side. Data issue."))
    
  }else if(act_up == "Fail"){
    #shap_model = xgb_base_up$finalModel
    shap_model = xgb_base_up$finalModel
    
  }else if(act_lw == "Fail"){
    #shap_model = xgb_base_lw$finalModel
    shap_model = xgb_base_lw$finalModel
  }
  
  
  shap_values = predict(shap_model, as.matrix(input_x), 
                        predcontrib = TRUE, approxcontrib = FALSE) %>%
    as.data.frame() %>%
    select(-BIAS)
  
  # shap trend plot
  
  trend_plot <- shap_values %>%
    cbind("batch_no" = output[,c("batch_no")]) %>%
    filter(batch_no %in% c(grp-bxplt_window:1+1)) %>%
    group_by(batch_no) %>%
    summarise_all(mean) %>%
    gather(feature, value, -batch_no) %>%
    mutate(
      feature = factor(feature)
    ) %>%
    filter(feature != "day_mean_obpf") %>%
    mutate(x.label = paste("<span style = 'color: ",
                           ifelse(batch_no == grp, "red", "black"),
                           ";'>",
                           batch_no,
                           "</span>", sep = ""),
           x.label = fct_reorder(x.label, as.character(batch_no))) %>%
    ggplot(aes(x=x.label, y=feature, fill=value)) +
    geom_tile() +
    theme_classic()+
    theme(
      axis.title.x=element_blank(),
      axis.text.x= ggtext::element_markdown(angle = 90),
      axis.ticks.x=element_blank()
    ) +
    viridis::scale_fill_viridis(discrete=FALSE)
  
  trend_plot2 <- shap_values %>%
    cbind(output[,c("caseid", "batch_no")]) %>%
    filter(batch_no %in% c(grp-bxplt_window:1+1)) %>%
    # group_by(batch_no) %>%
    # summarise_all(mean) %>%
    gather(feature, value, -batch_no,-caseid) %>%
    mutate(
      feature = factor(feature)
    ) %>%
    filter(feature != "day_mean_obpf") %>%
    mutate(x.label = paste("<span style = 'color: ",
                           ifelse(caseid == case, "red", "black"),
                           ";'>",
                           caseid,
                           "</span>", sep = ""),
           x.label = factor(x.label)) %>%
    ggplot(aes(x=x.label, y=feature, fill=value)) +
    geom_tile() +
    theme_classic()+
    theme(
      axis.title.x=element_blank(),
      axis.text.x= ggtext::element_markdown(angle = 90),
      axis.ticks.x=element_blank()
    ) +
    viridis::scale_fill_viridis(discrete=FALSE)
  
  figure <- ggpubr::ggarrange(trend_plot, trend_plot2,
                              ncol = 1, nrow = 2)
  
  return(figure)
}
}

# Python code ----
if(F){
xgboost::xgb.save(xgb_base_up$finalModel, "../xgb_base_up.model")
xgboost::xgb.save(xgb_base_lw$finalModel, "../xgb_base_lw.model")

write.csv(model_data, "../test_SHAP_python2.csv", row.names = F)

shap_value_up <- predict(xgb_base_up$finalModel, as.matrix(input_x), predcontrib = TRUE, approxcontrib = FALSE) %>%
  as.data.frame()
shap_value_lw <- predict(xgb_base_lw$finalModel, as.matrix(input_x), predcontrib = TRUE, approxcontrib = FALSE) %>%
  as.data.frame()
write.csv(shap_value_up, "../shap_value_up.csv", row.names = F)
write.csv(shap_value_lw, "../shap_value_lw.csv", row.names = F)


library(reticulate)
source_python('../shap_plt_generate.py')
gen_shp_plt(input_x[7149,], shap_value_up[7149,1:13], shap_value_up[1,14], 7149)

}

 
