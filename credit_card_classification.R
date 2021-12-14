
usethis::use_github()

# Dados utilizados foram retirados do Kaggle:


# https://www.kaggle.com/sakshigoyal7/credit-card-customers


# BIBLIOTECAS

library(tidyverse)
library(tidymodels)
library(corrplot)
library(themis)
library(vip)



# IMPORTAÇÃO DOS DADOS


dados <- read.csv("C:\\Users\\mcaio\\Desktop\\Cogumelos_classificação\\BankChurners.csv")


dados %>% str()


dados %>% summary()


glimpse(dados)


any(is.na(dados))



## ARRUMAR DADOS


## author dataset:

# PLEASE IGNORE THE LAST 2 COLUMNS (NAIVE BAYES CLAS…). 
#I SUGGEST TO RATHER DELETE IT BEFORE DOING ANYTHING**


dados <- dados %>% 
  select(-c(Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1,
            Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2)) %>% 
  mutate(Attrition_Flag = as.factor(Attrition_Flag))






## EDA----------------------------------------------------------------------------------------

dados %>% count(Marital_Status)
dados %>% count(Education_Level)
dados %>% count(Income_Category)
dados %>% count(Card_Category)


dados %>% ggplot(aes(x = Income_Category, fill = Attrition_Flag))+
  geom_bar()


dados %>% ggplot(aes(x = Marital_Status, fill = Attrition_Flag))+
  geom_bar()

dados %>% ggplot(aes(x = Education_Level, fill = Attrition_Flag))+
  geom_bar()

dados %>% group_by(Education_Level) %>% count(Attrition_Flag)

dados %>% ggplot(aes(x = Attrition_Flag, y = Credit_Limit))+geom_boxplot()


dados %>% ggplot(aes(x = Customer_Age))+geom_histogram()+theme_minimal()


dados %>% ggplot(aes(x = Gender))+geom_bar()


dados %>% select(where(is.numeric)) %>% cor() %>% corrplot()


## MODELAGEM DOS DADOS

# DIVISAO TREINO E TESTE ------------------------------------------------------------------


set.seed(42)


divisao <- initial_split(dados, prop = 0.8, strata = Attrition_Flag)


base_train <- training(divisao)

base_test <- testing(divisao)




# mesma proporção tanto na base de treino como na de teste

base_train %>% count(Attrition_Flag) %>% mutate(perc = n/sum(n)*100)

base_test %>% count(Attrition_Flag) %>% mutate(perc = n/sum(n)*100)



# RECIPES------------------------------------------------------------------



dados_recipe <- recipe(Attrition_Flag ~., base_train) %>% 
  step_dummy(all_nominal(), -all_outcomes())



juice(prep(dados_recipe))



# MODELO ------------------------------------------------------------------


# 1 - Arvore de decisao


arvore_decisao <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>% 
  set_mode("classification")


# 2 - Random Forest 


rforest <- rand_forest(
  mtry = tune(),
  trees = 30,
  min_n = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("ranger")




# Workflow------------------------------------------------------------------


# - Arvore de decisao

arvore_wf <- workflow() %>% 
  add_recipe(dados_recipe) %>% 
  add_model(arvore_decisao)


# - Random Forest

rf_wf <- workflow() %>% 
  add_recipe(dados_recipe) %>% 
  add_model(rforest)



# TUNAGEM -------------------------------------------------------------------------

arvore_grid <- grid_regular(cost_complexity(), 
                            tree_depth(), 
                            min_n(), 
                            levels = 3)


rf_grid <- grid_regular(
  mtry(range = c(10, 30)),
  min_n(range = c(2, 8)),
  levels = 3
)

# VALIDACAO CRUZADA ---------------------------------------------------------------------------

dados_cv <- vfold_cv(base_train, v = 5)


# FIT_RESAMPLES() -----------------------------------------------------------------------------


# 1 - Arvore de decisao



arvore_tune <- tune_grid(arvore_wf,
                         resamples = dados_cv,
                         grid = arvore_grid,
                         metrics = metric_set(recall, accuracy, spec, sens, roc_auc),
                         control = control_resamples(save_pred = TRUE))


collect_metrics(arvore_tune) %>% head(20)


autoplot(arvore_tune) + theme_bw() 


show_best(arvore_tune, metric = "recall")




# 2 - Random Forest

rf_tune <- tune_grid(rf_wf,
                     resamples = dados_cv,
                     grid = rf_grid,
                     metrics = metric_set(recall, accuracy, spec, sens, roc_auc))

rf_tune %>% collect_metrics()

rf_tune %>% show_best("recall")


autoplot(rf_tune)+theme_bw()


# COMPARANDO MODELOS------------------------------------------------------------------------------

modelos <- bind_rows(
  rf_tune %>% collect_metrics(summarise = TRUE) %>% mutate(model = 'random_forest'),
  arvore_tune %>% collect_metrics(summarise = TRUE) %>% mutate(model = 'decision_tree')
) %>% 
  select(model, .metric, mean, std_err)


modelos %>% group_by(model) %>% 
  filter(.metric == "recall") %>% 
  arrange(desc(mean)) %>% 
  filter(row_number()==1) %>%
  ggplot(aes(model, mean, fill = model))+
  geom_col()+
  coord_flip()+
  theme_minimal()+
  geom_text(aes(label = round(mean,3)))+
  ylab(NULL)+
  xlab(NULL)+
  ggtitle("Recall")

modelos %>% group_by(model) %>% 
  filter(.metric == "accuracy") %>% 
  arrange(desc(mean)) %>% 
  filter(row_number()==1) %>%
  ggplot(aes(model, mean, fill = model))+
  geom_col()+
  coord_flip()+
  theme_minimal()+
  geom_text(aes(label = round(mean,3)))+
  ylab(NULL)+
  xlab(NULL)+
  ggtitle("Accuracy")




# MODELO FINAL--------------------------------------------------------------------------------



# Vamos selecionar o Random forest devido ao melhor desempenho


best_param <- select_best(rf_tune, metric = 'recall')


rf_last_fit <- finalize_model(rforest, best_param)


final_wf <- workflow() %>% 
  add_recipe(dados_recipe) %>% 
  add_model(rf_last_fit)


final_res <- final_wf %>% last_fit(divisao, metrics = metric_set(recall, precision, roc_auc, sens, spec))



# Métricas Finais ---------------------------------------------------------------------------------

final_res %>% collect_metrics()

final_res %>% unnest(.predictions)



# MATRIZ CONFUSAO

matriz_confusao <- final_res %>% unnest(.predictions) %>% conf_mat(estimate = .pred_class, truth = Attrition_Flag)


autoplot(matriz_confusao, type = "heatmap")


summary(matriz_confusao)




