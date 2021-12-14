# Bibliotecas -----------------------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(ggplot2)
library(themis)


# Importando dados -----------------------------------------------------------------------------


dados <- read_csv("C:\\Users\\mcaio\\Desktop\\Random Forest\\income_evaluation.csv")

glimpse(dados)

dados %>% head(2)

any(is.na(dados$workclass))


dados %>% summary()


dados %>% count(income) %>% mutate(percentual = round((n/sum(n)*100),0))

# Transformação dos dados -----------------------------------------------------------------------------


dados <- dados %>% na_if("?") %>% mutate(income = forcats::as_factor(income),
                                         income = forcats::fct_rev(income))


str(dados$income)

dados %>% count(relationship)


# Divisao dos dados -----------------------------------------------------------------------------

set.seed(505)


divisao <- initial_split(dados, prop = 0.8, strata = income)

base_train <- training(divisao)
base_teste <- testing(divisao)


# Pré processamento -----------------------------------------------------------------------------


receita <- dados %>% recipe(income ~., base_train) %>% 
  step_rm(`native-country`, `marital-status`) %>% 
  step_impute_mode(workclass, occupation) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_smote(income)

teste <- juice(prep(receita))

glimpse(teste)



# Modelos ---------------------------------------------------------------------------------------



# 1 - Random Forest 


rforest <- rand_forest(
  mtry = tune(),
  trees = 80,
  min_n = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("ranger")



# 2 - Regressao logistica

rlog <- logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_mode("classification") %>% 
  set_engine("glmnet")


# 3 - Arvore de decisão

arvore <- decision_tree(cost_complexity = tune(),
                        tree_depth = tune(),
                        min_n = tune()) %>% 
          set_engine("rpart") %>% 
          set_mode("classification")





# Workflow------------------------------------------------------------------------------------


# - Random Forest


rf_wf <- workflow() %>% 
  add_recipe(receita) %>% 
  add_model(rforest)



# logistic regression


log_r <- workflow() %>% 
  add_recipe(receita) %>% 
  add_model(rlog)


# arvore de decisao

arvore_wf <- workflow() %>% 
  add_recipe(receita) %>% 
  add_model(arvore)



# VALIDACAO CRUZADA ---------------------------------------------------------------------------


dados_cv <- vfold_cv(base_train, v = 3)



# TUNAGEM  ---------------------------------------------------------------------------


# 1 - RANDOM FOREST


#  stackoverflow

rf_grid <- expand.grid(mtry = 10:13, min_n = c(80, 90, 100,120))


# mtry = 10:13, min_n = c(80,90,100,120) e o adasyn

# f_meas 68, precision 73 e recall 65

rf_res <- 
  rf_wf %>% 
  tune_grid(
    resamples = dados_cv,
    grid = rf_grid,
    metrics = metric_set(accuracy, precision, recall, f_meas, spec, sens)
  )

rf_res %>% unnest(.metrics)

rf_res %>% show_best("recall")
# recall 60 e precision 78
# smote recall 78 precision 64

rf_res %>% collect_metrics()

autoplot(rf_res)+theme_bw()



# 2 - LOGISTIC REGRESSION



lr_tune_grid <- tune_grid(
  log_r,
  resamples = dados_cv,
  metrics = metric_set(
    accuracy, 
    roc_auc,
    precision,
    recall
  )
)


lr_tune_grid %>% collect_metrics()

lr_tune_grid %>% show_best("precision")
# precision 77 e recall 59
# smote recall 85 e precision 57

# 3 - ARVORE DE DECISAO




arvore_grid <- grid_regular(cost_complexity(), 
                            tree_depth(), 
                            min_n(), 
                            levels = 3)


arvore_tune <- tune_grid(arvore_wf,
                         resamples = dados_cv, 
                         grid = arvore_grid,
                         metrics = metric_set(precision, recall, f_meas, accuracy, sens, spec, roc_auc))


arvore_tune %>% collect_metrics()

arvore_tune %>% show_best('recall')
# precision 95 e recall 61
#smote precision 56 e recall 93


# AVALIANDO DESEMPENHO DOS MODELOS---------------------------------------------------------


# 1 - RANDOM FOREST


# Selecionando os melhores hiperparametros

rf_res %>% show_best('recall')

new_rf <- rand_forest(mtry = 11, trees = 40, min_n = 120) %>% 
  set_mode('classification') %>% 
  set_engine('ranger')


new_wflow <- workflow() %>% 
  add_model(new_rf) %>% 
  add_recipe(receita)


fit_rf <- fit_resamples(new_wflow, 
                        dados_cv, 
                        metrics = metric_set(accuracy, recall, precision, f_meas, spec, sens),
                        control = control_resamples(save_pred = TRUE))


fit_rf %>% collect_metrics()



# LOGISTIC REGRESSION


# ARVORES DE DECISAO

arvore_tune %>% show_best('f_meas')

new_arvore <- decision_tree(cost_complexity = 0.0000000001,
                            tree_depth = 15,
                            min_n = 21) %>% 
  set_mode("classification") %>% 
  set_engine('rpart')


new_arv_wf <- workflow() %>% 
  add_recipe(receita) %>% 
  add_model(new_arvore)


fit_arvore_resample <- fit_resamples(new_arv_wf,
                                     dados_cv,
                                     metrics = metric_set(accuracy, recall, precision, f_meas, spec, sens),
                                     control = control_resamples(save_pred = TRUE))

fit_arvore_resample %>% collect_metrics()




# COMPARANDO MODELOS -----------


modelo <- bind_rows(
  fit_rf %>% collect_metrics(summarise = TRUE) %>% mutate(modelo = "Random Forest"),
  fit_arvore_resample %>% collect_metrics(summarise = TRUE) %>% mutate(modelo = "Arvore de decisao")
  ) %>% 
  select(modelo, .metric, mean, std_err)


modelo %>% 
  ggplot(aes(x = .metric, y = mean, fill = modelo))+
  geom_col(position = 'dodge')+
  coord_flip()+
  xlab("")+
  ylab("MÉDIA")+
  theme_minimal()+
  theme(axis.text.x = element_text(size =11, face = "bold"),
        axis.text.y = element_text(size = 13, face = "bold"))+
  ggtitle("Comparativo entre Modelos")



# Curva ROC
fit_rf %>% unnest(.predictions)

teste <- fit_rf %>% unnest(.predictions) %>% mutate(modelo = "Random Forest") %>% 
  bind_rows(fit_arvore_resample %>% unnest(.predictions) %>% mutate(modelo = 'Arvore de decisao')) 
  group_by(modelo) %>% 
  roc_curve(income, .pred_class)


roc_curv

arvore_tune

# SELECIONANDO O MODELO FINAL------------------------------------------------------------------------




best_parametro <- select_best(fit_rf, metric = 'f_meas')


rf_last_fit <- finalize_model(new_rf, best_parametro)


final_wf <- workflow() %>% 
  add_recipe(receita) %>% 
  add_model(rf_last_fit)


modelo_final <- final_wf %>% last_fit(divisao, metrics = metric_set(recall, precision, accuracy, f_meas))



# MÉTRICAS FINAIS ------------------------------------------------------------------------------------------


modelo_final %>% collect_metrics()

modelo_final %>% collect_predictions()


# MATRIZ DE CONFUSAO ------------------------------------------------------------

matriz_conf <- modelo_final %>% unnest(.predictions) %>% conf_mat(estimate = .pred_class, truth = income)


autoplot(matriz_conf, type = 'heatmap')

summary(matriz_conf)


