# Librerias ----
library(TeachingDemos)
library(pracma)
library(ggplot2)
library(GGally)
library(corrplot)
library(fastDummies)
library(here)
library(dplyr)
library(ggthemes)
library(writexl)
library(FactoMineR)
library(cluster)
library(caret)
library(gsl)
library(factoextra)
library(energy)
library(MVN)
library(pROC)
library(psych)
ruta <- "C:\\Users\\mois_\\OneDrive\\Moi\\Decimo_semestre\\Multivariado\\Analisis-Multivariado\\R\\DescriptiveAnalysis\\DescriptiveAnalysis.R"
source(ruta)

# Datos ----
ruta = "C:\\Users\\mois_\\OneDrive\\Moi\\Decimo_semestre\\Multivariado\\Proyecto\\churn-bigml-80.csv"
data80 <- read.csv(ruta, header = TRUE, stringsAsFactors = TRUE)
data80$Area.code <- as.factor(data80$Area.code)
str(data80)
ruta = "C:\\Users\\mois_\\OneDrive\\Moi\\Decimo_semestre\\Multivariado\\Proyecto\\churn-bigml-20.csv"
data20 <- read.csv(ruta, header = TRUE, stringsAsFactors = TRUE)
data20$Area.code <- as.factor(data20$Area.code)
str(data20)

# Análisis descriptivo ----
summary(data80)
covdata <- cov(data80[,sapply(data80, is.numeric)])
covdata
cordata <- cor(data80[,sapply(data80,is.numeric)])
cordata

data80num <- data80[,sapply(data80,is.numeric)]

ggpairs(
  data80num,
  aes(color= data80$Churn)
)+
  theme_bw() +
  labs(title = "Matriz de Dispersión entre Variables Numéricas")

ggcorr(
  data80num,
  label = TRUE,
  label_size = 3,
  hjust = 0.75,
  size = 3,
  color = "black"
) +
  theme_bw() +
  labs(title = "Matriz de Correlación entre Variables Numéricas")

cancelacion <- data80num[data80$Churn == "True", ]
ggcorr(
  cancelacion,
  label = TRUE,
  label_size = 3,
  hjust = 0.75,
  size = 3,
  color = "black"
) +
  theme_bw() +
  labs(title = "Matriz de Correlación - Clientes que Cancelaron"
  )

no_cancelacion <- data80num[data80$Churn == "False", ]
ggcorr(
  no_cancelacion,
  label = TRUE,
  label_size = 3,
  hjust = 0.75,
  size = 3,
  color = "black"
) +
  theme_bw() +
  labs(title = "Matriz de Correlación - Clientes que No Cancelaron"
  )

andrewsCurves(
  as.matrix(data80num),
  data80$Churn,
)

# Pruebas normalidad ----
# Prueba de Shapiro-Wilk (para cada variable)
apply(data80num, 2, function(x) shapiro.test(x)$p.value)
# Test de Mardia
mardia <- mvn(data = data80num, mvn_test = "mardia")
mardia$multivariate_normality
# Test de Henze-Zirkler
hz <- mvn(data = data80num, mvn_test = "hz")
hz$multivariate_normality
# Histogramas
par(mfrow=c(2,2))
for(i in 1:4) {
  hist(data80num[,i], main=names(data80num)[i], col="skyblue")
}

# QQ-plots
for(i in 1:4) {
  qqnorm(data80num[,i], main=names(data80num)[i])
  qqline(data80num[,i], col="red")
}

# PCA ----
kappa(cordata, exact = TRUE)

num_scaled <- data80num %>%
  dplyr::select(c(-Total.intl.charge, -Total.night.charge, 
            -Total.day.charge, -Total.eve.charge)) %>%
  dplyr::mutate(across(everything(), as.numeric))

kappa(cor(num_scaled), exact = TRUE)

pca<- prcomp(num_scaled, center = TRUE, scale. = TRUE)
summary(pca)

var_exp <- pca$sdev^2 / sum(pca$sdev^2)

fviz_eig(pca, 
         ncp = 11,
         addlabels = TRUE,
         ylim = c(0, 10),
         barfill = "steelblue",
         barcolor = "steelblue",
         linecolor = "red",
         ggtheme = theme_minimal())

fviz_pca_ind(pca,
             geom.ind = "point",
             col.ind = data80$Churn, # colorear por cancelación
             palette = c("#00AFBB", "#FC4E07"),
             addEllipses = TRUE,
             legend.title = "Churn")

# Análisis de correspondencias ----
categoricas <- data80[, c("State", "Area.code", "International.plan", "Voice.mail.plan", "Churn")]

str(categoricas)

res.mca <- MCA(
  categoricas,
  quali.sup = c(1, 2, 5),
  graph = FALSE
)

fviz_mca_var(
  res.mca,
  choice = "var.cat",
  repel = TRUE,
  select.var = list(name = c("International.plan_Yes", "International.plan_No", 
                             "Voice.mail.plan_Yes", "Voice.mail.plan_No", 
                             "True", "False", "408", "415", "510")),
  col.var = "blue",
  col.sup = "magenta",
  labelsize = 4,
  ggtheme = theme_minimal()
) +
  labs(
    title = "Mapa de Categorías del MCA",
    subtitle = "Relación de Planes con Estados y Churn",
    x = "Dimensión 1",
    y = "Dimensión 2"
  )

# Regresión lineal ----
datamodelos <- data80%>%
  dplyr::select(-c(Total.day.charge, Total.eve.charge, Total.night.charge,
            Total.intl.charge))

modelo_logit <- glm(
  Churn ~ .-State,
  data   = datamodelos,
  family = binomial(link = "logit")
)

summary(modelo_logit)

or_tabla <- data.frame(
  Variable  = names(coef(modelo_logit)),
  OR        = exp(coef(modelo_logit)),
  IC_lower  = exp(confint(modelo_logit))[, 1],
  IC_upper  = exp(confint(modelo_logit))[, 2]
)
print(or_tabla)

## Predicción y evaluación ----
prob_logit <- predict(modelo_logit, newdata = data20, type = "response")
pred_logit <- ifelse(prob_logit >= mean(data80$Churn == "True"), "True", "False")
pred_logit <- factor(pred_logit, levels = c("False", "True"))

cm_logit <- confusionMatrix(pred_logit, data20$Churn, positive = "True")
cm_logit

roc_logit <- roc(data20$Churn, prob_logit, levels = c("False", "True"))
auc_logit  <- auc(roc_logit)
cat("AUC Regresión Logística:", round(auc_logit, 4), "\n")

ggroc(roc_logit, color = "steelblue", size = 1) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed", color = "gray50") +
  labs(
    title    = paste0("Curva ROC - Regresión Logística (AUC = ", round(auc_logit, 3), ")"),
    x        = "Especificidad",
    y        = "Sensibilidad"
  ) +
  theme_minimal()

coef_df <- data.frame(
  Variable = names(coef(modelo_logit))[-1],   # quitar intercepto
  OR       = exp(coef(modelo_logit))[-1],
  lower    = exp(confint(modelo_logit))[-1, 1],
  upper    = exp(confint(modelo_logit))[-1, 2]
)

ggplot(coef_df, aes(x = reorder(Variable, OR), y = OR)) +
  geom_point(color = "steelblue", size = 3) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.3, color = "steelblue") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  coord_flip() +
  labs(
    title = "Odds Ratios - Regresión Logística",
    x     = "Variable",
    y     = "Odds Ratio (IC 95%)"
  ) +
  theme_minimal()

# Clustering ----
## Preparar datos ----
cancelacion_cluster <- data80[data80$Churn == "True", ]
churn_cluster <- cancelacion_cluster %>%
  dplyr::select(-c(State, Churn)) %>%
  dummy_cols(
    select_columns = c("International.plan", "Voice.mail.plan", "Area.code"),
    remove_first_dummy  = TRUE,
    remove_selected_columns = TRUE
  ) %>%
  mutate(across(everything(), as.numeric))

churn_scaled <- scale(churn_cluster)

## Hallar k ----
dist_matrix <- dist(churn_scaled, method = "euclidean")
### AGNES ----
modelo_agnes <- agnes(dist_matrix, diss = TRUE, metric = "euclidean", 
                                   stand = FALSE,method = "ward")
dev.off()
plot(as.dendrogram(modelo_agnes), 
     lwd = 0.5, 
     cex = 0.5, 
     main = "Cluster aglomerativo")
### DIANA ----
diana_pca <- diana(dist_matrix, diss = TRUE, metric = "euclidean", 
                   stand = FALSE)
plot(as.dendrogram(diana_pca), 
     lwd = 0.5, 
     cex = 0.5, 
     main = "Cluster divisivo")

## k- means ----
k <- 4

set.seed(123)
km_model <- kmeans(churn_scaled, centers = k, nstart = 25, iter.max = 100)

churn_scaled$Cluster <- as.factor(km_model$cluster)

fviz_cluster(km_model, data = churn_scaled,
             geom = "point", ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal(),
             main = "Clusters – Clientes con Churn TRUE")

churn_true %>%
  group_by(Cluster) %>%
  summarise(across(where(is.numeric), mean, .names = "mean_{.col}")) %>%
  print(width = Inf)

head(churn_true)

## Predicción y evaluación ----
data80$Cluster <- 4  # valor por defecto para Churn == False

# Asignar los clusters de churn_true a las filas correspondientes
data80$Cluster[data80$Churn == "True"] <- as.numeric(churn_true$Cluster)

# Convertir a factor
data80$Cluster <- as.factor(data80$Cluster)

