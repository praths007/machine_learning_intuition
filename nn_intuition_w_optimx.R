rm(list = ls())
setwd("D:\\my_work\\open_source\\machine_learning_intuitions")
### caret only used for stratified random sampling
library(caret)

### understanding neural networks using gradient descent #############


### loading dataset
############################################
iris_data = iris[which(iris$Species %in% c("setosa", "versicolor")),]
## not used 1 and 0 because array indexes in R start from 1.
## this will ease the computation while creating seq...
## also this can be used to do multiclass classification
## where labels could be 1,2,3 etc...

iris_data$Species = ifelse(iris_data$Species == "setosa", 1, 2)


### adding intercept column to the data
iris_data$intercept = 1

iris_data = cbind(intercept = iris_data$intercept, 
                  iris_data[,c("Petal.Length", "Petal.Width", "Species")])

index = unlist(createDataPartition(iris_data$Species, p =  0.7))

train_x = data.matrix(iris_data[index,c("intercept", "Petal.Length", "Petal.Width")])

train_y = data.matrix(iris_data[index,c("Species")])


test_x = data.matrix(iris_data[-index, c("intercept", "Petal.Length", "Petal.Width")])

test_y = data.matrix(iris_data[-index, c("Species")])

nrow(train_x)
nrow(test_x)
############################################

### defining required functions
############################################

squared_error_cost = function(y, yhat){
  return((1 / nrow(y)) * sum((y - yhat)^2))}

## evaluation metric/ cost function logloss is used for classification
## https://datawookie.netlify.com/blog/2015/12/making-sense-of-logarithmic-loss/
logloss_cost = function(y, yhat){
  return(-(1 / nrow(y)) * sum((y*log(yhat) + (1 - y) * log(1 - yhat))))}

## activation function - used to limit values between 0 and 1
sigmoid = function(x){
  return(1 / (1 + 2.71^-x))}

## used for backpropogation
sigmoidGradient = function(x){
  return(sigmoid(x) * (1 - sigmoid(x)))}
############################################



#### neural networks intuition
################################################

num_labels = 2 # 1 and 2 in our case

m = dim(train_x)[1]
n = dim(train_x)[2]

## number of nodes in 1st layer/input layer must
## equal number of columns/ features of input data
layer1_nodes = ncol(train_x)

## this layer can have as many nodes as required
## rule of thumb is usually 1.5 times or equal number
## this is the hidden layer
layer2_nodes = trunc(ncol(train_x) * 1.5)

# final output layer nodes = number of labels
output_layer_nodes = num_labels

#################################################
# Theta1 = array(runif(n = gaussian(), min = 0, max = 1), c(layer2_nodes, layer1_nodes))
# 
# Theta2 = array(runif(n = gaussian(), min = 0, max = 1), c(output_layer_nodes, layer2_nodes+1))
# 
# ## input layer
# a1 = train_x
# 
# z2 = a1 %*% t(Theta1)
# 
# a2 = sigmoid(z2)
# 
# # second layer (hidden layer)
# a2 = cbind(1, a2)
# 
# z3 = a2 %*% t(Theta2)
# 
# # third layer (output layer)
# a3 = sigmoid(z3)

## neural network cost:
# it is a triple summation of logistic cost (because we are using sigmoid as activation),
# where the innermost summation is sum of cost for each node within a layer
# think of this as a collection of logistic regression models where each node represents
# one logit model (having sigmoid as its activation, similar to what was done in logit intuition)
################################################

cost = c()
cost_nn_optm = function(data, x){
  Theta1 = matrix(unlist(x[1:12]), ncol = 3, byrow = TRUE)
  Theta2 = matrix(unlist(x[13:length(x)]), ncol = 5, byrow = TRUE)
  
  
  
  train_x = data[[1]]
  train_y = data[[2]]
  output_layer_nodes = data[[3]]
  
  m = dim(train_x)[[1]]
  
  a1 = train_x
  
  z2 = a1 %*% t(Theta1)
  
  a2 = sigmoid(z2)
  
  a2 = cbind(1, a2)
  
  
  z3 = a2 %*% t(Theta2)
  
  a3 = sigmoid(z3)
  
  individual_train = array(0L, m)
  individual_theta_k = array(0L, output_layer_nodes)
  for(i in seq(1, m)){
    for(k in seq(1, num_labels)){
      individual_theta_k[k] = as.numeric(train_y[i,] == k)  * log(a3[i, k]) + 
        (1 - as.numeric(train_y[i,] == k)) * log(1 - a3[i, k])
    }
    individual_train[i] = sum(individual_theta_k)
  }
  -(1/m) * sum(individual_train)
}   





cost = c()
cost_nn_gradient_optm = function(data, x){
  Theta1 = matrix(unlist(x[1:12]), ncol = 3, byrow = TRUE)
  Theta2 = matrix(unlist(x[13:length(x)]), ncol = 5, byrow = TRUE)
  
  
  
  train_x = data[[1]]
  train_y = data[[2]]
  output_layer_nodes = data[[3]]
  
  m = dim(train_x)[[1]]
  
  a1 = train_x
  
  z2 = a1 %*% t(Theta1)
  
  a2 = sigmoid(z2)
  
  a2 = cbind(1, a2)
  
  
  z3 = a2 %*% t(Theta2)
  
  a3 = sigmoid(z3)
  
  individual_train = array(0L, m)
  individual_theta_k = array(0L, output_layer_nodes)
  for(i in seq(1, m)){
    for(k in seq(1, num_labels)){
      individual_theta_k[k] = as.numeric(train_y[i,] == k)  * log(a3[i, k]) + 
        (1 - as.numeric(train_y[i,] == k)) * log(1 - a3[i, k])
    }
    individual_train[i] = sum(individual_theta_k)
  }
  cost <<- c(cost, -(1/m) * sum(individual_train))
  
  
  ## backpropogation (adjusting weights of previous layers wrt error in output layer)
  ## delta3 = error in output layer
  ## again done for each node in the layer
  delta3 = array(0L, c(dim(a3)))
  for(i in seq(1, m)){
    for(k in seq(1, num_labels)){
      delta3[i, k] = a3[i, k] - as.numeric(train_y[i,] == k)
    }
  }
  
  # this is the error
  # delta3
  
  # we will be taking a negative gradient of this error/ cost to calculate the
  # direction of steepest ascent and then adjust weights for each node of each layer
  # accordingly
  
  delta2 = (delta3 %*% Theta2[,2:ncol(Theta2)]) * sigmoidGradient(z2)
  
  Theta1_grad = (1 / m) * t(delta2) %*% a1
  Theta2_grad = (1 / m) * t(delta3) %*% a2
  
  c(Theta1_grad, Theta2_grad)
}   



Theta1 = array(runif(n = gaussian(), min = 0, max = 1), c(layer2_nodes, layer1_nodes))
Theta2 = array(runif(n = gaussian(), min = 0, max = 1), c(output_layer_nodes, layer2_nodes+1))


# result = optim(c(Theta1, Theta2), cost_nn_optm, data = list(train_x, train_y, output_layer_nodes))


result = optim(c(Theta1, Theta2), cost_nn_optm, cost_nn_gradient_optm, 
               data = list(train_x, train_y, output_layer_nodes), method = "BFGS")



Theta1 = matrix(unlist(result$par[1:12]), ncol = 3, byrow = TRUE)
Theta2 = matrix(unlist(result$par[13:length(result$par)]), ncol = 5, byrow = TRUE)



h1 = sigmoid(test_x %*% t(Theta1))
h2 = sigmoid(cbind(1, h1) %*% t(Theta2))


preds = apply(h2, 1, function(x) which(x == max(x)))

confusionMatrix(as.factor(preds), as.factor(test_y))



plot(cost)



