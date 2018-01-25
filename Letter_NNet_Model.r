View(nnet)
N<- 200
D<- 2
K<- 4
X<- data.frame()
y<-data.frame()
set.seed(308)

library(ggplot2)
library(caret)

N<- 200
D<- 2
K<- 4
X<- data.frame()
y<-data.frame()
set.seed(308)

for (j in (1:K)){
r <- seq(0.05,1,length.out = N) # radius
t <- seq((j-1)*4.7,j*4.7, length.out = N) + rnorm(N, sd = 0.3) # theta
Xtemp <- data.frame(x =r*sin(t) , y = r*cos(t))
ytemp <- data.frame(matrix(j, N, 1))
X <- rbind(X, Xtemp)
y <- rbind(y, ytemp)
}

data<- cbind(X,y)
colnames(data) <- c(colnames(X), 'label')
x<- as.matrix(X)
X<-as.matrix(X)

View(X)
N <- 200 # number of points per class
D <- 2 # dimensionality
K <- 4 # number of classes
X <- data.frame() # data matrix (each row = single example)
y <- data.frame() # class labels

set.seed(308)

for (j in (1:K)){
r <- seq(0.05,1,length.out = N) # radius
t <- seq((j-1)*4.7,j*4.7, length.out = N) + rnorm(N, sd = 0.3) # theta
Xtemp <- data.frame(x =r*sin(t) , y = r*cos(t))
ytemp <- data.frame(matrix(j, N, 1))
X <- rbind(X, Xtemp)
y <- rbind(y, ytemp)
}

data <- cbind(X,y)
colnames(data) <- c(colnames(X), 'label')
X <- as.matrix(X)
Y <- matrix(0, N*K, K)
for (i in 1:(N*K)){
Y[i, y[i,]] <- 1
}

# %*% dot product, * element wise product
nnet <- function(X, Y, step_size = 0.5, reg = 0.001, h = 10, niteration){

# get dim of input
N <- nrow(X) # number of examples
K <- ncol(Y) # number of classes
D <- ncol(X) # dimensionality

# initialize parameters randomly
W <- 0.01 * matrix(rnorm(D*h), nrow = D)
b <- matrix(0, nrow = 1, ncol = h)
W2 <- 0.01 * matrix(rnorm(h*K), nrow = h)
b2 <- matrix(0, nrow = 1, ncol = K)

# gradient descent loop to update weight and bias
for (i in 0:niteration){

# hidden layer, ReLU activation
hidden_layer <- pmax(0, X%*% W + matrix(rep(b,N), nrow = N, byrow = T))
hidden_layer <- matrix(hidden_layer, nrow = N)

# class score
scores <- hidden_layer%*%W2 + matrix(rep(b2,N), nrow = N, byrow = T)

# compute and normalize class probabilities
exp_scores <- exp(scores)
probs <- exp_scores / rowSums(exp_scores)

# compute the loss: sofmax and regularization
corect_logprobs <- -log(probs)
data_loss <- sum(corect_logprobs*Y)/N
reg_loss <- 0.5*reg*sum(W*W) + 0.5*reg*sum(W2*W2)
loss <- data_loss + reg_loss

# check progress
if (i%%1000 == 0 | i == niteration){
print(paste("iteration", i,': loss', loss))}

# compute the gradient on scores
dscores <- probs-Y
dscores <- dscores/N

# backpropate the gradient to the parameters
dW2 <- t(hidden_layer)%*%dscores
db2 <- colSums(dscores)

# next backprop into hidden layer
dhidden <- dscores%*%t(W2)

# backprop the ReLU non-linearity
dhidden[hidden_layer <= 0] <- 0

# finally into W,b
dW <- t(X)%*%dhidden
db <- colSums(dhidden)

# add regularization gradient contribution
dW2 <- dW2 + reg *W2
dW <- dW + reg *W

# update parameter
W <- W-step_size*dW
b <- b-step_size*db
W2 <- W2-step_size*dW2
b2 <- b2-step_size*db2
}
return(list(W, b, W2, b2))
}
nnetPred <- function(X, para = list()){
W <- para[[1]]
b <- para[[2]]
W2 <- para[[3]]
b2 <- para[[4]]
N <- nrow(X)

hidden_layer <- pmax(0, X%*% W + matrix(rep(b,N), nrow = N, byrow = T))
hidden_layer <- matrix(hidden_layer, nrow = N)
scores <- hidden_layer%*%W2 + matrix(rep(b2,N), nrow = N, byrow = T)
predicted_class <- apply(scores, 1, which.max)
return(predicted_class)
}

nnet.model <- nnet(X, Y, step_size = 0.4,reg = 0.0002, h=50, niteration = 20000)
predicted_class <- nnetPred(X, nnet.model)

print(paste('training accuracy:',mean(predicted_class == (y))))
print("iteration")
load("~/.RData")

nnet.model <- nnet(X, Y, step_size = 0.4,reg = 0.0002, h=50, niteration = 20000)
predicted_class <- nnetPred(X, nnet.model)
print(paste('training accuracy:',mean(predicted_class == (y))))