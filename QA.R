install.packages("proxy")
install.packages("dplyr")
library(NLP)
library(tm)
library(proxy)
library(dplyr)

iphone<-amazon[3833:3860,]

for(i in 1:27){
  x[i]<-c(as.String(iphone$Reviews[i]))
}

df <- x

N <- length(df)
names(df) <- paste0("doc", c(1:N))

query <- "Does the phone supports net10 and how is the screen"

#create a corpus

docs <- VectorSource(c(df, query))
docs$Names <- c(names(df), "query")

#preprocessing

corpus <- VCorpus(docs)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus,removeWords,stopwords("english"))

#create a tdm

tdm <- TermDocumentMatrix(my.corpus)
tdm
colnames(tdm) <- c(names(df), "query")
inspect(tdm[0:381, ])

#making matrix dense

mat <- as.matrix(tdm)
cat("Dense", object.size(mat), "Simple triplet matrix", object.size(tdm))

#tfidf

weights <- function(tf.vec) {
  # Computes tfidf weights from term frequency vector
  n.docs <- length(tf.vec)
  doc.frequency <- length(tf.vec[tf.vec > 0])
  weights <- rep(0, length(tf.vec))
  weights[tf.vec > 0] <- (1 + log2(tf.vec[tf.vec > 0])) * log2(n.docs/doc.frequency)
  return(weights)
}
# For a word appearing in 4 of 6 documents, occurring 1, 2, 3, and 6 times"
weights(c(1, 2, 3, 0, 0, 6))

#run the tfidf weighting function on every row of tdm

tfidf.matrix <- t(apply(mat, 1,FUN = function(row) {weights(row)}))
colnames(tfidf.matrix) <- colnames(mat)
tfidf.matrix[0:35, ]

#dot product geometry

angle <- seq(-pi, pi, by = pi/16)
plot(cos(angle) ~ angle, type = "b", xlab = "angle in radians",main = "Cosine similarity by angle")

tfidf.matrix <- scale(tfidf.matrix, center = FALSE,scale = sqrt(colSums(tfidf.matrix^2)))
tfidf.matrix[0:35, ]

#split

query.vector <- tfidf.matrix[, (N + 1)]
tfidf.matrix <- tfidf.matrix[, 1:N]

#cos theta

doc.scores <- t(query.vector) %*% tfidf.matrix

#rank by cosine similarities

results.df <- data.frame(doc = names(df), score = t(doc.scores),text = unlist(df))
results.df <- results.df[order(results.df$score, decreasing = TRUE), ]

#final

options(width = 2000)
print(results.df, row.names = FALSE, right = FALSE, digits = 2)

results.df$score
doc.scores
