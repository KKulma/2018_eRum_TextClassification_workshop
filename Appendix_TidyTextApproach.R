################################################################################################
# Building an Interpretable NLP model to classify tweets Workshop
# Appendix on tidytext example
# eRum 2018, Budapest 
################################################################################################


# load packages ####
library(readr)
library(dplyr)
library(stringr)
library(lubridate)
library(ggplot2)
library(tidytext)
library(tm)
library(text2vec)
library(wordcloud)
library(glmnet)
library(xgboost)
library(lime)
library(e1071)
library(SparseM)


## load data ####
tweet_csv <- read_csv("tweets.csv")

### data cleaning 
tweet_data <- tweet_csv %>% 
#  filter(is_retweet == "False") %>%
  select(author = handle, text, retweet_count, favorite_count, source_url, timestamp = time) %>% 
  mutate(date = as_date(str_sub(timestamp, 1, 10)),
         hour = hour(hms(str_sub(timestamp, 12, 19))),
         tweet_num = row_number()
  ) %>% select(-timestamp)


#### TIDYTEXT APPROACH ####

#show what tokenising is
example_text <- tweet_data %>%
  select(text) %>%
  dplyr::slice(1)

example_text %>%
  tidytext::unnest_tokens(sentence, text, token = "words")

example_text %>%
  tidytext::unnest_tokens(sentence, text, token = "sentences")

# data exploration ####
# data exploration that doesn't include tidytext has been removed

# who writes longer tweets?
sentence_data <- tweet_data %>% 
  select(tweet_num, text) %>% 
  tidytext::unnest_tokens(sentence, text, token = "sentences")
 
head(sentence_data)

word_data <- tweet_data %>% 
  select(tweet_num, text) %>% 
  tidytext::unnest_tokens(word, text, token = "words")

head(word_data)

sentences_count <- sentence_data %>% 
  group_by(tweet_num) %>% 
  summarise(n_sentences = n_distinct(sentence))

head(sentences_count)

word_count <- word_data %>% 
  group_by(tweet_num) %>% 
  summarise(n_words = n_distinct(word))

head(word_count)

## avg sentences per author  
tweet_data %>% 
  inner_join(sentences_count) %>% 
  group_by(author, date) %>% 
  summarise(avg_sentences  = mean(n_sentences)) %>% 
  ggplot(aes(date, avg_sentences, group = author, color = author)) +
    geom_line() +
    theme_minimal()
  

# avg words per author
tweet_data %>% 
  inner_join(word_count) %>% 
  group_by(author, date) %>% 
  summarise(avg_words = mean(n_words)) %>% 
  ggplot(aes(date, avg_words, group = author, color = author)) +
  geom_line() +
  theme_minimal()

### wordclouds
word_data %>%
  anti_join(get_stopwords()) %>% #?stopwords::stopwords
  count(word) %>%
  with(wordcloud(word, n, max.words = 100))

tweet_data %>% 
  inner_join(word_data) %>% 
  anti_join(get_stopwords()) %>%
  group_by(author) %>% 
  count(word) %>%
  mutate(colorSpecify = ifelse(author == "HillaryClinton", "blue", "red")) %>%
  with(wordcloud(word, n, max.words = 100,
          colors = colorSpecify, ordered.colors=TRUE))

#### most frequent words ####
word_data %>%
  anti_join(get_stopwords()) %>%
  filter(!(word %in% c("t.co")) %>%
  count(word, sort = TRUE)

#### format data for modelling ####

# create train and test data sets
indexes <- createDataPartition(tweet_data$author, times = 1,
                               p = 0.7, list = FALSE)

set.seed(32984)
indexes <- sample.int(n = nrow(tweet_data), size = floor(.8*nrow(tweet_data)), replace = F)

train_data <- tweet_data[indexes, ]
test_data <- tweet_data[-indexes, ]

#find the words in both train and test set
uniqueWords <- function(df) {
  allWords <- df %>% 
    select(tweet_num, author, text) %>% 
    tidytext::unnest_tokens(word, text, token = "tweets", strip_url = TRUE, strip_punct = TRUE)
  
  return(unique(allWords$word))
}

trainingWords_unique <- uniqueWords(train_data)
testWords_unique <- uniqueWords(test_data)

length(trainingWords_unique)
length(testWords_unique)
sum(trainingWords_unique %in% testWords_unique)

sameWords <- trainingWords_unique[trainingWords_unique %in% testWords_unique]

head(sameWords)


# word tokenization and sparce matrix creation
word_m <- function(df, sameWords){
  df %>% 
    select(tweet_num, author, text) %>% 
    tidytext::unnest_tokens(word, text, token = "tweets", strip_url = TRUE, strip_punct = TRUE) %>% 
    #tidytext::unnest_tokens(word, text) %>% 
    anti_join(get_stopwords()) %>%
    filter(word %in% sameWords) %>%
    count(tweet_num, word, sort = TRUE) %>%
    cast_sparse(tweet_num, word, n)
}

#set.seed(1)
train_m <- word_m(train_data, sameWords)
test_m <- word_m(test_data, sameWords)

str(train_m)
train_m[1:6, 1:6]
attributes(train_m)$Dimnames[[1]]

#some tweets were removed because they did not have any of the same words. 
# Must update train and test data.frames
dim(train_m)
dim(train_data)

head(train_m@Dimnames[[1]])
head(as.numeric(train_m@Dimnames[[1]]))
head(train_data)

removeUnwantedTweets <- function(df, sm){
  new_df <- df %>%
    filter(tweet_num %in% as.numeric(sm@Dimnames[[1]]))
}

train_data <- removeUnwantedTweets(train_data, train_m) 
test_data <- removeUnwantedTweets(test_data, test_m)


## glmnet model ####
set.seed(1234)
glm_model <- glmnet(train_m, train_data$author =="realDonaldTrump", family = "binomial")
glm_preds <- predict(glm_model, test_m) > 0.5


# Accuracy
mean(glm_preds == (test_data$author =="realDonaldTrump"))


## xgboost model ####

param <- list(max_depth = 7, 
              eta = 0.1, 
              objective = "binary:logistic", 
              eval_metric = "error", 
              nthread = 1)

set.seed(1234)
xgb_model <- xgb.train(
  param, 
  xgb.DMatrix(train_m, label = train_data$author == "realDonaldTrump"),
  nrounds = 50,
  verbose=0
)

dim(train_data)
dim(train_m)

# We use a (standard) threshold of 0.5
xgb_preds <- predict(xgb_model, test_m) > 0.5

# Accuracy
print(mean(xgb_preds == (test_data$author == "realDonaldTrump"))) 
# much lower accuracy than before
# potentially due to stop words not removed/noise


### SVM model ####
svm_model <- e1071::svm(train_m, as.numeric(train_data$author == "realDonaldTrump"), kernel='linear')
svm_preds <- predict(svm_model, test_m) > 0.5

# Accuracy
print(mean(svm_preds == (test_data$author == "realDonaldTrump")))



### LIME on glmnet model ####

# select only correct predictions
predictions_tbl <- glm_preds %>% 
  as_tibble() %>% 
  rename_(predict_label = names(.)[1]) %>%
  tibble::rownames_to_column()

correct_pred <- test_data %>%
  tibble::rownames_to_column() %>% 
  mutate(test_label = author == "realDonaldTrump") %>%
  left_join(predictions_tbl) %>%
  filter(test_label == predict_label) %>% 
  pull(text) %>% 
  head(4) 

str(correct_pred)

detach("package:dplyr", unload=TRUE)

library(lime)

explainer <- lime(correct_pred, 
                  model = xgb_model, 
                  preprocess = word_m(sameWords))

corr_explanation <- lime::explain(correct_pred, 
                                  explainer, 
                                  n_labels = 1, n_features = 6, cols = 2, verbose = 0)
plot_features(corr_explanation)


