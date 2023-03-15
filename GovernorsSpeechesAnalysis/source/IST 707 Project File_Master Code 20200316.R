##################################################################################
#file: IST707 Course Project File - Final Version 20200316
#authors: John Christman, Christina DaSilva , Christopher Hart, and Jorge Martinez
##################################################################################
  
## Project Title: How Your Health May Be Related to What Your Governor Says 
## Exploring relationships between indicators and social determinants of health and the emotional sentiments of elected leaders
  
## This project examines the relationship between two occurrences - a political speech and the regional measure of health and 
#well-being.  The general inquiry is to consider if the emotions and sentiments selected and expressed by elected leaders 
#may persuade or hinder the general health and well-being of those they are elected to serve.  

#(1) Use a suite of data mining techniques to explore the relationship between state (U.S.) political speeches and the status of population health in the state.   
#(2) Identify methods for classifying sentiments and attitudes expressed by political leaders and finding patterns of relationships to political parties and select health and social determinant indicators.    
#(3) Examine predictive classification models to test hypothesis assumptions about the potential influence of linguistics and political party affiliation on indicated health measures.   
#(4) Gain a deeper understanding of the practices associated with the data science discipline.  

##################################################################################
#About the Data (Part 1) - Governor Speeches (corpus of text files)
##################################################################################
library(dplyr)
library(ggplot2)
library(tm) #Corpus

setwd("C:/Users/chris/OneDrive/Documents/R/IST 707") #SET TO PERSONAL COMPUTER

## Datasource github.com/fivethirtyeight-'State of the State': https://github.com/fivethirtyeight/data/tree/master/state-of-the-state)
## Citation: The State Of The State Of The States, What America's Governors Are Talking About: https://fivethirtyeight.com/features/what-americas-governors-are-talking-about/) - JUN. 13, 2019  

## The State Of The State Of The States, What America's Governors Are Talking About** provides text files of all 50 Governors' 
#2019 state of the state speeches to conduct sentiment analysis and look deeper into what issues were talked about the most 
#and whether there were differences between what Democratic and Republican governors were focusing on.  
           
## Corpus: statespeechesCorpus folder contains 50 .txt files containing the text of each of the speeches.

## File: state-speeches-index.csv** contains a listing of each of the 50 speeches, one for each state as well 
#as the name and party of the state's governor and a link to an official source for the speech. If an official government 
#source could not be found, we have linked to a news media source that had a transcript of the speech.

## File: state-speeches-words.csv** contains every one-word phrase that was mentioned in at least 10 speeches and every 
#two- or three-word phrase that was mentioned in at least five speeches after a list of stop-words was removed and the 
#word "healthcare" was replaced with "health care" so that they were not counted as distinct phrases. It also contains 
#the results of a chi^2 test that shows the statistical significance of and associated p-value of phrases.  

## Data transformations to be evaluated and applied for each analysis model and data mining technique: Discretization 
#(continuous to discrete; nominal or ordinal factor); Log Transformation (actual to log values to maintain relationships; 
#reduce impact of outliers); Normalization; or Standardization.  

###########################################
#Load TXT documents (corpus)
#Corpus folder contains 50 .txt files containing the text of each of the speeches.
#Adding additional 'removeWords' ==>  

#Load Corpus - NOTE: If not knitting, may need to copy/paste outside of Rmd notebok and execute in console
file1 = "statespeechesCorpus"
speechesCorpus <- Corpus(DirSource(file1))

#Review SimpleCorpus 
(summary(speechesCorpus))
(length(speechesCorpus))

#(1) Create Document Term Matrix from corpus

#See transformations available in DTM function
(getTransformations())
#Create variable for 'stopwords' to be removed in 'speechDTM'(="english" listing in tm package)
speechStops <- stopwords('english')
#Assign additional words to be removed that may influence training models (=authorship attribution)
stateStops <- c("alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new", "hampshire", "jersey", "mexico", "york", "north", "carolina", "dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode", "island", "south", "carolina", "tennessee", "texas", "utah", "vermont", "virginia", "washington", "west", "wisconsin", "wyoming")

speechDTM <- DocumentTermMatrix(speechesCorpus,
                                control = list(
                                  wordLengths=c(3, 10), 
                                  removePunctuation = TRUE,
                                  removeNumbers = TRUE,
                                  tolower=TRUE,
                                  remove_separators = TRUE,
                                  removeWords=speechStops,
                                  removeWords=stateStops
                                ))

#(2) Transform DTM into MATRIX format for cleaning and normalization
speechMAT <- as.matrix(speechDTM)
#Check Matrix output
(speechMAT[1:20,1:10])

#(3) Apply function to normalize numeric word occurrence values in Matrix

#Normalize to 'feature (dimension) value' = number of word occurence in row / total number of words in row
#Ref: https://rpubs.com/Mentors_Ubiqum/Normalize 
#For 'apply => function', 1 designates rows, 2 designates columns
#The function to normalize is (x/sum(x)) - round percent value to 3 decimal places
#PROJECT NOTATION: 
#"Normalization" typically means that the range of values are "normalized to be from 0.0 to 1.0". 
#"Standardization" typically means that the range of values are "standardized" to measure how many standard deviations the value is from its mean.

#Apply normalization function to MATRIX 'speechMAT' 
speechMAT_Norm <- apply(speechMAT, 1, function(x) round(x/sum(x),3))
#View output
(speechMAT_Norm[1:20,1:10])
#Transpose output'Terms' and 'Docs'
speechMAT_Norm <- t(speechMAT_Norm)
#Recheck results
(speechMAT_Norm[1:20,1:10])

#(4) Create DATAFRAME formats from normalized MATRIX 'speechMAT_Norm'
speechDF_Norm <- as.data.frame(as.matrix(speechMAT_Norm))
str(speechDF_Norm)
#Check if any blanks/NAs in dataframe
(sum(is.na(speechDF_Norm)))

#Create dataframe with labeled State Speech vector
speechDF_Norm_Label<- speechDF_Norm%>%add_rownames()
#Convert back into normalized DATA.FRAME from table output for Decision Tree application
speechDF_Norm <- as.data.frame(speechDF_Norm_Label)
str(speechDF_Norm)
#View listing of row names for label conversion (feature generation)
(View(speechDF_Norm$rowname))

#Rename first column
colnames(speechDF_Norm)[1] <- "Speech Filename"
#Create vector for State names, add to dataframe, and rename column
stateNames <- c("Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming")
speechDF_Norm <- data.frame(stateNames, speechDF_Norm)
colnames(speechDF_Norm)[1] <- "State"
(View(speechDF_Norm[, 1:2]))

#Read in additional reference CSV files 2 and 3

file2="state_speeches_index.csv"
indexDF<-read.csv(file2)
(View(indexDF))
#Remove columns and create updated DATAFRAME for 'speechDF_Norm' that includes State, Governor, Party, Filename
#Verify alignment and then cleanup columns
indexDF$X <- NULL
indexDF$url <- NULL
speechDF_Reporting <- data.frame(indexDF, speechDF_Norm)
(View(speechDF_Reporting[, 1:7]))
speechDF_Reporting$State <- NULL
speechDF_Reporting$Speech.Filename <- NULL
(View(speechDF_Reporting[, 1:5]))
#Change Party values to full names (as factors) 
#Ref: https://stackoverflow.com/questions/5824173/replace-a-value-in-a-data-frame-based-on-a-conditional-if-statement
speechDF_Reporting$party <- as.character(speechDF_Reporting$party)
speechDF_Reporting$party[speechDF_Reporting$party == "D"] <- "Democratic"
speechDF_Reporting$party[speechDF_Reporting$party == "R"] <- "Republican"
speechDF_Reporting$party <- as.factor(speechDF_Reporting$party)
str(speechDF_Reporting)

file3="state_speeches_words.csv"
wordsDF<-read.csv(file3)
(head(wordsDF))
(View(wordsDF$phrase))

#Create Data Dictionary for 'state-speeches-index.csv'
data_dictionary_col <- c("n-gram", "category", "d_speeches", "r_speeches", "total", "percent_of_d_speeches", "percent_of_r_speeches", "chi2", "pval")
data_dictionary_def <- c("one-, two- or three-word phrase", "thematic categories for n-grams hand-coded by FiveThirtyEight staff: economy/fiscal issues, education, health care, energy/environment, crime/justice, mental health/substance abuse", "number of Democratic speeches containing the n-gram", "number of Republican speeches containing the n-gram", "total number of speeches containing the n-gram", "percent of the 23 Democratic speeches containing the phrase", "percent of the 27 Republican speeches containing the phrase", "chi^2 statistic", "p-value for chi^2 test")
data_dictionary_index <- data.frame(data_dictionary_col, data_dictionary_def)
colnames(data_dictionary_index) <- c("Column", "Definition")

library(knitr)
library(kableExtra)
#Words report from FiveThirtyFive analysis
kable(wordsDF) %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", full_width = T))
#Data dictionary for 'wordsDF'
kable(data_dictionary_index) %>%
  kable_styling(bootstrap_options = "striped", full_width = F)

###########################################
## Initial Working Corpus Datasets for Models and Analysis:  

## speechDF_Reporting** - - Normalized DATAFRAME with all observations + Factors: state, governor, party, filename  
## speechMAT_Norm** - Normalized MATRIX with all txt speeches; % word occurrence freq.   
## speechDF_Norm** - Normalized DATAFRAME with all observations + State label (factor) & Speech txt title (chr)    
## indexDF** - Dataframe of all 50 speeches with attributes from Data Dictionary    
## wordsDF** - Dataframe of speech phrases with chi^2 test that shows the statistical significance of and associated 
#p-value of identified phrases (FiveThirtyFive analysis) 

###########################################
## Before looking at NRC emotion/sentiment analysis, plot the "thematic categories" by Political Party 
#(count % of related phrases) as identified by FiveThirtyEight research. 

#Create dataframe for stacked bar chart
wordsDF_Dem <- data.frame("Democratic", wordsDF$d_speeches)
wordsDF_Rep <- data.frame("Republican", wordsDF$r_speeches)
colnames(wordsDF_Dem) <- c("Party", "Count")
colnames(wordsDF_Rep) <- c("Party", "Count")
wordsDF_PartyCount <- rbind(wordsDF_Rep, wordsDF_Dem)
wordsDF_Bar <- data.frame(wordsDF$category, wordsDF_PartyCount)
colnames(wordsDF_Bar) <- c("Category", "Party", "Count")
wordsDF_Bar[1:20, ]
#Remove rows with blank 'Category' values
wordsDF_Bar <- wordsDF_Bar[!(is.na(wordsDF_Bar$Category) | wordsDF_Bar$Category==""), ]

f1 <-ggplot(wordsDF_Bar, aes(y=Count, x=Category, fill=Party)) + 
  geom_bar(position = "fill", stat = "identity") + 
  scale_fill_manual("Political Party", values = c("Republican" = "red", "Democratic" = "blue"))
f1

## Beyond the words, emotions, and sentiments is the reminder that our elected leaders belong to political parties. 
#And each party appears to place emphasis on different issues that can directly impact our health and well-being at 
#a community level. 

#################################################################################################
## Models for Analysis (Part 1): Sentiment (emotion) NRC analysis for speeches by state and party
## Data Question:** What emotions and sentiments are conveyed by our elected leaders when 
#addressing those living in their state?
#################################################################################################
library(dplyr)
library(tidyr)

## TESTING processing code for SENTIMENT-EMOTION ANALYSIS to create functions (apply to state of Alabama )

#Create subset dataframe from 'speechDF_Norm' for the state - ALABAMA
#alabama <- subset(speechDF_Norm, speechDF_Norm$State == "Alabama")
#Remove Speech column
#alabama$State <- NULL
#alabama$Speech <- NULL
#Convert num value 0 to 'NA' (Ref:https://stackoverflow.com/questions/9977686/how-to-remove-rows-with-any-zero-value)
#alabama[alabama==0] <- NA
#Remove 'NA' columns
#alabama <- alabama[, colSums(is.na(alabama)) != nrow(alabama)]
#Check str() for column length - structure gather 2:#variables-1
#str(alabama)
#Convert to long format vector with Words using 'gather' (Ref: https://uc-r.github.io/tidyr)
#alabama <- alabama %>% gather(Word, Occurence, 2:319)
#ALwords <- alabama$Word

#Report description and sentiment analayis ref: https://www.tidytextmining.com/sentiment.html
#Lexicon ref: http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
#The NRC Emotion Lexicon is a list of English words and their associations with eight basic emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and two sentiments (negative and positive).
#The nrc lexicon identifies words in a binary fashion (0 (not associated) or 1 (associated)) into categories of positive, negative, anger, anticipation, disgust, fear, joy, sadness, surprise, and trust. 

## Sentiment-emotion analysis for state. Ref: http://rstudio-pubs-static.s3.amazonaws.com/283881_efbb666d653a4eb3b0c5e5672e3446c6.html

library(syuzhet) #NRC lexicon
library(plotly)

## Testing sentiment and emotion analysis ('nrc')
#emotionsAL <- get_nrc_sentiment(ALwords)
#emo_barAL = colSums(emotionsAL)
#emo_sumAL = data.frame(count=emo_barAL, emotion=names(emo_barAL))
#emo_sumAL$emotion = factor(emo_sumAL$emotion, levels=emo_sumAL$emotion[order(emo_sumAL$count, decreasing = TRUE)])

## Visualization testing for emotion analysis
#AL1 <- plot_ly(emo_sumAL, x=~emotion, y=~count, type="bar", color=~emotion) %>%
#layout(xaxis=list(title=""), showlegend=FALSE,
#title="Distribution of emotions and sentiment for Alabama Speech")

###########################################
## BASED ON TESTING, create FUNCTIONS for NRC sentiment analysis

#Create a FUNCTION to create a long format vector of words for each state for sentiment analysis 
stateWords <- function(stateName) {
  statewords <- subset(speechDF_Norm, speechDF_Norm$State == stateName)
  statewords$State <- NULL
  statewords$Speech <- NULL
  statewords[statewords==0] <- NA
  statewords <- statewords[, colSums(is.na(statewords)) != nrow(statewords)]
  indexState=length(statewords)-1
  statewords <- statewords %>% gather(Word, Occurence, 2:indexState)
  statewords <- statewords$Word
  return(statewords)
}

#Run function to create vector of words for each state ==> apply variables to next function
ALwords <- stateWords("Alabama")
AKwords <- stateWords("Alaska")
AZwords <- stateWords("Arizona")
ARwords <- stateWords("Arkansas")
CAwords <- stateWords("California")
COwords <- stateWords("Colorado")
CTwords <- stateWords("Connecticut")
DEwords <- stateWords("Delaware")
FLwords <- stateWords("Florida")
GAwords <- stateWords("Georgia")
HIwords <- stateWords("Hawaii")
IDwords <- stateWords("Idaho")
ILwords <- stateWords("Illinois")
INwords <- stateWords("Indiana")
IAwords <- stateWords("Iowa")
KSwords <- stateWords("Kansas")
KTwords <- stateWords("Kentucky")
LAwords <- stateWords("Louisiana")
MEwords <- stateWords("Maine")
MDwords <- stateWords("Maryland")
MAwords <- stateWords("Massachusetts")
MIwords <- stateWords("Michigan")
MNwords <- stateWords("Minnesota")
MSwords <- stateWords("Mississippi")
MOwords <- stateWords("Missouri")
MTwords <- stateWords("Montana")
NEwords <- stateWords("Nebraska")
NVwords <- stateWords("Nevada")
NHwords <- stateWords("New Hampshire")
NJwords <- stateWords("New Jersey")
NMwords <- stateWords("New Mexico")
NYwords <- stateWords("New York")
NCwords <- stateWords("North Carolina")
NDwords <- stateWords("North Dakota")
OHwords <- stateWords("Ohio")
OKwords <- stateWords("Oklahoma")
ORwords <- stateWords("Oregon")
PAwords <- stateWords("Pennsylvania")
RIwords <- stateWords("Rhode Island")
SCwords <- stateWords("South Carolina")
SDwords <- stateWords("South Dakota")
TNwords <- stateWords("Tennessee")
TXwords <- stateWords("Texas")
UTwords <- stateWords("Utah")
VTwords <- stateWords("Vermont")
VAwords <- stateWords("Virginia")
WAwords <- stateWords("Washington")
WVwords <- stateWords("West Virginia")
WIwords <- stateWords("Wisconsin")
WYwords <- stateWords("Wyoming")

#Create a FUNCTION to count associations and report emotional sentiments ('nrc' lexicon) 
stateEmotions <- function(STwords) {
  emotions <- get_nrc_sentiment(STwords)
  emo_bar = colSums(emotions)
  emo_sum = data.frame(count=emo_bar, emotion=names(emo_bar))
  emo_sum$emotion = factor(emo_sum$emotion, levels=emo_sum$emotion[order(emo_sum$count, decreasing = TRUE)])
  return(emo_sum)
}

#Run function for each STATEwords ==> apply variables to next function
#Rbind reporting dataframe with each report
emotionsCols <- c("state", "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "negative", "positive")

ALemotions <- stateEmotions(ALwords)
ALreport <- c("alabama", ALemotions$count)

AKemotions <- stateEmotions(AKwords)
AKreport <- c("alaska", AKemotions$count)
emotionsStates <- rbind(ALreport, AKreport)
colnames(emotionsStates) <- c("state", "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "negative", "positive")

AZemotions <- stateEmotions(AZwords)
AZreport <- c("arizona", AZemotions$count)
emotionsStates <- rbind(emotionsStates, AZreport)

ARemotions <- stateEmotions(ARwords)
ARreport <- c("arkansas", ARemotions$count)
emotionsStates <- rbind(emotionsStates, ARreport)

CAemotions <- stateEmotions(CAwords)
CAreport <- c("california", CAemotions$count)
emotionsStates <- rbind(emotionsStates, CAreport)

COemotions <- stateEmotions(COwords)
COreport <- c("colorado", COemotions$count)
emotionsStates <- rbind(emotionsStates, COreport)

CTemotions <- stateEmotions(CTwords)
CTreport <- c("connecticut", CTemotions$count)
emotionsStates <- rbind(emotionsStates, CTreport)

DEemotions <- stateEmotions(DEwords)
DEreport <- c("delaware", DEemotions$count)
emotionsStates <- rbind(emotionsStates, DEreport)

FLemotions <- stateEmotions(FLwords)
FLreport <- c("florida", FLemotions$count)
emotionsStates <- rbind(emotionsStates, FLreport)

GAemotions <- stateEmotions(GAwords)
GAreport <- c("georgia", GAemotions$count)
emotionsStates <- rbind(emotionsStates, GAreport)

HIemotions <- stateEmotions(HIwords)
HIreport <- c("hawaii", HIemotions$count)
emotionsStates <- rbind(emotionsStates, HIreport)

IDemotions <- stateEmotions(IDwords)
IDreport <- c("idaho", IDemotions$count)
emotionsStates <- rbind(emotionsStates, IDreport)

ILemotions <- stateEmotions(ILwords)
ILreport <- c("illinois", ILemotions$count)
emotionsStates <- rbind(emotionsStates, ILreport)

INemotions <- stateEmotions(INwords)
INreport <- c("indiana", INemotions$count)
emotionsStates <- rbind(emotionsStates, INreport)

IAemotions <- stateEmotions(IAwords)
IAreport <- c("iowa", IAemotions$count)
emotionsStates <- rbind(emotionsStates, IAreport)

KSemotions <- stateEmotions(KSwords)
KSreport <- c("kansas", KSemotions$count)
emotionsStates <- rbind(emotionsStates, KSreport)

KTemotions <- stateEmotions(KTwords)
KTreport <- c("kentucky", KTemotions$count)
emotionsStates <- rbind(emotionsStates, KTreport)

LAemotions <- stateEmotions(LAwords)
LAreport <- c("louisiana", LAemotions$count)
emotionsStates <- rbind(emotionsStates, LAreport)

MEemotions <- stateEmotions(MEwords)
MEreport <- c("maine", MEemotions$count)
emotionsStates <- rbind(emotionsStates, MEreport)

MDemotions <- stateEmotions(MDwords)
MDreport <- c("maryland", MDemotions$count)
emotionsStates <- rbind(emotionsStates, MDreport)

MAemotions <- stateEmotions(MAwords)
MAreport <- c("massachusetts", MAemotions$count)
emotionsStates <- rbind(emotionsStates, MAreport)

MIemotions <- stateEmotions(MIwords)
MIreport <- c("michigan", MIemotions$count)
emotionsStates <- rbind(emotionsStates, MIreport)

MNemotions <- stateEmotions(MNwords)
MNreport <- c("minnesota", MNemotions$count)
emotionsStates <- rbind(emotionsStates, MNreport)

MSemotions <- stateEmotions(MSwords)
MSreport <- c("mississippi", MSemotions$count)
emotionsStates <- rbind(emotionsStates, MSreport)

MOemotions <- stateEmotions(MOwords)
MOreport <- c("missouri", MOemotions$count)
emotionsStates <- rbind(emotionsStates, MOreport)

MTemotions <- stateEmotions(MTwords)
MTreport <- c("montana", MTemotions$count)
emotionsStates <- rbind(emotionsStates, MTreport)

NEemotions <- stateEmotions(NEwords)
NEreport <- c("nebraska", NEemotions$count)
emotionsStates <- rbind(emotionsStates, NEreport)

NVemotions <- stateEmotions(NVwords)
NVreport <- c("nevada", NVemotions$count)
emotionsStates <- rbind(emotionsStates, NVreport)

NHemotions <- stateEmotions(NHwords)
NHreport <- c("new hampshire", NHemotions$count)
emotionsStates <- rbind(emotionsStates, NHreport)

NJemotions <- stateEmotions(NJwords)
NJreport <- c("new jersey", NJemotions$count)
emotionsStates <- rbind(emotionsStates, NJreport)

NMemotions <- stateEmotions(NMwords)
NMreport <- c("new mexico", NMemotions$count)
emotionsStates <- rbind(emotionsStates, NMreport)

NYemotions <- stateEmotions(NYwords)
NYreport <- c("new york", NYemotions$count)
emotionsStates <- rbind(emotionsStates, NYreport)

NCemotions <- stateEmotions(NCwords)
NCreport <- c("north carolina", NCemotions$count)
emotionsStates <- rbind(emotionsStates, NCreport)

NDemotions <- stateEmotions(NDwords)
NDreport <- c("north dakota", NDemotions$count)
emotionsStates <- rbind(emotionsStates, NDreport)

OHemotions <- stateEmotions(OHwords)
OHreport <- c("ohio", OHemotions$count)
emotionsStates <- rbind(emotionsStates, OHreport)

OKemotions <- stateEmotions(OKwords)
OKreport <- c("oklahoma", OKemotions$count)
emotionsStates <- rbind(emotionsStates, OKreport)

ORemotions <- stateEmotions(ORwords)
ORreport <- c("oregon", ORemotions$count)
emotionsStates <- rbind(emotionsStates, ORreport)

PAemotions <- stateEmotions(PAwords)
PAreport <- c("pennsylvania", PAemotions$count)
emotionsStates <- rbind(emotionsStates, PAreport)

RIemotions <- stateEmotions(RIwords)
RIreport <- c("rhode island", RIemotions$count)
emotionsStates <- rbind(emotionsStates, RIreport)

SCemotions <- stateEmotions(SCwords)
SCreport <- c("south carolina", SCemotions$count)
emotionsStates <- rbind(emotionsStates, SCreport)

SDemotions <- stateEmotions(SDwords)
SDreport <- c("south dakota", SDemotions$count)
emotionsStates <- rbind(emotionsStates, SDreport)

TNemotions <- stateEmotions(TNwords)
TNreport <- c("tennessee", TNemotions$count)
emotionsStates <- rbind(emotionsStates, TNreport)

TXemotions <- stateEmotions(TXwords)
TXreport <- c("texas", TXemotions$count)
emotionsStates <- rbind(emotionsStates, TXreport)

UTemotions <- stateEmotions(UTwords)
UTreport <- c("utah", UTemotions$count)
emotionsStates <- rbind(emotionsStates, UTreport)

VTemotions <- stateEmotions(VTwords)
VTreport <- c("vermont", VTemotions$count)
emotionsStates <- rbind(emotionsStates, VTreport)

VAemotions <- stateEmotions(VAwords)
VAreport <- c("virginia", VAemotions$count)
emotionsStates <- rbind(emotionsStates, VAreport)

WAemotions <- stateEmotions(WAwords)
WAreport <- c("washington", WAemotions$count)
emotionsStates <- rbind(emotionsStates, WAreport)

WVemotions <- stateEmotions(WVwords)
WVreport <- c("west virginia", WVemotions$count)
emotionsStates <- rbind(emotionsStates, WVreport)

WIemotions <- stateEmotions(WIwords)
WIreport <- c("wisconsin", WIemotions$count)
emotionsStates <- rbind(emotionsStates, WIreport)

WYemotions <- stateEmotions(WYwords)
WYreport <- c("wyoming", WYemotions$count)
emotionsStates <- rbind(emotionsStates, WYreport)

library(knitr)
library(kableExtra)
#TABLE: Count of emotions and sentiments by state speech
kable(emotionsStates) %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", full_width = T))

#Convert MATRIX 'emotionsStates' to DATAFRAME 
emotionsStates_Norm <- as.data.frame(emotionsStates[1:50,2:11])
#Convert emotion factors to numeric variables - need to use 'as.character' to retain numeric value versus level
emotionsStates_Norm$anger <- as.numeric(as.character(emotionsStates_Norm$anger))
emotionsStates_Norm$anticipation <- as.numeric(as.character(emotionsStates_Norm$anticipation))
emotionsStates_Norm$disgust <- as.numeric(as.character(emotionsStates_Norm$disgust))
emotionsStates_Norm$fear <- as.numeric(as.character(emotionsStates_Norm$fear))
emotionsStates_Norm$joy <- as.numeric(as.character(emotionsStates_Norm$joy))
emotionsStates_Norm$sadness <- as.numeric(as.character(emotionsStates_Norm$sadness))
emotionsStates_Norm$surprise <- as.numeric(as.character(emotionsStates_Norm$surprise))
emotionsStates_Norm$trust <- as.numeric(as.character(emotionsStates_Norm$trust))
emotionsStates_Norm$negative <- as.numeric(as.character(emotionsStates_Norm$negative))
emotionsStates_Norm$positive <- as.numeric(as.character(emotionsStates_Norm$positive))
(head(emotionsStates_Norm))

#Add Political Party and 'stateNames' column references back in as factors in DATAFRAME 
politicalParty <- speechDF_Reporting$party
stateAbbrev <- c("AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KT", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY")
emotionsStates_DF <- cbind(politicalParty, stateAbbrev, emotionsStates_Norm)
(head(emotionsStates_DF))

library(ggplot2)
library(RColorBrewer)

#PLOTS: Emotions and sentiments - states and parties

#Figure 1: Anger in Governor's Speeches (scatterplot)
e1 <-ggplot(emotionsStates_DF, aes(x =stateAbbrev, y =anger, fill =politicalParty, size =anger)) +
  geom_point(shape =21) +
  scale_fill_manual(values = c("Blue", "Red")) +
  ggtitle("Anger in Governor's Speeches") +
  labs(x ="By State", y ="Number of Sentiments in Speeches") +
  theme(legend.position ="bottom", legend.direction ="horizontal",plot.title =element_text(hjust =0.5),axis.title =element_text(size =12),legend.text =element_text(size =9),legend.title=element_text(size=9))
e1

#Figure 2: Anticipation in Governor's Speeches (scatterplot)
e2 <-ggplot(emotionsStates_DF, aes(x =stateAbbrev, y =anticipation, fill =politicalParty, size =anticipation)) +
  geom_point(shape =21) +
  scale_fill_manual(values = c("Blue", "Red")) +
  ggtitle("Anticipation in Governor's Speeches") +
  labs(x ="By State", y ="Number of Sentiments in Speeches") +
  theme(legend.position ="bottom", legend.direction ="horizontal",plot.title =element_text(hjust =0.5),axis.title =element_text(size =12),legend.text =element_text(size =9),legend.title=element_text(size=9))
e2

#Figure 3: Disgust in Governor's Speeches (scatterplot)
e3 <-ggplot(emotionsStates_DF, aes(x =stateAbbrev, y =disgust, fill =politicalParty, size =disgust)) +
  geom_point(shape =21) +
  scale_fill_manual(values = c("Blue", "Red")) +
  ggtitle("Disgust in Governor's Speeches") +
  labs(x ="By State", y ="Number of Sentiments in Speeches") +
  theme(legend.position ="bottom", legend.direction ="horizontal",plot.title =element_text(hjust =0.5),axis.title =element_text(size =12),legend.text =element_text(size =9),legend.title=element_text(size=9))
e3

#Figure 4: Fear in Governor's Speeches (scatterplot)
e4 <-ggplot(emotionsStates_DF, aes(x =stateAbbrev, y =fear, fill =politicalParty, size =fear)) +
  geom_point(shape =21) +
  scale_fill_manual(values = c("Blue", "Red")) +
  ggtitle("Fear in Governor's Speeches") +
  labs(x ="By State", y ="Number of Sentiments in Speeches") +
  theme(legend.position ="bottom", legend.direction ="horizontal",plot.title =element_text(hjust =0.5),axis.title =element_text(size =12),legend.text =element_text(size =9),legend.title=element_text(size=9))
e4

#Figure 5: Joy in Governor's Speeches (scatterplot)
e5 <-ggplot(emotionsStates_DF, aes(x =stateAbbrev, y =joy, fill =politicalParty, size =joy)) +
  geom_point(shape =21) +
  scale_fill_manual(values = c("Blue", "Red")) +
  ggtitle("Joy in Governor's Speeches") +
  labs(x ="By State", y ="Number of Sentiments in Speeches") +
  theme(legend.position ="bottom", legend.direction ="horizontal",plot.title =element_text(hjust =0.5),axis.title =element_text(size =12),legend.text =element_text(size =9),legend.title=element_text(size=9))
e5

#Figure 6: Sadness in Governor's Speeches (scatterplot)
e6 <-ggplot(emotionsStates_DF, aes(x =stateAbbrev, y =sadness, fill =politicalParty, size =sadness)) +
  geom_point(shape =21) +
  scale_fill_manual(values = c("Blue", "Red")) +
  ggtitle("Sadness in Governor's Speeches") +
  labs(x ="By State", y ="Number of Sentiments in Speeches") +
  theme(legend.position ="bottom", legend.direction ="horizontal",plot.title =element_text(hjust =0.5),axis.title =element_text(size =12),legend.text =element_text(size =9),legend.title=element_text(size=9))
e6

#Figure 7: Surprise in Governor's Speeches (scatterplot)
e7 <-ggplot(emotionsStates_DF, aes(x =stateAbbrev, y =surprise, fill =politicalParty, size =surprise)) +
  geom_point(shape =21) +
  scale_fill_manual(values = c("Blue", "Red")) +
  ggtitle("Surprise in Governor's Speeches") +
  labs(x ="By State", y ="Number of Sentiments in Speeches") +
  theme(legend.position ="bottom", legend.direction ="horizontal",plot.title =element_text(hjust =0.5),axis.title =element_text(size =12),legend.text =element_text(size =9),legend.title=element_text(size=9))
e7

#Figure 8: Trust in Governor's Speeches (scatterplot)
e8 <-ggplot(emotionsStates_DF, aes(x =stateAbbrev, y =trust, fill =politicalParty, size =trust)) +
  geom_point(shape =21) +
  scale_fill_manual(values = c("Blue", "Red")) +
  ggtitle("Trust in Governor's Speeches") +
  labs(x ="By State", y ="Number of Sentiments in Speeches") +
  theme(legend.position ="bottom", legend.direction ="horizontal",plot.title =element_text(hjust =0.5),axis.title =element_text(size =12),legend.text =element_text(size =9),legend.title=element_text(size=9))
e8

#Figure 9: Negative Sentiments in Governor's Speeches (scatterplot)
e9 <-ggplot(emotionsStates_DF, aes(x =stateAbbrev, y =negative, fill =politicalParty, size =negative)) +
  geom_point(shape =21) +
  scale_fill_manual(values = c("Blue", "Red")) +
  ggtitle("Negative Sentiments in Governor's Speeches") +
  labs(x ="By State", y ="Number in Speeches") +
  theme(legend.position ="bottom", legend.direction ="horizontal",plot.title =element_text(hjust =0.5),axis.title =element_text(size =12),legend.text =element_text(size =9),legend.title=element_text(size=9))
e9

#Figure 10: Positive Sentiments Governor's Speeches (scatterplot)
e10 <-ggplot(emotionsStates_DF, aes(x =stateAbbrev, y =positive, fill =politicalParty, size =positive)) +
  geom_point(shape =21) +
  scale_fill_manual(values = c("Blue", "Red")) +
  ggtitle("Positive Sentiments in Governor's Speeches") +
  labs(x ="By State", y ="Number in Speeches") +
  theme(legend.position ="bottom", legend.direction ="horizontal",plot.title =element_text(hjust =0.5),axis.title =element_text(size =12),legend.text =element_text(size =9),legend.title=element_text(size=9))
e10

#Add column to reporting DATAFRAME for % positive sentiment
#Test before appying sequence => pos/sum(neg + pos)
#testDF <- emotionsStates_DF
#str(testDF)
#testAdd <- rowSums(testDF[,c("negative", "positive")])
#testDF <- cbind(testDF,testAdd)
#testDF$testAdd <- testDF$positive/testDF$testAdd

#Apply to reporting DATAFRAME
percentPositive <- rowSums(emotionsStates_DF[,c("negative", "positive")])
emotionsStates_DF <- cbind(emotionsStates_DF,percentPositive)
emotionsStates_DF$percentPositive <- emotionsStates_DF$positive/emotionsStates_DF$percentPositive
#Check results
(head(emotionsStates_DF))
positivePercentOrder <- emotionsStates_DF[order(emotionsStates_DF$percentPositive), ]
#Check ordering
(View(positivePercentOrder$percentPositive))

#Identify high/low positive percentages for plots
(head(positivePercentOrder))
(tail(positivePercentOrder))

#Figures: Plots of high-low positive sentiments (select states)
#Need to expand color palette size
nb.cols <- 10
chartColors <- colorRampPalette(brewer.pal(8, "Set2"))(nb.cols)

#Highest % positive setiment: Mississippi emotions ('MSemotions')
MS1 <- plot_ly(MSemotions, x=~emotion, y=~count, type="bar", color=chartColors) %>%
  layout(xaxis=list(title=""), showlegend=FALSE,
         title="Distribution of emotions and sentiments for Mississippi speech")
MS1

#Lowest % positive setiment: Maryland emotions ('MDemotions')
MD1 <- plot_ly(MDemotions, x=~emotion, y=~count, type="bar", color=chartColors) %>%
  layout(xaxis=list(title=""), showlegend=FALSE,
         title="Distribution of emotions and sentiments for Maryland speech")
MD1

#Figures: Plots for health reporting section - Physical and Mental Health Days areas of focus (Oklahoma and West Virginia)

OK1 <- plot_ly(OKemotions, x=~emotion, y=~count, type="bar", color=chartColors) %>%
  layout(xaxis=list(title=""), showlegend=FALSE,
         title="Distribution of emotions and sentiments for Oklahoma speech")
OK1

WV1 <- plot_ly(WVemotions, x=~emotion, y=~count, type="bar", color=chartColors) %>%
  layout(xaxis=list(title=""), showlegend=FALSE,
         title="Distribution of emotions and sentiments for West Virginia speech")
WV1

## As we might expect given that Governors are politicians, the number of positive sentiments in any given speech 
#is significantly higher than its negative sentiments.  

## What is interesting about sentiments is that underneath them are actual words and related emotions that create 
#sentimental attitudes as illustrated in the differences between Maryland and Mississippi speeches which reflected 
#the low and high end of positivity (sentimental differences driven by emotions of anger and fear).  

###########################################
## Word Clouds provide a further illustration of emotional emphasis within select speeches.  
library(tm)
cloudCorpus <- speechesCorpus

#Review SimpleCorpus 
(summary(cloudCorpus))
(length(cloudCorpus))

#Create Document Term Matrix from corpus
speechStops <- stopwords('english')
cloudDTM <- DocumentTermMatrix(cloudCorpus,
                               control = list(
                                 wordLengths=c(5, 15), 
                                 removePunctuation = TRUE,
                                 removeNumbers = TRUE,
                                 tolower=TRUE,
                                 remove_separators = TRUE,
                                 stopwords = TRUE,
                                 removeWords=speechStops
                               ))

#Transform DTM into MATRIX format for cleaning and normalization
cloudMAT <- as.matrix(cloudDTM)

#Word Cloud for OK and WV - HIGH HEALTH RISK INDICATORS IN DATA REPORTING SECTION
#Ref: https://towardsdatascience.com/create-a-word-cloud-with-r-bde3e7422e8a
library(wordcloud2)

(View(cloudMAT[1:50, 1:5]))
speechWords <- data.frame(cloudMAT)
#Create dataframe with words & frequency for OK state - row 36
OK_DF <- speechWords[36, ]
OK_words <- colnames(OK_DF)
OK_freq <- OK_DF[1, ]
OK_freq <- as.numeric(OK_freq[1, ])
OK_cloud <- data.frame(rbind(OK_words, OK_DF))
rownames(OK_cloud) <- c("word", "freq")
str(OK_cloud)

#Transpose in dataframe
OK_cloud_DF <- as.data.frame(t(OK_cloud))
OK_cloud_DF$word <- as.character(OK_cloud_DF$word)
OK_cloud_DF$freq <- as.numeric(as.character(OK_cloud_DF$freq))
str(OK_cloud_DF)

#WORD CLOUD: OK-Oklahoma
OKcloud2 <- wordcloud2(data = OK_cloud_DF, size = 1.6, color = 'random-dark')
OKcloud2

#Create dataframe with words & frequency for WV state - row 48
WV_DF <- speechWords[48, ]
WV_words <- colnames(WV_DF)
WV_freq <- WV_DF[1, ]
WV_freq <- as.numeric(WV_freq[1, ])
WV_cloud <- data.frame(rbind(WV_words, WV_DF))
rownames(WV_cloud) <- c("word", "freq")
str(WV_cloud)

#Transpose in dataframe
WV_cloud_DF <- as.data.frame(t(WV_cloud))
WV_cloud_DF$word <- as.character(WV_cloud_DF$word)
WV_cloud_DF$freq <- as.numeric(as.character(WV_cloud_DF$freq))
str(WV_cloud_DF)

#WV-West Virginia word cloud
WVcloud2 <- wordcloud2(data = WV_cloud_DF, size = 1.6, color = 'random-dark')
WVcloud2

#Comparative Word Cloud for SD - LOW HEALTH RISK INDICATORS IN DATA REPORTING SECTION
#Create dataframe with words & frequency for SD state - row 41
SD_DF <- speechWords[41, ]
SD_words <- colnames(SD_DF)
SD_freq <- SD_DF[1, ]
SD_freq <- as.numeric(SD_freq[1, ])
SD_cloud <- data.frame(rbind(SD_words, SD_DF))
rownames(SD_cloud) <- c("word", "freq")
str(SD_cloud)

#Transpose in dataframe
SD_cloud_DF <- as.data.frame(t(SD_cloud))
SD_cloud_DF$word <- as.character(SD_cloud_DF$word)
SD_cloud_DF$freq <- as.numeric(as.character(SD_cloud_DF$freq))
str(SD_cloud_DF)

#SD-South Dakota word cloud
SDcloud2 <- wordcloud2(data = SD_cloud_DF, size = 1.6, color = 'random-dark')
SDcloud2

###########################################
#Other possible sentiment analysis from AFINN and bing lexicons
#The bing lexicon categorizes words in a binary fashion into positive and negative categories.
#AFINN lexicon assigns words with a score that runs between -5 and 5, with negative scores indicating 
#negative sentiment and positive scores indicating positive sentiment.

#afinn_AL <- get_sentiment(ALwords, method="afinn")
#bing_AL <- get_sentiment(ALwords, method="bing")
#AL_sentiments <- data.frame(bing_AL, afinn_AL, ALwords)

#Viz1: Sentiment scores across methods
#AL2 <- plot_ly(AL_sentiments, x=~ALwords, y=~bing_AL, type="scatter", mode="jitter", name="bing") %>%add_trace(y=~afinn_AL, mode="lines", name="afinn") %>%layout(title="Word sentiments for Alabama Speech",yaxis=list(title="score"), xaxis=list(title="word"))

###########################################
## Additional Working Datasets for Models and Analysis:    
  
## emotionsStates_DF - Transformed DATAFRAME with association counts for emotions and sentiments + 
#Percent Positive in speech + Factors: politicalParty, stateAbbrev (for ggplot)  

## positivePercentOrder - Ordered emotionsStates_DF by Percent Positive (low to high) 

#################################################################################################
## About the Data - 2019 County Health Rankings (reporting CSV on 5 slected indicators)
## Data Question: What are the key indicators and social determinants of health that elected 
#leaders may be responding or reacting to when delivering a speech? 
#################################################################################################
library(arules)
file4="state-health-indicators-full.csv" 
rawHealth <- read.csv(file4, header=TRUE, na.strings="NA")

#New DATAFRAME to put in cleased data (and keep rawHealth untouched)
cleanHealth <- rawHealth

#Create Data Dictionary for 'state-health-indicators-full.csv'
h_data_dictionary_col <- c("County-level FIPS codes", "state", "county", 
                           "x1Days", "x1CIL", "x1CIH", "x1Z", 
                           "x2Days", "x2CIL", "x2CIH", "x2Z", 
                           "x3Num", "x3Per", "x3CIL", "x3CIH", "x3Z", 
                           "x4Num", "x4Tot", "x4Per", "x4Z", 
                           "x5Per", "x5CIL", "x5CIH", "x5Z")

h_data_dictionary_def <- c("FIPS", "state", "county", 
                           "Physically Unhealthy Days. Average number of physically unhealthy days reported in past 30 days (age-adjusted)", "95% CI - Low", "95% CI - High", "County z-scoring for standardization", 
                           "Mentally Unhealthy Days. Average number of mentally unhealthy days reported in past 30 days (age-adjusted)", "95% CI - Low", "95% CI - High", "County z-scoring for standardization", 
                           "Number uninsured", "Percentage of population under age 65 without health insurance", "95% CI - Low", "95% CI - High", "County z-scoring for standardization", 
                           "Number unemployed", "Total labor force number", "Percentage of population ages 16 and older unemployed but seeking work", "County z-scoring for standardization", 
                           "Percentage of households with at least 1 of 4 housing problems: overcrowding, high housing costs, lack of kitchen facilities, or lack of plumbing facilities", "95% CI - Low", "95% CI - High", "County z-scoring for standardization")
h_data_dictionary_index <- data.frame(h_data_dictionary_col, h_data_dictionary_def)
colnames(h_data_dictionary_index) <- c("Column", "Definition")

library(knitr)
library(kableExtra)
#Data dictionary for Health Indicators
kable(h_data_dictionary_index) %>%
  kable_styling(bootstrap_options = "striped", full_width = F)

###########################################
#Health indicator data cleaning and preparation
#Check health indicator data types
(str(cleanHealth)) #Data types look correct (factors, ints and nums where appropriate)
#Check for missing values (total number of NAs in the data frame)
(sum(is.na(cleanHealth)))
#View the total number of NAs by column
(sapply(cleanHealth, function(x) sum(is.na(x))))
# View rows with NAs
cleanHealth[(is.na(cleanHealth$x1Z)==TRUE),]
cleanHealth[(is.na(cleanHealth$x2Z)==TRUE),]
cleanHealth[(is.na(cleanHealth$x3Num)==TRUE),]
cleanHealth[(is.na(cleanHealth$x3Per)==TRUE),]
cleanHealth[(is.na(cleanHealth$x3CIL)==TRUE),]
cleanHealth[(is.na(cleanHealth$x3CIH)==TRUE),]
cleanHealth[(is.na(cleanHealth$x3Z)==TRUE),]
cleanHealth[(is.na(cleanHealth$x4Num)==TRUE),]
cleanHealth[(is.na(cleanHealth$x4Tot)==TRUE),]
cleanHealth[(is.na(cleanHealth$x4Per)==TRUE),]
cleanHealth[(is.na(cleanHealth$x4Z)==TRUE),]
cleanHealth[(is.na(cleanHealth$x5CIL)==TRUE),]
cleanHealth[(is.na(cleanHealth$x5CIH)==TRUE),]
cleanHealth[(is.na(cleanHealth$x5Z)==TRUE),]
#NA summary: 1 row with NAs for x3 and x4 indicators, 61 NAs for Z scores (all the same rows for each NA in Z score)

#Create a dataframe with ONLY health indicator data by county
#Row removed: the one row with NAs for health indicators  x3 and x4
#Columns removed: FIPS identifier column and statistical columns (CIL, CIH, Z scores)

#View row with NA in health indicator columns
(NArows<-rownames(cleanHealth[(is.na(cleanHealth$x3Num)==TRUE),]))
#View column names to determine which are statistically based (to remove)
(colnames(cleanHealth))
#Add only applicable columns/rows to new data frame (healthIndicators_DF)
(healthIndicators_DF <- cleanHealth[-548,-c(1,5:7,9:11,14:16,20,22:24)])
#Double check NAs were removed
(sum(is.na(healthIndicators_DF)))

#A dataframe with indicator data by county AND statistical columns - to be used for summary stats
#Row removed: one row with NAs for health indicators x3 and x4
#Columns removed: the 61 rows with NA for Z values
(healthIndicatorsWithStats_DF <- cleanHealth[-c(68, 73, 91, 95, 254, 272, 285, 301, 405, 548, 563, 567, 981, 987, 989, 1428, 1603, 1616, 1620, 
                                                1623, 1632, 1635, 1649, 1652, 1656, 1657, 1658, 1682, 1691, 1696, 1699, 1705, 1710, 1711, 1712,  
                                                1736, 1739, 1745, 1751, 1752, 1806, 1993, 2022, 2031, 2033, 2242, 2392, 2397, 2398, 2420, 2539,  
                                                2609, 2640, 2653, 2654, 2657, 2673, 2678, 2719, 2738, 2781, 2792),])
#Double check NAs were removed
(sum(is.na(healthIndicatorsWithStats_DF)))

###########################################
## OUTLIERS examination
#Ref: https://www.r-bloggers.com/identify-describe-plot-and-remove-the-outliers-from-the-dataset/
#Parameters:
# dt = data 
# var = variable 
# varName = variable name/description
outlierKD <- function(dt, var, varName) {
  var_name <- eval(substitute(var),eval(dt))
  na1 <- sum(is.na(var_name))
  m1 <- mean(var_name, na.rm = T)
  par(mfrow=c(2, 2), oma=c(0,0,3,0))
  boxplot(var_name, main="With outliers", col="light yellow")
  hist(var_name, main="With outliers", xlab=NA, ylab=NA, col="light yellow")
  outlier <- boxplot.stats(var_name)$out
  mo <- mean(outlier)
  var_name <- ifelse(var_name %in% outlier, NA, var_name)
  boxplot(var_name, main="Without outliers", col="light blue")
  hist(var_name, main="Without outliers", xlab=NA, ylab=NA, col="light blue")
  title1=paste(varName,"Outlier Check", sep=" ")
  title(title1, outer=TRUE)
  na2 <- sum(is.na(var_name))
  cat("Outliers identified:", na2 - na1, "\n")
  cat("Proportion (%) of outliers:", round((na2 - na1) / sum(!is.na(var_name))*100, 1), "\n")
  cat("Mean of the outliers:", round(mo, 2), "\n")
  m2 <- mean(var_name, na.rm = T)
  cat("Mean without removing outliers:", round(m1, 2), "\n")
  cat("Mean if we remove outliers:", round(m2, 2), "\n")
  #response <- readline(prompt="Do you want to remove outliers and to replace with NA? [yes/no]: ")
  #if(response == "y" | response == "yes"){
  #  dt[as.character(substitute(var))] <- invisible(var_name)
  #  assign(as.character(as.list(match.call())$dt), dt, envir = .GlobalEnv)
  #  cat("Outliers successfully removed", "\n")
  #  return(invisible(dt))
  #} else{
  #  cat("Nothing changed", "\n")
  #  return(invisible(var_name))
  #}
}

#Outlier check for physical health days (x1Days)
outlierKD(healthIndicators_DF, healthIndicators_DF$x1Days, "Physical Health Days")
#Outlier check for mental health days (x2Days)
outlierKD(healthIndicators_DF, healthIndicators_DF$x2Days, "Mental Health Days")
#Outlier check for number of uninsured (x3Num)
outlierKD(healthIndicators_DF, healthIndicators_DF$x3Num, "Number of Uninsured")
#Outlier check for percent uninsured (x3Per)
outlierKD(healthIndicators_DF, healthIndicators_DF$x3Per, "Percentage Uninsured")
#Outlier check for number of unemployed (x4Num)
outlierKD(healthIndicators_DF, healthIndicators_DF$x4Num, "Number of Unemployed")
#Outlier check for total labor force (x4Tot)
outlierKD(healthIndicators_DF, healthIndicators_DF$x4Tot, "Total Labor Force")
#Outlier check for percent unemployed (x4Per)
outlierKD(healthIndicators_DF, healthIndicators_DF$x4Per, "Percentage Unemployed")
#Outlier check for percentage with severe housing conditions (x5Per)
outlierKD(healthIndicators_DF, healthIndicators_DF$x5Per, "Percentage with Severe Housing Conditions")

## Decision made not to remove outliers** from health indicators dataset since actual measurements 
#can not be verified as accurate or inaccurate.  

###########################################
## AGGREGATE County Health Indicators to State - sum by state (using 'plyr' package)
library(plyr)

healthIndicators_byState <- ddply(healthIndicators_DF,.(state), numcolwise(sum))
(str(healthIndicators_byState))

#Exclude percentages (x3Per, x4Per, X5Per)
healthIndicators_byState <- healthIndicators_byState[,-c(5,8,9)]
(str(healthIndicators_byState))

#Relative-to-Populations and Comparative Normalization for County Health Indicators to State
#WORKING DATASET: 'healthIndicators_byState'

#Convert x1 and x2 to average days per month (365/12 = 30.42 days)
#New standardized comparative = sum the avg. number of days of poor health in state (total county values) 
#divided by 30.42 (average number of days in a month) for the percentage of days in a month 
#that someone in the State is having physically or mentally unhealthy day (somewhere in that state).
healthIndicators_byState_x1 <- (healthIndicators_byState[ , 2] / 30.42)
healthIndicators_byState_x2 <- (healthIndicators_byState[ , 3] / 30.42)

#Add state populations to dataframe
#State populations source: U.S. Census Bureau, Population Division. Release Date: December 2019
file5="census-state-populations.csv"
statePop <-read.csv(file5, stringsAsFactor=FALSE)
(View(statePop))
#Check for alignment
#(View(healthIndicators_byState$state))
#Check str() and change population to number
#(str(statePop))
statePop$population <- as.numeric(statePop$population)
#Add state population to dataset and clean up
healthIndicators_byState <- cbind(healthIndicators_byState, statePop)
healthIndicators_byState$stateName <- NULL

#Convert x3
healthIndicators_byState_x3 <- (healthIndicators_byState[ , 4]/healthIndicators_byState[ , 7])

#Convert x4 which is x4Num/x4Tot
healthIndicators_byState_x4 <- (healthIndicators_byState[ , 5]/healthIndicators_byState[ , 6])

#Build new DATAFRAME and convert to normalized relative frequency for all variables
healthIndicators_byState_Relative <- data.frame(healthIndicators_byState$state, healthIndicators_byState$population, healthIndicators_byState_x1, healthIndicators_byState_x2, healthIndicators_byState_x3, healthIndicators_byState_x4)
(str(healthIndicators_byState_Relative))
(sum(is.na(healthIndicators_byState_Relative)))

#Rename columns
colnames(healthIndicators_byState_Relative) <- c("state", "population", "x1AvgDays", "x2AvgDays", "x3NumPop", "x4NumTot")

###########################################
## Updated Working Datasets for Models and Analysis:    

## healthIndicators_byState_Relative - Transformed DATAFRAME of health indicators with applied relative population features  
## emotionsStates_DF - Transformed DATAFRAME with association counts for emotions and sentiments + Percent Positive in 
#speech + Factors: politicalParty, stateAbbrev 

###########################################
## DATA TRANSFORMATION: Normalize data and create MASTER DATAFRAMEfor modeling and analysis
#Convert emotions by state 'counts' to % of emotions words (of all the words said by a state, % that were X - comparative)
emotionsNUM <- emotionsStates_DF
emotionsNUM$politicalParty <- NULL
emotionsNUM$stateAbbrev <- NULL
emotionsNUM$percentPositive <- NULL

#Apply normalization  
emotionsNum_Norm <- apply(emotionsNUM, 1, function(x) round(x/sum(x),3))
#View output
(emotionsNum_Norm[1:5,1:5])
#Transpose output'speech' and 'words'
emotionsNum_Norm <- t(emotionsNum_Norm)
#Recheck results
(emotionsNum_Norm[1:5,1:5])

#Create DATAFRAME formats from normalized MATRIX 'speechMAT_Norm'
emotionsNum_Norm_DF <- as.data.frame(as.matrix(emotionsNum_Norm))
(str(emotionsNum_Norm_DF))
#Check if any blanks/NAs in dataframe
(sum(is.na(emotionsNum_Norm_DF)))

#Add back 'politicalParty', 'stateAbbrev' and 'percentPositive'
politicalPartyEmotions <- emotionsStates_DF$politicalParty
stateAbbrevEmotions <- emotionsStates_DF$stateAbbrev
statePositive <- emotionsStates_DF$percentPositive
emotionsNorm_DF <- data.frame(politicalPartyEmotions, stateAbbrevEmotions, statePositive, emotionsNum_Norm_DF)
(head(emotionsNorm_DF))

#Rename columns 
colnames(emotionsNorm_DF) <- c("politicalParty", "stateAbbrev", "posSentiment", "anger", 
                               "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "negative", "positive")

#ASSEMBLE MASTER DATAFRAME
state_emotions_health_DF <- data.frame(emotionsNorm_DF, healthIndicators_byState_Relative)

#Check dataframe
#state_emotions_health_DF[1:20, ]
#(str(state_emotions_health_DF))

#Reorder columns for project team using 'tibble'
library(tibble)
state_emotions_health <- as_data_frame(state_emotions_health_DF)
(colnames(state_emotions_health))
state_emotions_health2 <- state_emotions_health[,c(14, 15, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19)]
state_emotions_health <- as.data.frame(state_emotions_health2)
state_emotions_health_DF[1:20, ]

###########################################
## Master Dataframe:  
#'state_emotions_health' with class labels and variable attributes (50 observations of 19 variables):
#+ state - State name (nominal factor)
#+ population - State population, 2019 census forecast
#+ stateAbbrev - State abbreviation (nominal factor)
#+ politicalParty - Political Party, Republican or Democratic (factor 2 class levels)
#+ posSentiment - Percentage postive sentiment in Governor speech
#+ anger - % of Governor's speech words that were emotion = anger
#+ anticipation - % of Governor's speech words that were emotion = anticipation
#+ disgust - % of Governor's speech words that were emotion = disgust
#+ fear - % of Governor's speech words that were emotion = fear
#+ joy - % of Governor's speech words that were emotion = joy
#+ sadness - % of Governor's speech words that were emotion = sadness
#+ surprise - % of Governor's speech words that were emotion = surprise
#+ trust - % of Governor's speech words that were emotion = trust
#+ negative - % of Governor's speech words that were sentiment = negative
#+ positive - % of Governor's speech words that were sentiment = positive
#+ x1AvgDays - % of days in a month that someone in the State is having a physically unhealthy day
#+ x2AvgDays - % of days in a month that someone in the State is having a mentally unhealthy day
#+ x3NumPop - % uninsured of state population (number uninsured/state population)
#+ x4NumTot - % unemployed of state total labor force  

###########################################
## DATA TRANFORMATION: Discretization
## Create FIRST FUNCTION to discretize a variable into three categories (low, medium, high)
#low = bottom third of variable values
#medium = middle third of variable values
#high = top third of variable values

discretizeLMH <- function(var) {
  # Variable min value
  varMin<-min(var)
  # Variable max value
  varMax<-max(var)
  # Num bins to divide variable into
  bins <- 3
  # Width of each bin
  width <- ((varMax-varMin)/bins)
  # Low if value is bottom third of variable values
  # Medium if between bottom and top third of variable values
  # High if value is top third of variable values
  return(as.factor(ifelse(var < (varMin+width), "low", 
                          (ifelse(var >= (varMin+(2*width)),"high",
                                  "medium")))))
}  

## Create SECOND FUNCTION to discretize for arules
#Discretize a variable into three unique categories (low, medium, high) with the variable name appended to the value ex. low-sentiment
#low = bottom third of variable values
#medium = middle third of variable values
#high = top third of variable values

discretizeLMHname <- function(var, vname) {
  # Variable min value
  varMin<-min(var)
  # Variable max value
  varMax<-max(var)
  # Num bins to divide variable into
  bins <- 3
  # Width of each bin
  width <- ((varMax-varMin)/bins)
  # Low if value is bottom third of variable values
  # Medium if between bottom and top third of variable values
  # High if value is top third of variable values
  return(as.factor(ifelse(var < (varMin+width), paste0("low",vname), 
                          (ifelse(var >= (varMin+(2*width)),paste0("high", vname) ,
                                  paste0("medium", vname))))))
}  

## APPLY discretized functions for new dataframes
#New data frame with discretized variables (include state and county)
disc_healthIndicators_DF <- state_emotions_health_DF[,16:19]

#New data frame with discretized variables (include state and county) for association rules
disc_healthIndicators_DF2 <- state_emotions_health_DF[,16:19]

#Call discretizeLMH function to discretize physical health days (x1AvgDays) variable
disc_healthIndicators_DF$x1AvgDays_Disc<-discretizeLMH(state_emotions_health_DF$x1AvgDays)
(str(disc_healthIndicators_DF$x1AvgDays_Disc))
(summary(disc_healthIndicators_DF$x1AvgDays_Disc))

#Call discretizeLMHname function to uniquely discretize physical health days (x1AvgDays) variable 
disc_healthIndicators_DF2$x1AvgDays_Disc<-discretizeLMHname(state_emotions_health_DF$x1AvgDays, "-x1")
(str(disc_healthIndicators_DF2$x1AvgDays_Disc))
(summary(disc_healthIndicators_DF2$x1AvgDays_Disc))

#Call discretizeLMH function to discretize mental health days (x2AvgDays) variable
disc_healthIndicators_DF$x2AvgDays_Disc<-discretizeLMH(state_emotions_health_DF$x2AvgDays)
(str(disc_healthIndicators_DF$x2AvgDays_Disc))
(summary(disc_healthIndicators_DF$x2AvgDays_Disc))

#Call discretizeLMHname function to uniquely discretize mental health days (x2AvgDays) variable
disc_healthIndicators_DF2$x2AvgDays_Disc<-discretizeLMHname(state_emotions_health_DF$x2AvgDays, "-x2")
(str(disc_healthIndicators_DF2$x2AvgDays_Disc))
(summary(disc_healthIndicators_DF2$x2AvgDays_Disc))

#Call discretizeLMH function to discretize number uninsured (x3NumPop) variable
disc_healthIndicators_DF$x3NumPop_Disc<-discretizeLMH(state_emotions_health_DF$x3NumPop)
(str(disc_healthIndicators_DF$x3NumPop_Disc))
(summary(disc_healthIndicators_DF$x3NumPop_Disc))

#Call discretizeLMHname function to uniquely discretize number uninsured (x3NumPop) variable
disc_healthIndicators_DF2$x3NumPop_Disc<-discretizeLMHname(state_emotions_health_DF$x3NumPop, "-x3")
(str(disc_healthIndicators_DF2$x3NumPop_Disc))
(summary(disc_healthIndicators_DF2$x3NumPop_Disc))

#Call discretizeLMH function to discretize number unemployed (x4NumTot) variable
disc_healthIndicators_DF$x4NumTot_Disc<-discretizeLMH(state_emotions_health_DF$x4NumTot)
(str(disc_healthIndicators_DF$x4NumTot_Disc))
(summary(disc_healthIndicators_DF$x4NumTot_Disc))

#Call discretizeLMHname function to uniquely discretize number unemployed (x4NumTot) variable
disc_healthIndicators_DF2$x4NumTot_Disc<-discretizeLMHname(state_emotions_health_DF$x4NumTot, "-x4")
(str(disc_healthIndicators_DF2$x4NumTot_Disc))
(summary(disc_healthIndicators_DF2$x4NumTot_Disc))

#Remove initial numeric indicator columns
disc_healthIndicators_DF <- disc_healthIndicators_DF[,5:8]

#Remove initial numeric indicator columns for the association rules frame
disc_healthIndicators_DF2 <- disc_healthIndicators_DF2[,5:8]

#Check discretized data frame summaries - DF and DF2
(summary(disc_healthIndicators_DF))
(str(disc_healthIndicators_DF))

(summary(disc_healthIndicators_DF2))
(str(disc_healthIndicators_DF2))

###########################################
#Add discretized variables to master state_meotions_health data frame
state_emotions_health$x1AvgDays_Disc <- disc_healthIndicators_DF$x1AvgDays_Disc
state_emotions_health$x2AvgDays_Disc <- disc_healthIndicators_DF$x2AvgDays_Disc
state_emotions_health$x3NumPop_Disc <- disc_healthIndicators_DF$x3NumPop_Disc
state_emotions_health$x4NumTot_Disc <- disc_healthIndicators_DF$x4NumTot_Disc
(str(state_emotions_health))

###########################################
## Updated Master Dataframe:  
#'state_emotions_health' with class labels, attribute variables, 
#and discretized factors (50 observations of 23 variables) 

###########################################
## TRANSACTION formats for ARM - convert dataframes to type transaction
#stateParty <- Add state/party

#Create new data frame with more readable discretized values to create transactions with
disc_Transform <- disc_healthIndicators_DF
disc_Transform$x1AvgDays_Disc <- as.factor(ifelse(disc_healthIndicators_DF$x1AvgDays_Disc=="low","x1=Low",
                                                  ifelse(disc_healthIndicators_DF$x1AvgDays_Disc=="high","x1=High","x1=Medium")))
disc_Transform$x2AvgDays_Disc <- as.factor(ifelse(disc_healthIndicators_DF$x2AvgDays_Disc=="low","x2=Low",
                                                  ifelse(disc_healthIndicators_DF$x2AvgDays_Disc=="high","x2=High","x2=Medium")))
disc_Transform$x3NumPop_Disc <- as.factor(ifelse(disc_healthIndicators_DF$x3NumPop_Disc=="low","x3=Low",
                                                 ifelse(disc_healthIndicators_DF$x3NumPop_Disc=="high","x3=High","x3=Medium")))
disc_Transform$x4NumTot_Disc <- as.factor(ifelse(disc_healthIndicators_DF$x4NumTot_Disc=="low","x4=Low",
                                                 ifelse(disc_healthIndicators_DF$x4NumTot_Disc=="high","x4=High","x4=Medium")))
#Check data types
(str(disc_Transform))

library(arules)
#Create transactions list with updated values
healthTransactions <- as(disc_Transform, "transactions")

#Includes state, party, discretized health indicators
healthInd_party <- state_emotions_health[,c(1,4,20:23)]

#Create new data frame with more readable discretized values to create transactions with
disc2_Transform <- disc_Transform
disc2_Transform$politicalParty <- as.factor(paste("Party=",state_emotions_health$politicalParty,sep=""))
disc2_Transform$state <- as.factor(paste("State=",state_emotions_health$state,sep=""))

#Check data types
(str(disc2_Transform))

#Create transactions list with updated values
healthPlusTransactions <- as(disc2_Transform, "transactions")
(View(healthPlusTransactions))

#Transaction data set with only political party and health indicators
disc_Transform <- disc_healthIndicators_DF
disc3_Transform <- disc_Transform
disc3_Transform$politicalParty <- as.factor(state_emotions_health$politicalParty)

disc4_Transform <- disc_healthIndicators_DF2 #Transaction data set with discretized posSentiment
disc4_Transform$Sentiment <- discretizeLMHname(state_emotions_health$posSentiment, "-sentiment") #Validate that sentiment is a factor
disc4_Transform$politicalParty <- disc2_Transform$politicalParty
disc4_Transform$state <- disc2_Transform$state

healthPartyTransactions <- as(disc3_Transform, "transactions")
(View(healthPartyTransactions))

healthPartyTransactions2 <- as(disc4_Transform, "transactions")
(View(healthPartyTransactions2))

###########################################
## Updated Datasets:  
#'state_emotions_health' - Master dataset with class labels, attribute variables, 
#and discretized health indicator factor classes (50 observations of 23 variables):  
#+ healthPlusTransactions - ARM transactions data set with readable, discretized indicator variables AND state AND political party (50 transaction rows and 64 items)
#+ disc_healthIndicators_DF: contains only discretized health indicator variables
#+ disc_Transform: data frame used to transform disc_healthIndicators_DF variables for easier ARM readability
#+ disc2_Transform: data frame used to transform disc_healthIndicators_DF variables PLUS political party and state for easier ARM readability
#+ healthTransactions and healthPartyTransactions2: transactions data set with only readable, discretized indicator variables  

#################################################################################################
## About the Data - Additional Exploratory Data Visualizations (dataset: 'healthindicators_DF')
#################################################################################################
#Data descriptions and visualizations (variables)
library(ggplot2)
library(ggmap)

#Boxplots by variable
#boxplot(healthIndicators_DF$x1Days, main="Poor Mental Health Days")
#boxplot(healthIndicators_DF$x2Days, main="Poor Physical Health Days")
#boxplot(healthIndicators_DF$x3Num, main="Number Uninsured")
#boxplot(healthIndicators_DF$x3Per, main="Percentage Uninsured")
#boxplot(healthIndicators_DF$x4Num, main="Number Unemployed")
#boxplot(healthIndicators_DF$x4Per, main="Percentage Unemployed")
#boxplot(healthIndicators_DF$x5Per, main="Percentage with Severe Housing Problems")

#Histograms by variable
#hist(healthIndicatorsWithStats_DF) 
#hist(healthIndicators_DF$x1Days, main="Poor Mental Health Days")
#hist(healthIndicators_DF$x2Days, main="Poor Physical Health Days")
#hist(healthIndicators_DF$x3Num, main="Number Uninsured")
#hist(healthIndicators_DF$x3Per, main="Percentage Uninsured")
#hist(healthIndicators_DF$x4Num, main="Number Unemployed")
#hist(healthIndicators_DF$x4Per, main="Percentage Unemployed")
#hist(healthIndicators_DF$x5Per, main="Percentage with Severe Housing Problems")

#Exploratory Density Plots
(gPhysHealthDays_DP <- ggplot(healthIndicators_DF, aes(x=healthIndicators_DF$x1Days)) + geom_histogram(aes(y=..density..)) + geom_density(alpha=.2, fill="Green") + labs(title="Poor Physical Health Days"))# + facet_grid(. ~ healthIndicators_DF$state)
(gMentHealthDays <- ggplot(healthIndicators_DF, aes(x=healthIndicators_DF$x2Days)) + geom_histogram(aes(y=..density..)) + geom_density(alpha=.2, fill="Blue") + labs(title="Poor Mental Health Days"))
(gUninsuredPer <- ggplot(healthIndicators_DF, aes(x=healthIndicators_DF$x3Per)) + geom_histogram(aes(y=..density..)) + geom_density(alpha=.2, fill="Red") + labs(title="Percentage Uninsured"))
(gUnemploymentPer <- ggplot(healthIndicators_DF, aes(x=healthIndicators_DF$x4Per)) + geom_histogram(aes(y=..density..)) + geom_density(alpha=.2, fill="Orange") + labs(title="Percentage Unemployed"))
(gHousing <- ggplot(healthIndicators_DF, aes(x=healthIndicators_DF$x5Per)) + geom_histogram(aes(y=..density..)) + geom_density(alpha=.2, fill="Yellow") + labs(title="Percentage with Severe Housing Problems"))

#Boxplots by state
b1 <- gPhysHealthDays_BP <- ggplot(healthIndicators_DF, aes(x=healthIndicators_DF$state,y=healthIndicators_DF$x1Days, fill=healthIndicators_DF$state)) + geom_boxplot(show.legend = FALSE) + theme(axis.text.x = element_text(angle = 90)) + labs(title="Poor Physical Health Days") #, x="States", y="Days"
b1
b2 <- gMentHealthDays_BP <- ggplot(healthIndicators_DF, aes(x=healthIndicators_DF$state,y=healthIndicators_DF$x2Days, fill=healthIndicators_DF$state)) + geom_boxplot(show.legend = FALSE) + theme(axis.text.x = element_text(angle = 90)) + labs(title="Poor Mental Health Days") #, x="States", y="Days"
b2
(gUninsuredPer_BP <- ggplot(healthIndicators_DF, aes(x=healthIndicators_DF$state,y=healthIndicators_DF$x3Per, fill=healthIndicators_DF$state)) + geom_boxplot(show.legend = FALSE) + theme(axis.text.x = element_text(angle = 90)) + labs(title="Percentage Uninsured"))#, x="States", y="Days"))
(gUnemploymentPer_BP <- ggplot(healthIndicators_DF, aes(x=healthIndicators_DF$state,y=healthIndicators_DF$x4Per, fill=healthIndicators_DF$state)) + geom_boxplot(show.legend = FALSE) + theme(axis.text.x = element_text(angle = 90)) + labs(title="Percentage Unemployed"))#, x="States", y="Days"))
(gHousing_BP <- ggplot(healthIndicators_DF, aes(x=healthIndicators_DF$state,y=healthIndicators_DF$x5Per, fill=healthIndicators_DF$state)) + geom_boxplot(show.legend = FALSE) + theme(axis.text.x = element_text(angle = 90)) + labs(title="Percentage with Severe Housing Problems"))#, x="States", y="Days"))

#Maps reporting dataframe
healthIndicators_DF$stateReporting <- healthIndicators_DF$state
healthIndicators_DF$state <- tolower(healthIndicators_DF$state)
(head(healthIndicators_DF))

#Maps base
us <- map_data("state") 

#x1 Poor Physical Health Days
m1 <- ggplot(healthIndicators_DF, aes(map_id=state))
m1 <- m1 + geom_map(map=us, aes(fill=healthIndicators_DF$x1Days))
m1 <- m1 + expand_limits(x = us$long, y = us$lat)
m1 <- m1 + coord_map() 
m1 <- m1 + scale_fill_distiller(palette = "Spectral")
m1 <- m1 + theme(axis.text.x = element_blank(),
                 axis.text.y = element_blank(),
                 axis.ticks = element_blank(),
                 rect = element_blank())
m1 <- m1 + labs(title = "Average Number of Poor Physical Health Days (30-Days) by U.S. State", fill = "Days")
m1

#x2 Poor Mental Health Days
m2 <- ggplot(healthIndicators_DF, aes(map_id=state))
m2 <- m2 + geom_map(map=us, aes(fill=healthIndicators_DF$x2Days))
m2 <- m2 + expand_limits(x = us$long, y = us$lat)
m2 <- m2 + coord_map() 
m2 <- m2 + scale_fill_distiller(palette = "Spectral")
m2 <- m2 + theme(axis.text.x = element_blank(),
                 axis.text.y = element_blank(),
                 axis.ticks = element_blank(),
                 rect = element_blank())
m2 <- m2 + labs(title = "Average Number of Poor Mental Health Days (30-Days) by U.S. State", fill = "Days")
m2

#x3 Uninsured
m3 <- ggplot(healthIndicators_DF, aes(map_id=state))
m3 <- m3 + geom_map(map=us, aes(fill=healthIndicators_DF$x3Per))
m3 <- m3 + expand_limits(x = us$long, y = us$lat)
m3 <- m3 + coord_map() 
m3 <- m3 + scale_fill_distiller(palette = "Spectral")
m3 <- m3 + theme(axis.text.x = element_blank(),
                 axis.text.y = element_blank(),
                 axis.ticks = element_blank(),
                 rect = element_blank())
m3 <- m3 + labs(title = "Percentage of Uninsured Persons under Age 65 by U.S. State", fill = "Percentage")
m3

#x4 Unemployment
m4 <- ggplot(healthIndicators_DF, aes(map_id=state))
m4 <- m4 + geom_map(map=us, aes(fill=healthIndicators_DF$x4Per))
m4 <- m4 + expand_limits(x = us$long, y = us$lat)
m4 <- m4 + coord_map() 
m4 <- m4 + scale_fill_distiller(palette = "Spectral")
m4 <- m4 + theme(axis.text.x = element_blank(),
                 axis.text.y = element_blank(),
                 axis.ticks = element_blank(),
                 rect = element_blank())
m4 <- m4 + labs(title = "Percentage of Unemployed Persons over Age 16 by U.S. State", fill = "Percentage")
m4

#x5 Housing Issues
m5 <- ggplot(healthIndicators_DF, aes(map_id=state))
m5 <- m5 + geom_map(map=us, aes(fill=healthIndicators_DF$x5Per))
m5 <- m5 + expand_limits(x = us$long, y = us$lat)
m5 <- m5 + coord_map() 
m5 <- m5 + scale_fill_distiller(palette = "Spectral")
m5 <- m5 + theme(axis.text.x = element_blank(),
                 axis.text.y = element_blank(),
                 axis.ticks = element_blank(),
                 rect = element_blank())
m5 <- m5 + labs(title = "Percentage of Households with 1-4 Housing Problems by U.S. State", fill = "Percentage")
m5

## Project notation: The x5 (Severe Housing) variable was removed for additional analysis because it 
#does not include by-county numbers necessary for normalizing by state % of population.  

#EDA continued with detailed maps with political parties and health indicators

partyHealthState<-state_emotions_health[,c(1,2,4,16:19,20:23)]
#16:19/20:23 includes both discretized and continuous health indicator columns
partyHealthState$state <- tolower(partyHealthState$state)

#Import states.csv which has a central latitude/longitude for each state - to be used for mapping points by state
#State coordinates from: https://developers.google.com/public-data/docs/canonical/states_csv
file6 = "states.csv"
st<-read.csv(file6)
st$name<-tolower(st$name)
newLoc <- merge(partyHealthState,st,by.x="state", by.y="name")
#Drop Alaska/Hawaii
newStates <- newLoc[-c(2,11),]
(str(newStates))

#Map of Poor Physical Health Days by Political Party and US State
us <- map_data("state") 
pm1 <- ggplot(partyHealthState, aes(map_id=state))
pm1 <- pm1 + geom_map(map=us, aes(fill=partyHealthState$politicalParty), color="gray") + theme(legend.title=element_text())
pm1 <- pm1 + geom_point(data=newStates, aes(x=lon, y=lat, size=5, color=x1AvgDays_Disc)) 
pm1 <- pm1 + scale_color_manual(values=c("#F75151", "#6AF751", "#F4F751")) # red, green, yellow
pm1 <- pm1 + scale_fill_manual(values=c("#758CB4", "#C66F6F"))#"#9CBEFA", "#FA9C9C")) #blue, then red
pm1 <- pm1 + expand_limits(x = us$long, y = us$lat)
pm1 <- pm1 + coord_map()
pm1 <- pm1 + ggtitle("Poor Physical Health Days by US State") + theme(axis.title.x=element_blank(), axis.title.y=element_blank())
pm1 <- pm1 + labs(color="Number of Poor Physical Health Days (per month)",  fill="Political Party") + guides(size=FALSE)
pm1

#Map of Poor Mental Health Days by Political Party and US State
us <- map_data("state") 
pm2 <- ggplot(partyHealthState, aes(map_id=state))
pm2 <- pm2 + geom_map(map=us, aes(fill=partyHealthState$politicalParty), color="gray") + theme(legend.title=element_text())
pm2 <- pm2 + geom_point(data=newStates, aes(x=lon, y=lat, size=5, color=x2AvgDays_Disc)) 
pm2 <- pm2 + scale_color_manual(values=c("#F75151", "#6AF751", "#F4F751")) # red, green, yellow
pm2 <- pm2 + scale_fill_manual(values=c("#758CB4", "#C66F6F"))#"#9CBEFA", "#FA9C9C")) #blue, then red
pm2 <- pm2 + expand_limits(x = us$long, y = us$lat)
pm2 <- pm2 + coord_map()
pm2 <- pm2 + ggtitle("Poor Mental Health Days by US State") + theme(axis.title.x=element_blank(), axis.title.y=element_blank())
pm2 <- pm2 + labs(color="Number of Poor Mental Health Days (per month)",  fill="Political Party") + guides(size=FALSE)
pm2

#################################################################################################
## Results from Models for Analysis

## With completion of data transformations and exploratory data analysis from emotion and sentiment 
#analysis and state indicators and determinants of health, the project team examined a set of data 
#questions through applied data mining techniques and classification algorithms. 
#Those questions included:
  
#+ Do political party positions, as expressed through their words, elicit certain emotions that may have an impact on our health and well-being?  
#+ Do certain emotions and sentiments, as shared by our leaders, persuade or constrain our general health and well-being?  
#+ Can certain expressed emotions predict our general health and well-being status?  
#+ Can certain health indicators actually be attributed to a specific political party?  
#+ Do political parties recognize that they are associated with specific health and well-being conditions in states?  
  
## These examinations are highlighted in sections that follow and categorized by applied learning model and technique. 
#################################################################################################
  
#################################################################################################
## Association Rules Mining (ARM)  
#Unsupervised learning alogorithm for identifying co-occurrence frequency/probability of 
#items in a transaction.
#################################################################################################
library(plyr)
library(dplyr)
library(arules)

#Apply apriori algorithm
healthRules <- arules::apriori(data=healthPartyTransactions2, parameter = list(supp=0.2, conf = 0.8, minlen=3))
healthRules <- arules::sort(healthRules, decreasing = TRUE, by='confidence')
inspect(healthRules)

#Additional explorations of association rules - NO RESULTS REPORTED
# healthRules2 <- arules::apriori(data=healthPartyTransactions2, parameter = list(supp=0.2, conf = 0.8, minlen=3), appearance = list(default="rhs", lhs='high-sentiment'))
# healthRules2 <- arules::sort(healthRules2, decreasing = TRUE, by='confidence')
# arules::inspect(healthRules2)
# healthRules3 <- arules::apriori(data=healthPartyTransactions2, parameter = list(supp=0.2, conf = 0.8, minlen=3), appearance = list(lhs="low-sentiment"))
# healthRules3 <- arules::sort(healthRules2, decreasing = TRUE, by='confidence')
# arules::inspect(healthRules3)
# healthRulesx1h <- arules::apriori(data=healthPartyTransactions2, parameter = list(supp=0.2, conf = 0.8, minlen=3), appearance = list(lhs="high-x1"))
# healthRulesx1h <- arules::sort(healthRulesx1h, decreasing = TRUE, by='confidence')
# arules::inspect(healthRulesx1h)
# healthRulesx1l <- arules::apriori(data=healthPartyTransactions2, parameter = list(supp=0.2, conf = 0.8, minlen=3), appearance = list(lhs="low-x1"))
# healthRulesx1l <- arules::sort(healthRulesx1l, decreasing = TRUE, by='confidence')
# arules::inspect(healthRulesx1l)
# healthRulesx2h <- arules::apriori(data=healthPartyTransactions2, parameter = list(supp=0.2, conf = 0.8, minlen=3), appearance = list(lhs="high-x2"))
# healthRulesx2h <- arules::sort(healthRulesx2h, decreasing = TRUE, by='confidence')
# arules::inspect(healthRulesx2h)
# healthRulesx2l <- arules::apriori(data=healthPartyTransactions2, parameter = list(supp=0.2, conf = 0.8, minlen=3), appearance = list(lhs="low-x2"))
# healthRulesx2l <- arules::sort(healthRulesx2l, decreasing = TRUE, by='confidence')
# arules::inspect(healthRulesx2l)
# healthRulesx3h <- arules::apriori(data=healthPartyTransactions2, parameter = list(supp=0.2, conf = 0.8, minlen=3), appearance = list(lhs="high-x3"))
# healthRulesx3h <- arules::sort(healthRulesx3h, decreasing = TRUE, by='confidence')
# arules::inspect(healthRulesx3h)
# healthRulesx3l <- arules::apriori(data=healthPartyTransactions2, parameter = list(supp=0.2, conf = 0.8, minlen=3), appearance = list(lhs="low-x3"))
# healthRulesx3l <- arules::sort(healthRulesx3l, decreasing = TRUE, by='confidence')
# arules::inspect(healthRulesx3l)
# healthRulesx4h <- arules::apriori(data=healthPartyTransactions2, parameter = list(supp=0.2, conf = 0.8, minlen=3), appearance = list(lhs="high-x4"))
# healthRulesx4h <- arules::sort(healthRulesx4h, decreasing = TRUE, by='confidence')
# arules::inspect(healthRulesx4h)
# healthRulesx4l <- arules::apriori(data=healthPartyTransactions2, parameter = list(supp=0.2, conf = 0.8, minlen=3), appearance = list(lhs="low-x4"))
# healthRulesx4l <- arules::sort(healthRulesx4l, decreasing = TRUE, by='confidence')
# arules::inspect(healthRulesx4l)

library(arulesViz)
dev.off()

(plot(healthRules, method = "graph", engine = 'interactive', shading = "confidence"))
#plot(healthRules2, method = "graph", engine = 'interactive', shading = "confidence")

#Item frequency analysis for health data
#healthPlusTransactions - ARM transactions dataset with readable, discretized health indicator variables, state, and political parties (50 transaction rows and 64 items)     
#healthPartyTransactions - ARM transactions dataset with readable, discretized health indicator variables and political parties (50 transaction rows and 14 items) 

#Generate Figures - Item Frequency Plots > Health Transactions
#itemFrequencyPlot(healthPlusTransactions, topN=20, type="absolute")
#itemFrequencyPlot(healthPartyTransactions, topN=20, type="absolute")

#Figure: Absolute frequency - healthPlus
itemFrequencyPlot(healthPlusTransactions,
                  type="absolute",
                  topN=10, 
                  horiz=TRUE,
                  col='steelblue3',
                  xlab='',
                  main='healthPlusTransactions Top 10 Item Frequency (Absolute)')

#Figure: Absolute frequency - healthParty
itemFrequencyPlot(healthPartyTransactions,
                  type="absolute",
                  topN=10, 
                  horiz=TRUE,
                  col='red',
                  xlab='',
                  main='healthPartyTransactions Top 10 Item Frequency (Absolute)')

#Political party and health associations
#Check for highest frequency values
#itemFrequencyPlot(healthPartyTransactions, topN=20, type="absolute", col="Dark Green", main="Top 20 Health Indicator Values", ylab="Frequency")

###########################################
#Target Democratic Party to generate rules
dem_rules <- arules::apriori(data=healthPartyTransactions, parameter=list(supp=0.02, conf=0.8, minlen=2),
                             appearance = list (default="lhs", rhs="politicalParty=Democratic"),
                             control = list(verbose=F))
dem_rules <- sort(dem_rules, decreasing=TRUE, by="confidence")

#Get summary info about all rules
(summary(dem_rules))
#Inspect 7 rules in depth
inspect(dem_rules[1:7])
#Visualize the ARM Results
dem_rules2 <- head(sort(dem_rules, by="lift"), 7)

#Plot rules (static graph) - NOT USED
#plot(dem_rules2, method="graph", main="Associations between Health Indicator Values and the Democratic Political Party")

#Plot rules (interactive graph)
plot(dem_rules2, method="graph", interactive=TRUE)

###########################################
#Target Republican Party to generate rules
rep_rules <- arules::apriori(data=healthPartyTransactions, parameter=list(supp=0.02, conf=0.8, minlen=2),
                             appearance = list (default="lhs", rhs="politicalParty=Republican"),
                             control = list(verbose=F))
rep_rules <- sort(rep_rules, decreasing=TRUE, by="confidence")

#Get summary info about all rules
(summary(rep_rules))
#Inspect 35 rules in depth
inspect(rep_rules[1:35])
#Visualize the ARM Results
rep_rules2 <- head(sort(rep_rules, by="lift"), 10)

#Plot rules (interactive graph)
plot(rep_rules2, method="graph", interactive=TRUE)

#################################################################################################
## ARM results show associations between democratic party and low poor mental and physical health 
#days. In addition, there is an association between republican party and high poor mental health 
#and physical health days. 
#################################################################################################

#################################################################################################
## Clustering Algorithms (kMeans and HAC)  
#Unsupervised classification alogorithms for identifying relationships and clusters of 
#similarities within dataset.
#################################################################################################
require(HAC)
require(cluster)
require(tm)
library(VIM)  
require(ggfortify)
library(factoextra)

#kmeans Modeling
disc2_numerical <- state_emotions_health[,c(1,4,5,16,17,18,19)]
aggr(disc2_numerical)
disc2_test <- disc2_numerical
disc2_test$state <- as.numeric(disc2_test$state)
disc2_test$politicalParty <- as.numeric(disc2_test$politicalParty)  
model_h <- kmeans(disc2_test, centers =  2)  #2 is the number of clusters,

cluster_assignment <- data.frame(disc2_test, model_h$cluster)
(View(cluster_assignment))  #view the cluster assignment.

#Clusters for analysis
plot(disc2_test$state ~ jitter(model_h$cluster, 1), pch=21)  #visualize the state attribute
plot(disc2_test$politicalParty ~ jitter(model_h$cluster, 1), pch=21)  #visualize the political party attribute
plot(disc2_test$posSentiment ~ jitter(model_h$cluster, 1), pch=21)  #visualize the sentiment
plot(disc2_test$x1AvgDays ~ jitter(model_h$cluster, 1), pch=21)  #visualize the x1
plot(disc2_test$x2AvgDays ~ jitter(model_h$cluster, 1), pch=21)  #visualize the x2
plot(disc2_test$x3NumPop ~ jitter(model_h$cluster, 1), pch=21)  #visualize the x3
plot(disc2_test$x4NumTot ~ jitter(model_h$cluster, 1), pch=21)  #visualize the x4

#fviz_cluster(model_h, as.matrix(disc2_test)) - this produced Error in prcomp.default(data, scale = FALSE, center = FALSE) : cannot rescale a constant/zero column to unit 

model_h3 <- kmeans(disc2_test, centers =  3)  #3 is the number of clusters,  looking for clusters by High, med, low

cluster_assignment3 <- data.frame(disc2_test, model_h3$cluster)
(View(cluster_assignment3))  #view the cluster assignment.

#Clusters for analysis
plot(disc2_test$posSentiment ~ jitter(model_h3$cluster, 1), pch=21)  #visualize the sentiment
plot(disc2_test$x1AvgDays ~ jitter(model_h3$cluster, 1), pch=21)  #visualize the x1
plot(disc2_test$x2AvgDays ~ jitter(model_h3$cluster, 1), pch=21)  #visualize the x2
plot(disc2_test$x3NumPop ~ jitter(model_h3$cluster, 1), pch=21)  #visualize the x3
plot(disc2_test$x4NumTot ~ jitter(model_h3$cluster, 1), pch=21)  #visualize the x4

#Plot the cluster assignments with the states
autoplot(prcomp(cluster_assignment), data = disc2_test, colour = 'state', label = TRUE, label.size=3) 

#Plot the cluster assignments with the political parties
autoplot(prcomp(cluster_assignment), data = disc2_test, colour = 'politicalParty')  

#HAC Modeling - A
disc2_HAC <- disc2_numerical
distdiscHealth_E <- dist(disc2_HAC, method="euclidean")  #caluculate the L2 distance
distdiscHealth_M <- dist(disc2_HAC, method="manhattan")  #calculate the L1 distance

discHac = dist(disc2_HAC)  #default HAC distance
hcluster=hclust(discHac)  #plot default
testcut <- cutree(hcluster, k=2)  #cut at 2 clusters (Republican and Democrat)
plot(hcluster)  #plot default
rect.hclust(hcluster, k=2, border = 2:6)  #define the box
abline(h=2, col='red')  #display the box on the plot

#Repeat above using the L2 distance
discHac_E <- hclust(distdiscHealth_E, method = "ward.D")  
plot(discHac_E)
rect.hclust(discHac_E, k=2, border = 2:6)
abline(h=2, col='red')

#Try using the centroid method (UPGMC) which is the squared Euclidian distances
discHac_E_C <- hclust(distdiscHealth_E, method = "centroid")  
plot(discHac_E_C)

#Repeat the plot using the L1 distance
discHac_M <- hclust(distdiscHealth_M, method = "ward.D")
plot(discHac_M)
rect.hclust(discHac_M, k=2, border = 2:6)
abline(h=2, col='red')

#Visualization of HAC Manhattan distance with states colored by political party
library(dendextend)
rnames <- disc2_numerical$state
rownames(disc2_HAC)<- rnames
disc2_HAC
distdiscHealth_M <- dist(disc2_HAC, method="manhattan")  #calculate the L1 distance
discHac_M <- as.dendrogram(hclust(distdiscHealth_M, method = "ward.D"))
colors_to_use <- as.numeric(disc2_HAC$politicalParty)  #set color by political party
labels_colors(discHac_M) <- colors_to_use
plot(discHac_M)
#rect.hclust(discHac_M, k=2, border = 2:6)

#################################################################################################
## Decision Trees  
#Classification learning model for predicting of party based on health indicators.
#################################################################################################
library(rattle)
library(rpart)

#Create Decision Trees Using 10 Fold Validation
#head(state_emotions_health)
partyHealth<-state_emotions_health[,c(4,16:19)]
#head(partyHealth)

#Create Decision Trees Modeling function using 'rpart' algorithm
decisionTrees_partyHealth <- function(i) {
  #Set.seed allows for reproduction of sampling results (increments by one for each run/validation)
  seed=i*10
  set.seed(seed)
  #Size is set based on 80% of the data for training, 80% for testing
  sample <- sample.int(n = nrow(partyHealth), size=floor(0.8*nrow(partyHealth)), replace=FALSE)
  train <- partyHealth[sample, ]
  test <- partyHealth[-sample, ]
  #Train / test ratio
  length(sample)/nrow(partyHealth)
  
  #REMOVE LABELED DATA FROM TEST DATA SET AND PUT INTO NEW DATAFRAME
  testLabeled <- test
  test <- test[,-1]
  str(testLabeled)
  str(test)
  
  train_tree <- rpart(politicalParty ~ x1AvgDays, x2AvgDays, data=train, method="class", control=rpart.control(cp=0.08, minsplit=2))
  summary(train_tree)
  
  #Predict the test dataset using the model for training each tree
  predicted = predict(train_tree, test, type="class")
  
  #Visualize decision tree
  vizTitle <- paste("Decision Tree #",i)
  fancyRpartPlot(train_tree, main=vizTitle)
  
  #Show confusion matrix of predictions (correct/incorrect)
  confMat <- paste("Confustion Matrix #",i)
  print(confMat)
  cm <- table(Party=predicted, true=testLabeled$politicalParty)
  print(cm)
  
  n <- sum(as.matrix(cm)) # number of instances
  diag <- diag(as.matrix(cm)) # number of correctly classified instances per class 
  
  accuracy <- signif((sum(diag)/n*100), digits = 3)
  print(paste("Accuracy of Model #",i,": ",accuracy,"%"))
  return(accuracy)
}

dt1<-decisionTrees_partyHealth(1)
dt2<-decisionTrees_partyHealth(2)
dt3<-decisionTrees_partyHealth(3)
dt4<-decisionTrees_partyHealth(4)
dt5<-decisionTrees_partyHealth(5)
dt6<-decisionTrees_partyHealth(6)
dt7<-decisionTrees_partyHealth(7)
dt8<-decisionTrees_partyHealth(8)
dt9<-decisionTrees_partyHealth(9)
dt10<-decisionTrees_partyHealth(10)

#RESULTS - overall not great accuracy ~ 58%
avgDTaccuracy <- sum(dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9, dt10)/10
paste("Average Decision Tree Accuracy: ", avgDTaccuracy, "%")

#################################################################################################
## naive Bayes  
#classifier algorithm based on conditional probabilities modeling applied to evaluate potential 
#relationship between political party and health indicators.
#################################################################################################
library(e1071)
# Check for independence of variables with correlation visuals - required for naive bayes:
(head(state_emotions_health))
# Remove unwanted columns (only need party and numeric health indicators)
partyHealth<-state_emotions_health[,c(4,16:19)]
partyHealth$politicalParty <- as.numeric(partyHealth$politicalParty)
(str(partyHealth))
(cn1 <- cor(partyHealth))

library(corrgram)
(cv1<-corrgram(partyHealth, order=TRUE, lower.panel=panel.shade,
               upper.panel=panel.pie, text.panel=panel.txt,
               main="Health Indicator Correlations"))

#x2Avg Days and x1AvgDays are highly correlated (seen in cn1/cv1, 0.99759179)

#Try correlation without physical health indicator(x1)
partyHealth_nox1 <- partyHealth[,c(1,3:5)]
(cn2a <- cor(partyHealth_nox1))
(cv2a<-corrgram(partyHealth_nox1, order=TRUE, lower.panel=panel.shade,
                upper.panel=panel.pie, text.panel=panel.txt,
                main="Health Indicator Correlations"))

#Try correlation without mental health indicator(x2)
partyHealth_nox2 <- partyHealth[,c(1:2,4:5)]
(cn2b <- cor(partyHealth_nox2))
(cv2b<-corrgram(partyHealth_nox2, order=TRUE, lower.panel=panel.shade,
                upper.panel=panel.pie, text.panel=panel.txt,
                main="Health Indicator Correlations"))

#Try correlation without both physical and mental health indicators (x1 and x2)
partyHealth_nox1or2 <- partyHealth[,c(1,4:5)]
(cn3 <- cor(partyHealth_nox1or2))
(cv3<-corrgram(partyHealth_nox1or2, order=TRUE, lower.panel=panel.shade,
               upper.panel=panel.pie, text.panel=panel.txt,
               main="Health Indicator Correlations"))

#Project notation: This section commented out RETURNED ERROR due to no predictions/data set being too small
#Additional examination with NB completed at later stage
#Run naive bayes with political party and indicators x2, x3, x4
#naiveBayes_forPartyHealthx2_3_4  <- function(i) {  
#  # Set.seed allows for reproduction of sampling results (increments by one for each run/validation)
#  seed=i*10
#  set.seed(seed)
#  # Size is set based on 80% of the data for training, 20% for testing
#  sample <- sample.int(n = nrow(partyHealth_nox1), size=floor(0.8*nrow(partyHealth_nox1)), replace=FALSE)
#  train <- partyHealth_nox1[sample, ]
#  test <- partyHealth_nox1[-sample, ]
#  # train / test ratio
#  length(sample)/nrow(partyHealth_nox1)
#  str(train)
#  str(test)
#  # REMOVE LABELED DATA FROM TEST DATA SET AND PUT INTO NEW DATAFRAME
#  testLabeled <- test
#  testUnlabeled <- test[,-1]
#  #str(testLabeled)
#  #str(testUnlabeled)
#  # Run training model and prediction
#  NB_Training_Model <- naiveBayes(politicalParty ~., data=train, laplace=1)
#  NB_Prediction <- predict(NB_Training_Model, testUnlabeled)
#  # Show confusion matrix of predictions (correct/incorrect)
#  print(paste("Confusion Matrix #",i, sep=" "))
#  cm <- table(NB_Prediction, testLabeled$politicalParty, dnn=c("Prediction","Actual"))
#  print(cm)
# numberedTitle <- paste("Predicted Number Frequency #", i, sep=" ")
#  plot(NB_Prediction, main=numberedTitle, xlab="Predicted Number", ylab="Frequency")
#  n <- sum(as.matrix(cm)) # number of instances
#  diag <- diag(as.matrix(cm)) # number of correctly classified instances per class 
#  # Calculate and print accuracy
#  accuracy <- signif((sum(diag)/n*100), digits = 3)
#  print(paste("Accuracy of Model #",i,": ",accuracy,"%"))
#  return(accuracy)
#}
#Run naive bayes with political party and indicators x1, x3, x4
#naiveBayes_forPartyHealthx1_3_4 <- function(i) {  
#  # Set.seed allows for reproduction of sampling results (increments by one for each run/validation)
#  i=1
#  seed=i*10
#  set.seed(seed)
#  # Size is set based on 80% of the data for training, 20% for testing
#  sample <- sample.int(n = nrow(partyHealth_nox2), size=floor(0.8*nrow(partyHealth_nox2)), replace=FALSE)
#  train <- partyHealth_nox2[sample, ]
#  test <- partyHealth_nox2[-sample, ]
#  # train / test ratio
#  length(sample)/nrow(partyHealth_nox2)
#  str(train)
#  str(test)
#  # REMOVE LABELED DATA FROM TEST DATA SET AND PUT INTO NEW DATAFRAME
#  testLabeled <- test
#  testUnlabeled <- test[,-1]
#  #str(testLabeled)
#  #str(testUnlabeled)
#  # Run training model and prediction
#  NB_Training_Model <- naiveBayes(politicalParty ~., data=train, laplace=1)
#  NB_Prediction <- predict(NB_Training_Model, testUnlabeled)
#  # Show confusion matrix of predictions (correct/incorrect)
#  print(paste("Confusion Matrix #",i, sep=" "))
#  cm <- table(NB_Prediction, testLabeled$politicalParty, dnn=c("Prediction","Actual"))
#  print(cm)
#  numberedTitle <- paste("Predicted Number Frequency #", i, sep=" ")
#  plot(NB_Prediction, main=numberedTitle, xlab="Predicted Number", ylab="Frequency")
#  n <- sum(as.matrix(cm)) # number of instances
#  diag <- diag(as.matrix(cm)) # number of correctly classified instances per class 
#  # Calculate and print accuracy
#  accuracy <- signif((sum(diag)/n*100), digits = 3)
#  print(paste("Accuracy of Model #",i,": ",accuracy,"%"))
#  return(accuracy)
#}
#Run naive bayes with political party and indicators x3, x4
#naiveBayes_forPartyHealthx3_4  <- function(i) {  
#  # Set.seed allows for reproduction of sampling results (increments by one for each run/validation)
#  seed=i*10
#  set.seed(seed)
#  # Size is set based on 80% of the data for training, 20% for testing
#  sample <- sample.int(n = nrow(partyHealth_nox1), size=floor(0.8*nrow(partyHealth_nox1or2)), replace=FALSE)
#  train <- partyHealth_nox1or2[sample, ]
#  test <- partyHealth_nox1or2[-sample, ]
#  # train / test ratio
#  length(sample)/nrow(partyHealth_nox1or2)
#  str(train)
#  str(test)
#  # REMOVE LABELED DATA FROM TEST DATA SET AND PUT INTO NEW DATAFRAME
#  testLabeled <- test
#  testUnlabeled <- test[,-1]
#  #str(testLabeled)
#  #str(testUnlabeled
#Run training model and prediction
#NB_Training_Model <- naiveBayes(politicalParty ~., data=train, laplace=1)
#NB_Prediction <- predict(NB_Training_Model, testUnlabeled)
#  Show confusion matrix of predictions (correct/incorrect)
#  print(paste("Confusion Matrix #",i, sep=" "))
#  cm <- table(NB_Prediction, testLabeled$politicalParty, dnn=c("Prediction","Actual"))
#  print(cm)
#  numberedTitle <- paste("Predicted Number Frequency #", i, sep=" ")
#  plot(NB_Prediction, main=numberedTitle, xlab="Predicted Number", ylab="Frequency")
#  n <- sum(as.matrix(cm)) # number of instances
#  diag <- diag(as.matrix(cm)) # number of correctly classified instances per class 
#  # Calculate and print accuracy
#  accuracy <- signif((sum(diag)/n*100), digits = 3)
#  print(paste("Accuracy of Model #",i,": ",accuracy,"%"))
#  return(accuracy)
#}
#Call functions
#n_a1 <- naiveBayes_forPartyHealthx1_3_4(1)
#n_a2 <- naiveBayes_forPartyHealthx1_3_4(2)
#n_a3 <- naiveBayes_forPartyHealthx1_3_4(3)
#n_a4 <- naiveBayes_forPartyHealthx1_3_4(4)
#n_a5 <- naiveBayes_forPartyHealthx1_3_4(5)
#n_b1 <- naiveBayes_forPartyHealthx2_3_4(1)
#n_b2 <- naiveBayes_forPartyHealthx2_3_4(2)
#n_b3 <- naiveBayes_forPartyHealthx2_3_4(3)
#n_b4 <- naiveBayes_forPartyHealthx2_3_4(4)
#n_b5 <- naiveBayes_forPartyHealthx2_3_4(5)
#n_c1 <- naiveBayes_forPartyHealthx3_4(1)
#n_c2 <- naiveBayes_forPartyHealthx3_4(2)
#n_c3 <- naiveBayes_forPartyHealthx3_4(3)
#n_c4 <- naiveBayes_forPartyHealthx3_4(4)
#n_c5 <- naiveBayes_forPartyHealthx3_4(5)
# Five fold validation:
#(overallAccuracyNB_x1_3_4 <- sum(n_a1, n_a2, n_a3, n_a4, n_a5)/5)
#(overallAccuracyNB_x2_3_4 <- sum(n_b1, n_b2, n_b3, n_b4, n_b5)/5)
#(overallAccuracyNB_x3_4 <- sum(n_c1, n_c2, n_c3, n_c4, n_c5)/5)

#################################################################################################
## Primary classification and prediction modeling (kNN, SVM, and Random Forest)

## kNN is an instance-based learning algorithm for classifying observations by distance 
#similarity (= "nearest neighbors"). 

## Support Vector Machine (SVM)** is applied for class-level predictions with modeling that looks 
#for the highest prediction accuracy from the lowest number of support vectors as a best fitting model.

## Random Forest** is an ensemble of Decision Trees (DT) that generates multiple models on a 
#training dataset and combines (averages) output rules for classification purposes. 
#################################################################################################

#Creating training and testing datasets - discretized ('discTrain') and numeric ('discTrainN') for learning models

set.seed(8)
mRandIndex <- sample(1:dim(disc2_Transform)[1])
mCutPoint3_4 <-  floor(3*nrow(disc2_Transform)/4)
discTrain <- disc2_Transform[mRandIndex[1:mCutPoint3_4],]
discTest <- disc2_Transform[mRandIndex[(mCutPoint3_4+1):nrow(disc2_Transform)],]

#Verify cut points
(nrow(discTrain))
(nrow(discTest))

mRandIndexN <- sample(1:dim(disc2_numerical)[1])
mCutPoint3_4N <-  floor(3*nrow(disc2_numerical)/4)
discTrainN <- disc2_numerical[mRandIndexN[1:mCutPoint3_4N],]
discTestN <- disc2_numerical[mRandIndexN[(mCutPoint3_4N+1):nrow(disc2_numerical)],]

discTrainNsent <- discTrainN
discTestNsent <- discTestN
discTrainNsent$posSentiment <-discretizeLMHname(discTrainNsent$posSentiment, "-sentiment")
discTestNsent$posSentiment <-discretizeLMHname(discTestNsent$posSentiment, "-sentiment") 

#Verify cutpoints
(nrow(discTrainN))
(nrow(discTestN))

###########################################
## knn Modeling for classification and prediction 
#of sentiment and political party.

library(class)
library(ggraph)
library(randomForest) 
library(GGally)
library(caret)

k <- round(sqrt(nrow(discTrainN)))
dblk <- k*2
halfk <- k/2

#Train based on SENTIMENT
discTrainLabelsN = discTrainNsent[,3]
discTrainNoLabelN = discTrainNsent[,-3]
discTestLabelsN = discTestNsent[,3]
discTestNoLabelsN = discTestNsent[,-3]

discTrainNoLabelN$state <- as.numeric(discTrainNoLabelN$state)
discTrainNoLabelN$politicalParty <- as.numeric(discTrainNoLabelN$politicalParty)

discTestNoLabelsN$state <- as.numeric(discTestN$state)
discTestNoLabelsN$politicalParty <- as.numeric(discTestN$politicalParty)

kNNdiscTrain <- class::knn(train=discTrainNoLabelN, test=discTestNoLabelsN, cl=discTrainLabelsN ,k = halfk, prob=TRUE)  
#Note: Small dataset for knn small
(print(kNNdiscTrain))
(table(kNNdiscTrain, discTestLabelsN))

kNNNumTrainDbl <- class::knn(train=discTrainNoLabelN, test=discTestNoLabelsN, cl=discTrainLabelsN ,k = dblk, prob=TRUE)
print(kNNNumTrainDbl)
(table(kNNNumTrainDbl, discTestLabelsN))

kNNNumTrainHalf <- class::knn(train=discTrainNoLabelN, test=discTestNoLabelsN, cl=discTrainLabelsN ,k = halfk, prob=TRUE)
print(kNNNumTrainHalf)
(table(kNNNumTrainHalf, discTestLabelsN))

testmat <- as.matrix(table(kNNNumTrainHalf, discTestLabelsN))
cmHalf <- confusionMatrix(kNNNumTrainHalf, discTestLabelsN)
cmdbl <-  confusionMatrix(kNNNumTrainDbl, discTestLabelsN)
cm <- confusionMatrix(kNNdiscTrain, discTestLabelsN)

(cmHalf$overall)
(cmdbl$overall)
(cm$overall)

#Train based on POLITICAL PARTY
discTrainLabelsparty = discTrainN[,2]
discTrainNoLabelparty = discTrainN[,-2]
discTrainNoLabelparty$state <- as.numeric(discTrainNoLabelparty$state)

discTestparty <- discTestN
discTestparty$state <- as.numeric(discTestN$state)
discTestpartyLabel <- discTestparty[,2]
discTestpartyNoLabel <- discTestparty[,-2]

kNNdiscTrainp <- class::knn(train=discTrainNoLabelparty, test=discTestpartyNoLabel, cl=discTrainLabelsparty ,k = halfk, prob=TRUE)  
#Note: Small data set for kNN
(print(kNNdiscTrainp))
(table(kNNdiscTrainp, discTestpartyLabel))

kNNNumTrainDblp <- class::knn(train=discTrainNoLabelparty, test=discTestpartyNoLabel, cl=discTrainLabelsparty ,k = dblk, prob=TRUE)
(print(kNNNumTrainDblp))
(table(kNNNumTrainDblp, discTestpartyLabel))

kNNNumTrainHalfp <- class::knn(train=discTrainNoLabelparty, test=discTestpartyNoLabel, cl=discTrainLabelsparty ,k = halfk, prob=TRUE)
(print(kNNNumTrainHalfp))
(table(kNNNumTrainHalfp, discTestpartyLabel))

testmatp <- as.matrix(table(kNNNumTrainHalfp, discTestpartyLabel))
cmHalfp <- confusionMatrix(kNNNumTrainHalfp, discTestpartyLabel)
cmdblp <-  confusionMatrix(kNNNumTrainDblp, discTestpartyLabel)
cmp <- confusionMatrix(kNNdiscTrainp, discTestpartyLabel)

(cmHalfp$overall)
(cmdblp$overall)
(cmp$overall)

cmHalfresultsMatrix <- as.matrix(table(kNNNumTrainHalf, discTestLabelsN))

###########################################
## kNN modeling visualizations

library(plyr)
library(ggplot2)

plot.df = data.frame(discTestpartyNoLabel, predicted = kNNdiscTrainp)

plot.df1 = data.frame(x = plot.df$posSentiment, 
                      y = plot.df$x1AvgDays, 
                      predicted = plot.df$predicted)

find_hull = function(df) df[chull(df$x, df$y), ]
boundary = ddply(plot.df1, .variables = "predicted", .fun = find_hull)

#Sentiment to x1 health indicator
ggplot(plot.df, aes(posSentiment, x1AvgDays, color = predicted, fill = predicted)) + 
  geom_point(size = 5) + 
  geom_polygon(data = boundary, aes(x,y), alpha = 0.5)

plot.df = data.frame(discTestpartyNoLabel, predicted = kNNdiscTrainp)

plot.df1 = data.frame(x = plot.df$posSentiment, 
                      y = plot.df$x2AvgDays, 
                      predicted = plot.df$predicted)

find_hull = function(df) df[chull(df$x, df$y), ]
boundary = ddply(plot.df1, .variables = "predicted", .fun = find_hull)

#Sentiment to x2 health indicator
ggplot(plot.df, aes(posSentiment, x2AvgDays, color = predicted, fill = predicted)) + 
  geom_point(size = 5) + 
  geom_polygon(data = boundary, aes(x,y), alpha = 0.5)

plot.df = data.frame(discTestpartyNoLabel, predicted = kNNdiscTrainp)

plot.df1 = data.frame(x = plot.df$posSentiment, 
                      y = plot.df$x4NumTot, 
                      predicted = plot.df$predicted)

find_hull = function(df) df[chull(df$x, df$y), ]
boundary = ddply(plot.df1, .variables = "predicted", .fun = find_hull)

#Sentiment to x4 health indicator
ggplot(plot.df, aes(posSentiment, x4NumTot, color = predicted, fill = predicted)) + 
  geom_point(size = 5) + 
  geom_polygon(data = boundary, aes(x,y), alpha = 0.5)

###########################################
## Random Forest modeling

discTestNRFLabels <- discTestNsent[,3]
discTestNRF <- discTestNsent[,-3]

numdiscRF <- randomForest(posSentiment ~ . , data = discTrainNsent)  #random forest small data set
(print(numdiscRF))
prednum_RF<-predict(numdiscRF, discTestNRF) 
(table(prednum_RF, discTestNRFLabels))
varImpPlot(numdiscRF)

#Number of nodes in the trees in the RF. 
hist(treesize(numdiscRF))

RFcm <- confusionMatrix(prednum_RF, discTestNRFLabels)
(RFcm$overall)

discTestNRFpartyLabels <- discTestN[,2]
discTestNRFparty <- discTestN[,-2]

numdiscRFparty <- randomForest(politicalParty ~ . , data = discTrainN)  #random forest small data set
(print(numdiscRFparty))
prednum_RFparty<-predict(numdiscRFparty, discTestNRFparty) 
(table(prednum_RFparty, discTestNRFpartyLabels))
varImpPlot(numdiscRFparty)

#Number of nodes in the trees in the RF. 
hist(treesize(numdiscRFparty))

RFcmparty <- confusionMatrix(prednum_RFparty, discTestNRFpartyLabels)
(RFcmparty$overall)

#################################################################################################
## Secondary classification and prediction modeling (kNN, SVM, and Random Forest)

## A secondary round of modeling was conducted to address or support specific data questions that 
#emerged during the project time period. The section below provides each inquiry and then applies 
#a learning model approach for examination. 

## Data Question: Do spoken emotions and sentiments have a relationship to specific health indicators? 
#################################################################################################
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(Cairo)
library(caret)

emotions_health <- state_emotions_health
emotions_health[,1:5] <- NULL
emotions_health$x1AvgDays <- NULL
emotions_health$x2AvgDays <- NULL
emotions_health$x3NumPop <- NULL
emotions_health$x4NumTot <- NULL
str(emotions_health)

trainHealth1 <- emotions_health[,1:11]
trainHealth1_neg <- trainHealth1[,c(1,3,4,6,9,11)]
trainHealth1_pos <- trainHealth1[,c(2,5,7,8,10,11)]
str(trainHealth1_neg)
str(trainHealth1_pos)

#Training and testing datasets for x1
(sample_size <- floor(0.60 * nrow(trainHealth1))) #30 / 20

#Create two random/sample partitioning variables for subsets (set seed to reproduce)
train_select4 <- sample(seq_len(nrow(trainHealth1_neg)), size = sample_size)
train_select5 <- sample(seq_len(nrow(trainHealth1_pos)), size = sample_size)
train_select6 <- sample(seq_len(nrow(trainHealth1)), size = sample_size)

#Generate sets of training and testing data; store and remove test labels
trainHealth1all <- trainHealth1[train_select6, ] 
testHealth1all <- trainHealth1[-train_select6, ] 
testHealth1all_label <- testHealth1all$x1AvgDays_Disc
testHealth1all$x1AvgDays_Disc <- NULL

#Check that the random sampling for balance of factors
(View(trainHealth1all$x1AvgDays_Disc))

#RF modeling x1 - ALL
RF_em_h1 <- randomForest(trainHealth1all$x1AvgDays_Disc ~., data=trainHealth1all)

#Print RF training output
(print(RF_em_h1))
(plot(RF_em_h1))

#Classification predictions for health indicators based on emotions
pred_RF4 <- predict(RF_em_h1, testHealth1all)

(resultsTab11 <- table(pred_RF4, testHealth1all_label))
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(resultsTab11) #60%

#NB 1
health_NBclassfier2 <- naiveBayes(trainHealth1all$x1AvgDays_Disc ~.,data=trainHealth1all, type = "class")
NBclassfier2_Predict <- predict(health_NBclassfier2, testHealth1all)

health_NBclassfier2 #RESULTS - explain conditional probabilities

(confusionMatrix(NBclassfier2_Predict, testHealth1all_label)) #65%

plot(NBclassfier2_Predict)

(resultsTab12 <- table(NBclassfier2_Predict, testHealth1all_label))

#Create visualization to show Frequency of Predictions to Actuals for each numeric possibility (0-9) 
results_DF <- as.data.frame(resultsTab12)
colnames(results_DF) <- c("Prediction", "Actual", "Frequency")
results_DF$Actual <- as.factor(results_DF$Actual)
results_DF$Frequency <- as.factor(results_DF$Frequency)

#View to determine row ranges for each possible image
(View(results_DF))

r0 <- results_DF

r0Bar <- ggplot(r0, aes(y=r0$Frequency, x=r0$Prediction, fill = r0$Frequency)) + 
  geom_bar(position="dodge", stat="identity") +
  ggtitle("Frequency of Predictions for Poor Physical Health (x1) based on Emotions") +
  xlab("Predicted Image Number") +
  scale_y_discrete(name="Frequency of Prediction") +
  labs(fill = "Freq. Predict") 
r0Bar

#################################################################################################
## Data Question: Can specific health indicators be attributed to political parties in charge? 
#(If we can predict classification as R or D, then health measures may correlate (good or bad) 
#to political parties in charge.)  

## This secondary exploration involves looking at all county data (3,140 obersvations in U.S.) 
#to increase training data beyond the aggregated state-level analysis. 
#################################################################################################

#View political parties by state
(View(emotionsStates_DF[,1:2]))

SVM_healthIndicators <- healthIndicators_DF
#Change stateReporting to character
SVM_healthIndicators$stateReporting <- as.character(SVM_healthIndicators$stateReporting)
(str(SVM_healthIndicators))

#Change statesReporting to party for all county obsevations #3,140
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Alabama"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Alaska"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Arizona"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Arkansas"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "California"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Colorado"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Connecticut"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Delaware"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Florida"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Georgia"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Hawaii"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Idaho"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Illinois"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Indiana"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Iowa"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Kansas"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Kentucky"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Louisiana"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Maine"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Maryland"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Massachusetts"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Michigan"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Minnesota"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Mississippi"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Missouri"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Montana"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Nebraska"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Nevada"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "New Hampshire"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "New Jersey"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "New Mexico"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "New York"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "North Carolina"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "North Dakota"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Ohio"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Oklahoma"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Oregon"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Pennsylvania"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Rhode Island"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "South Carolina"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "South Dakota"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Tennessee"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Texas"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Utah"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Vermont"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Virginia"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Washington"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "West Virginia"] <- "R"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Wisconsin"] <- "D"
SVM_healthIndicators$stateReporting[SVM_healthIndicators$stateReporting == "Wyoming"] <- "R"

#Convert stateReporting back to a 2-level class label factor 
SVM_healthIndicators$stateReporting <- as.factor(SVM_healthIndicators$stateReporting)
(str(SVM_healthIndicators))

#Remove columns not specific to inquiry (keep numeric health indicator values for all U.S. counties)
SVM_subset <-SVM_healthIndicators
SVM_subset$state<- NULL
SVM_subset$county <- NULL
SVM_subset$x3Per <- NULL
SVM_subset$x4Tot <- NULL
SVM_subset$x4Per <- NULL
SVM_subset$x5Per <- NULL
(str(SVM_subset))

#Convert int data type to numeric
library(taRifx)
SVM_subset <- japply(SVM_subset, which(sapply(SVM_subset, class)=="integer"), as.numeric)
str(SVM_subset)

#Min-Max-Function to rescale and normalize column data - data values scaled into the range of [0, 1]
min_max_norm <- function(x) { 
  z=x
  if(min(x)<max(x)){ 
    z=(x - min(x)) / (max(x) - min(x))
  }
  return(z)
} 

#Test with small subset
SVM_subset_test <- SVM_subset[1:50, ]
str(SVM_subset_test)

SVM_norm <- as.data.frame(lapply(SVM_subset_test[,1:4], min_max_norm))
str(SVM_norm)

#Create variable for class label and apply min_max_norm to dataset
SVM_subset_label <- SVM_subset$stateReporting
SVM_norm <- as.data.frame(lapply(SVM_subset[,1:4], min_max_norm))
str(SVM_norm)
#Add label
SVM_subset_norm <- cbind(SVM_subset_label, SVM_norm)
colnames(SVM_subset_norm) <- c("party", "physh", "mh", "uninsure", "unemploy")

#DATASET: 'SVM_subset_norm' 3140 observations (counties) of 4 health indicators + factor political party (D, R)

#Create training and testing data for modeling

#Random sampling to create training dataset at 75% of observations
(sample_size <- floor(0.75 * nrow(SVM_subset_norm))) #2355

#Create random/sample partitioning variables for training and testing datasets
set.seed(123)
train_select1 <- sample(seq_len(nrow(SVM_subset_norm)), size = sample_size)

#Generate training and testing data; store and remove test class label
train_DF_norm <- SVM_subset_norm[train_select1, ] #2355 obs 5 variables (incl. class)
test_DF_norm <- SVM_subset_norm[-train_select1, ] #785 obs 5 variables (incl. class)
test_DF_class <- test_DF_norm$party
test_DF_norm$party <- NULL

#Random Forest modeling
library(randomForest)
#train_DF_norm
#test_DF_norm

#RF modeling
RF_health1 <- randomForest(party ~., data=train_DF_norm)

#Print RF training output
print(RF_health1)

#ALT FIGURES: Histograms of RF trees -number of nodes in the trees in RF
hist(treesize(RF_health1))

#Classification predictions for Political Parties based on county health indicators
pred_RF1 <- predict(RF_health1, test_DF_norm)

(resultsTab1 <- table(pred_RF1, test_DF_class))
accuracy(resultsTab1) #72.87%

#FIGURE: Confusion Matrix for RF1 (w/Accuracy 72.87%)
(View(resultsTab1))

r1_Results <- data.frame(resultsTab1)
True_Class <- factor(c("Republican", "Republican", "Democratic", "Democratic"))
Predicted_Class <- factor(c("Republican", "Democratic", "Republican", "Democratic"))
Y1  <- c(389, 115, 98, 183) #Read in bottom to top from r#_Results
df1 <- data.frame(True_Class, Predicted_Class, Y1)

r1 <- ggplot(data =  df1, mapping = aes(x = True_Class, y = Predicted_Class)) +
  geom_tile(aes(fill = Y1), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y1)), vjust = 1, size=10) +
  scale_fill_gradient(low = "grey50", high = "darkorchid1") +
  theme_bw() + theme(legend.position = "none")
r1

#################################################################################################
## Data Question: Can certain health indicators predict the level of positive sentiment in 
#political speeches? 
#################################################################################################

#Datasets for RF and SVM
RF_state_emotions_health <- state_emotions_health

#Discretization for posSentiment
#Create function to discretize a variable into three categories (low, medium, high)
#low = bottom third of variable values
#medium = middle third of variable values
#high = top third of variable values

discretizeLMH <- function(var) {
  # Variable min value
  varMin<-min(var)
  # Variable max value
  varMax<-max(var)
  # Num bins to divide variable into
  bins <- 3
  # Width of each bin
  width <- ((varMax-varMin)/bins)
  # Low if value is bottom third of variable values
  # Medium if between bottom and top third of variable values
  # High if value is top third of variable values
  return(as.factor(ifelse(var < (varMin+width), "low", 
                          (ifelse(var >= (varMin+(2*width)),"high",
                                  "medium")))))
}  

#Call discretizeLMH function to discretize posSentiment variable
RF_state_emotions_health$posSentiment<-discretizeLMH(RF_state_emotions_health$posSentiment)
str(RF_state_emotions_health)
summary(RF_state_emotions_health$posSentiment)
#Reference positive sentiment for state assignment
(View(RF_state_emotions_health[,3:5]))

RF_healthIndicators <- SVM_healthIndicators
RF_healthIndicators_state <- RF_healthIndicators$state

#Change state to posSentiment factor for all counties
RF_healthIndicators$state[RF_healthIndicators$state == "alabama"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "alaska"] <- "low"
RF_healthIndicators$state[RF_healthIndicators$state == "arizona"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "arkansas"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "california"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "colorado"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "connecticut"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "delaware"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "florida"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "georgia"] <- "low"
RF_healthIndicators$state[RF_healthIndicators$state == "hawaii"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "idaho"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "illinois"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "indiana"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "iowa"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "kansas"] <- "low"
RF_healthIndicators$state[RF_healthIndicators$state == "kentucky"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "louisiana"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "maine"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "maryland"] <- "low"
RF_healthIndicators$state[RF_healthIndicators$state == "massachusetts"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "michigan"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "minnesota"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "mississippi"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "missouri"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "montana"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "nebraska"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "nevada"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "new hampshire"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "new jersey"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "new mexico"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "new york"] <- "low"
RF_healthIndicators$state[RF_healthIndicators$state == "north carolina"] <- "low"
RF_healthIndicators$state[RF_healthIndicators$state == "north dakota"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "ohio"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "oklahoma"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "oregon"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "pennsylvania"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "rhode island"] <- "low"
RF_healthIndicators$state[RF_healthIndicators$state == "south carolina"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "south dakota"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "tennessee"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "texas"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "utah"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "vermont"] <- "high"
RF_healthIndicators$state[RF_healthIndicators$state == "virginia"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "washington"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "west virginia"] <- "low"
RF_healthIndicators$state[RF_healthIndicators$state == "wisconsin"] <- "medium"
RF_healthIndicators$state[RF_healthIndicators$state == "wyoming"] <- "high"

str(RF_healthIndicators)
#Convert stateReporting back to a 2-level class label factor 
RF_healthIndicators$state <- as.factor(RF_healthIndicators$state)
str(RF_healthIndicators)

#Remove columns not specific to inquiry (keep numeric health indicator values for all U.S. counties)
RF_subset <-RF_healthIndicators
RF_subset_party <- RF_subset$stateReporting
RF_subset$county <- NULL
RF_subset$x3Per <- NULL
RF_subset$x4Tot <- NULL
RF_subset$x4Per <- NULL
RF_subset$x5Per <- NULL
RF_subset$stateReporting <- NULL
str(RF_subset)

#Convert int data type to numeric
library(taRifx)
RF_subset <- japply(RF_subset, which(sapply(RF_subset, class)=="integer"), as.numeric)
str(RF_subset)

#Min-Max-Function to rescale and normalize column data - data values scaled into the range of [0, 1]
min_max_norm <- function(x) { 
  z=x
  if(min(x)<max(x)){ 
    z=(x - min(x)) / (max(x) - min(x))
  }
  return(z)
} 

#Test with small subset
RF_subset_test <- RF_subset[1:50, ]
str(RF_subset_test)

RF_test_norm <- as.data.frame(lapply(RF_subset_test[,2:5], min_max_norm))
str(SVM_norm)

#Create variable for class label and apply min_max_norm to dataset
RF_subset_label <- RF_subset$state
RF_norm <- as.data.frame(lapply(RF_subset[,2:5], min_max_norm))
str(RF_norm)
#Add label
RF_subset_norm <- cbind(RF_subset_label, RF_norm)
colnames(RF_subset_norm) <- c("posSent", "physh", "mh", "uninsure", "unemploy")

#DATASET: 'RF_subset_norm' 3140 observations (counties) of 4 health indicators + factor Positive Sentiment (high, low, medium)

#Create training and testing data for RF - SVM

#Random sampling to create training dataset at 75% of observations
(sample_size <- floor(0.75 * nrow(RF_subset_norm))) #2355

#Create random/sample partitioning variables for training and testing datasets
set.seed(123)
train_select2 <- sample(seq_len(nrow(RF_subset_norm)), size = sample_size)

#Generate training and testing data; store and remove test class label
train_RF_norm <- RF_subset_norm[train_select2, ] #2355 obs 5 variables (incl. class)
test_RF_norm <- RF_subset_norm[-train_select2, ] #785 obs 5 variables (incl. class)
test_RF_class <- test_RF_norm$posSent
test_RF_norm$posSent <- NULL

#Random Forest modeling
#train_RF_norm
#test_RF_norm

#RF modeling
RF_sentiment1 <- randomForest(posSent ~., data=train_RF_norm)

#Print RF training output
print(RF_sentiment1)

#Classification predictions for Political Parties based on county health indicators
pred_RF2 <- predict(RF_sentiment1, test_RF_norm)

(resultsTab2 <- table(pred_RF2, test_RF_class))
accuracy(resultsTab2) #59.23%
(View(resultsTab2))
varImpPlot(RF_sentiment1)

#SVM modeling
library(e1071) #algorithms
library(caret)

#SVM kernel = "linear" (c=0.1 46.75%, 5 53.12%)
sentiment_SVMclassifier1 <- svm(train_RF_norm$posSent ~., data=train_RF_norm, 
                                kernel="linear", cost=10, 
                                scale=FALSE)

print(sentiment_SVMclassifier1)
#Number of support vectors = 2236

#Predict Type1
pred_SVM1 <- predict(sentiment_SVMclassifier1, test_RF_norm, type="class")

#Create results table and confirm accuracy with function 
(resultsTab4 <- table(pred_SVM1, test_RF_class))
accuracy(resultsTab4) #53.73%

#SVM kernel = "radial"
sentiment_SVMclassifier2 <- svm(train_RF_norm$posSent ~., data=train_RF_norm, 
                                kernel="radial", cost=.1, 
                                scale=FALSE)

print(sentiment_SVMclassifier2)
#Number of support vectors = 2187

#Predict Type1
pred_SVM2 <- predict(sentiment_SVMclassifier2, test_RF_norm, type="class")

#Create results table and confirm accuracy with function 
(resultsTab5 <- table(pred_SVM2, test_RF_class))
accuracy(resultsTab5) #46.75% 

#################################################################################################
## Data Question: Can health indicators predict political party? 
#(Extension of initial secondary data question.) 
#################################################################################################

#Random Forest modeling
RF_subset_norm <- cbind(RF_subset_party, RF_subset_norm)
RF_subset_norm_health <- RF_subset_norm
RF_subset_norm_health$posSent <- NULL
str(RF_subset_norm_health)

#Create training and testing data for RF 
#Random sampling to create training dataset at 75% of observations
(sample_size <- floor(0.75 * nrow(RF_subset_norm_health))) #2355

#Create random/sample partitioning variables for training and testing datasets
set.seed(123)
train_select3 <- sample(seq_len(nrow(RF_subset_norm_health)), size = sample_size)

#Generate training and testing data; store and remove test class label
train_RF_health <- RF_subset_norm_health[train_select3, ] #2355 obs 5 variables (incl. class)
test_RF_health <- RF_subset_norm_health[-train_select3, ] #785 obs 5 variables (incl. class)
test_RF_health_class <- test_RF_health$RF_subset_party
test_RF_health$RF_subset_party <- NULL

#RF modeling x1
RF_h1 <- randomForest(RF_subset_party ~., data=train_RF_health)

#Print RF training output
(print(RF_h1))
(plot(RF_h1))

#Classification predictions for Political Parties based on county health indicators
pred_RF3 <- predict(RF_h1, test_RF_health)

(resultsTab3 <- table(pred_RF3, test_RF_health_class))

accuracy(resultsTab3) #72.86%
(View(resultsTab3))

#SVM modeling
#train_RF_health 
#test_RF_health
#test_RF_health_class

#SVM kernel = "linear" (c=0.1, 64.20%, 10 65.47% )
party_SVMclassifier1 <- svm(train_RF_health$RF_subset_party ~., data=train_RF_health, 
                            kernel="linear", cost=10, 
                            scale=FALSE)

print(party_SVMclassifier1)
#Number of support vectors = 1870

#Predict Party1
pred_SVM3 <- predict(party_SVMclassifier1, test_RF_health, type="class")

#Create results table and confirm accuracy with function 
(resultsTab8 <- table(pred_SVM3, test_RF_health_class))
accuracy(resultsTab8) #65.47%

#################################################################################################
## Data Question: Can certain emotions accurately predict political party? 
#################################################################################################

library(cluster)  
library(factoextra)
library(e1071)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(rpart)
library(caTools)

s_df <- state_emotions_health
str(s_df)
s_df

#Clustering data to show clusters of emotion
emotions <- s_df[,c(5:15)]
length(s_df)
colnames(s_df)
parties <- s_df[,4]

#k-MEANS
#Determine optimal number of clusters
fviz_nbclust(emotions, kmeans, method = "wss")

#k-Means Cluster Analysis
k_fit <- kmeans(emotions, centers = 2,  nstart = 20) # 2 cluster solution
rownames(emotions) <-  make.names(parties, unique=TRUE)

#k-means viz
fviz_cluster(k_fit, data = emotions)

#k-means (pairwise scatter plot)
emotions %>%
  as_tibble() %>%
  mutate(cluster = k_fit$cluster,
         state = row.names(emotions)) %>%
  ggplot(aes(negative, trust, color = factor(cluster), label = state)) +
  geom_text()

#HCLUST - Hierarchical Clustering
rownames(emotions) = make.names(parties, unique=TRUE)
d <- dist(emotions, method = "euclidean") # distance matrix
fit <- hclust(d, method="ward")
plot(fit) # display dendogram
groups <- cutree(fit, k=2) # cut tree into 2 clusters

#Draw dendogram with red borders around the 2 clusters
rect.hclust(fit, k=2, border="red")

set.seed(109)
emo_df <- cbind(emotions, parties)
split = sample.split(emo_df$parties, SplitRatio = 0.65) 

training_set = subset(emo_df, split == TRUE) 
test_set = subset(emo_df, split == FALSE)

#RPART - tree construction based on information gain
tree <- rpart(parties ~ ., data = training_set, method = 'class', cp = 0.051)

t_pred <- predict(tree,test_set[-ncol(test_set)],type="class")
con_tree <- table(test_set[,ncol(test_set)],t_pred)
accuracy_tree <- sum(diag(con_tree))/sum(con_tree)
printcp(tree)

#Plot mytree
fancyRpartPlot(tree, caption = NULL)

#naive Bayes modeling
nb_model <- naiveBayes(parties ~ ., data = training_set) #without labels
nb_model

#Prediction on the dataset
NB_Predictions=predict(nb_model,test_set[-ncol(test_set)])

#Confusion matrix to check accuracy
con_nb <- table(test_set[,ncol(test_set)], NB_Predictions)
accuracy_nb <- sum(diag(con_nb))/sum(con_nb)

#SVM modeling

#Fit model and produce plot
classifier = svm(formula = parties ~ ., 
                 data = training_set, 
                 type = 'C-classification', 
                 kernel = 'linear')

#Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-ncol(test_set)]) #without labels

#Making the Confusion Matrix 
con_svm = table(test_set[,ncol(test_set)], y_pred)
accuracy_svm <- sum(diag(con_svm))/sum(con_svm)

#Classification accuracies
accuracy_tree
accuracy_nb
accuracy_svm

#################################################################################################
## Conclusions    

## Summary conclusions based on exploration of selected data and examination of daat questions 
#proposed by the project team:  
  
## 1. There is no evidence that specific emotions and sentiments can accurately predict our 
#general health and welfare.

## 2. There are distinct emotions and sentiments that appear to be associated with political 
#party affiliation.

## 3. While emotions and sentiments expressed by political leaders are poor predictors of 
#population health, select indicators and determinants of health can be used to classify political 
#parties.  
#################################################################################################



