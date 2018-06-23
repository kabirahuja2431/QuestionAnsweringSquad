# QuestionAnsweringSquad
Implementing a question answering system using Stanford's Question Answering Dataset (SQUAD). Different Attention models like BIDAF, Coattention, Self The code is implemented as the **default project of the course CS224n**.

## Requirements:
- Python 2.7
- Tensorflow 1.4.1
- colorama 0.3.9
- nltk 3.2.5
- numpy 1.14.0
- six 1.11.0
- tqdm 4.19.5

## Setup
- Clone assignment repositery: https://github.com/abisee/cs224n-win18-squad.git
- Copy the files from this repositery to the code directory
- run get_started.sh (If you are on a machine without gpu, go to requirements.txt and replace tensorflow-gpu with tensorflow)
- Downlodad pretrained model from https://drive.google.com/drive/folders/1Ivv_9rSo1QwoKvGAGkaxbyZRKIqTU-md?usp=sharing and       extract it to experiments/RNet_Trans/best_checkpoint (you will need to create these directories in experiments)

## Models implemented:
- Dot Product Attention + CharCNN + Pointer Network
- Self Attention
- Self Attention(Additive Attention)
- Self Attention(Scaled Dot Product Attention)

# Training the network
Excecute the following commands
- cd code
- source activate squad
- python main_chars.py --experiment_name=RNet_Trans --mode=train

# Testing the model on dev examples (you can edit data/dev.* files for testing on custom comprehensions)
Execute the following commands
- cd code
- source activate squad
- python main_chars.py --experiment_name=RNet_Trans --mode=show_examples

# Results

| Model | Train F1 | Train EM | Dev F1 | Dev EM|
| ----- |:--------:|:--------:|:------:|------:|
| Basic dot attention + CNN | 0.69 | 0.52 | 0.38 | 0.27|
| Self Attention  | 0.7195| 0.55 | 0.6256 | 0.4727|
| Self Attention (Scaled Dot product Attention) | 0.73 | 0.594 | 0.6438 | 0.4947|

# Experiments:
1. Context: in January 1880 , two of tesla 's uncles put together enough money to help him leave gospić for Prague where he was to study . unfortunately , he arrived too late to enroll at _charles-ferdinand_ university ; he never studied Greek , a required subject ; and he was illiterate in Czech , another required subject . tesla did , however , attend lectures at the university , although , as an auditor , he did not receive grades for the courses .
 QUESTION: which university did tesla audit in 1880 ?
TRUE ANSWER: charles-ferdinand university
 PREDICTED ANSWER: charles-ferdinand university
 F1 SCORE ANSWER: 1.000
 EM SCORE: True
 
 2. CONTEXT: In the 1930s , radio in the united states was dominated by three companies : the Columbia broadcasting system ( cbs ) , the mutual broadcasting system and the national broadcasting company ( nbc ) . the last was owned by electronics manufacturer radio corporation of america ( rca ) , which owned two radio networks that each ran different varieties of programming , nbc blue and nbc red . the nbc blue network was created in 1927 for the primary purpose of testing new programs on markets of lesser importance than those served by nbc red , which served the major cities , and to test drama series . 
QUESTION: what two radio networks did rca own ? 
TRUE ANSWER: nbc blue and nbc red 
PREDICTED ANSWER: nbc blue and nbc red
 F1 SCORE ANSWER: 1.000 
EM SCORE: True

3. Context: From 2005 to 2014 , there were two major league soccer teams in los angeles — the la galaxy and chivas USA— that both played at the stub hub center and were local rivals . however , chivas were suspended following the 2014 mls season , with a second mls team scheduled to return in 2018 . 
QUESTION: which team was suspended from the mls ?
 TRUE ANSWER: chivas USA 
PREDICTED ANSWER: chivas
 F1 SCORE ANSWER: 0.667 
EM SCORE: False

4*. Context: oxygen gas can also be produced through electrolysis of water into molecular oxygen and hydrogen . dc electricity must be used : if ac is used , the gases in each limb consist of hydrogen and oxygen in the explosive ratio 2:1 . contrary to popular belief , the 2:1 ratio observed in the dc electrolysis of acidified water does not prove that the empirical formula of water is h2o unless certain assumptions are made about the molecular formulae of hydrogen and oxygen themselves . a similar method is the _electrocatalytic_ o 2 evolution from oxides and _oxoacids_ . chemical catalysts can be used as well , such as in chemical oxygen generators or oxygen candles that are used as part of the life-support equipment on submarines , and are still part of standard equipment on commercial airliners in case of depressurization emergencies . another air separation technology involves forcing air to dissolve through ceramic membranes based on zirconium dioxide by either high pressure or an electric current , to produce nearly pure o 2 gas .
 QUESTION: what does the electrolysis of water produce ? TRUE ANSWER: oxygen and hydrogen 
PREDICTED ANSWER: molecular oxygen and hydrogen
 F1 SCORE ANSWER: 0.857
 EM SCORE: False


