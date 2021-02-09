import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import pandas
import seaborn as sns
import numpy as np
import datetime
import pathlib
import operator
#pobieranie danych
mypath = pathlib.Path().absolute()
fullpath = str(mypath) + "\\vaccination_tweets.csv"

twitty = pandas.read_csv(fullpath)
type(twitty)
twitty.head()
twitty.columns
twitty.shape

#czyszczenie danych
def oczysc_tekst(tekst):
 tekst = re.sub('@[A-Za-z0–9]+', '', tekst) 
 tekst = re.sub('#', '', tekst) 
 tekst = re.sub('https?:\/\/\S+', '', tekst)
 return tekst

twitty['text'] = twitty['text'].apply(oczysc_tekst)

#subiektywność i polaryzacja
def wyznacz_subiektywnosc(tekst):
    return TextBlob(tekst).sentiment.subjectivity

def wyznacz_polaryzacje(tekst):
    return TextBlob(tekst).sentiment.polarity

print(twitty)

twitty['subjectivity'] = twitty['text'].apply(wyznacz_subiektywnosc)
twitty['polarity'] = twitty['text'].apply(wyznacz_polaryzacje)

twitty.columns

#Podział na pozytywne i negatywne twitty
pozytywneTwity = {}
negatywneTwity = {}
iD = 0
listaIndexow = []
for x in twitty['id']:
    listaIndexow.append(x)

for x in twitty['subjectivity']:
    if float(x) >= float(0.5):
        pozytywneTwity[listaIndexow[iD]] = x
    else: negatywneTwity[listaIndexow[iD]] = x
    iD = iD + 1

ilePozytywnych = len(pozytywneTwity)
ileNegatywnych = len(negatywneTwity)
informacja = "Ilość twitów pozytywnych: " + str(ilePozytywnych) + " Ilośc twitów negatywnych: " + str(ileNegatywnych)
print(informacja)

#Diagram podziału twittów na kraje

plt.figure(figsize=(10,12))
sns.barplot(twitty["user_location"].value_counts().values[0:10],
            twitty["user_location"].value_counts().index[0:10]);
plt.title("Top 10 krajów w ilości twittów ",fontsize=14)
plt.xlabel("Liczba twittów",fontsize=14)
plt.ylabel("Kraje",fontsize=14)
plt.show()

#zestawienie subiektywności i polaryzacji
plt.figure()
for i in range(0, twitty.shape[0]):
  plt.scatter(twitty['polarity'].iat[i], twitty['subjectivity'].iat[i], alpha=0.3,
            cmap='viridis')

plt.title('Wyniki analizy')
plt.xlabel('Polaryzacja') 
plt.ylabel('Subiektywność') 
plt.show()


#sortowanie po dacie oraz wykres polaryzacji na przestrzeni czasu
twitty=twitty.sort_values(by='user_created',ascending=True)
fig, ax = plt.subplots(figsize=(10, 10))

# Add x-axis and y-axis
ax.scatter(twitty['user_created'],
           twitty['polarity'],
           color='purple')

# Set title and labels for axes
ax.set(xlabel="czas",
       ylabel="Polaryzacja",
       title="Polaryzacja na przestrzeni czasu")

plt.show()


#    uczenie
#zamiana datetime na date
import datetime as dt

twitty['user_created'] = pandas.to_datetime(twitty['user_created'])
twitty['user_created'] = pandas.to_datetime(twitty.user_created).dt.strftime('%d/%m/%Y')

#wyszukiwanie tych samych dat i wyliczanie dla nich średniej polaryzacji
tempPolarity=0
countTemp = 0
mlDictionary = {}
listOfDate =  twitty['user_created'].values
listOfValue = twitty['polarity'].values
print(listOfDate)
print(listOfValue)
indexY = 0
for x in listOfDate:
    for y in listOfDate:
        if x == y:
            tempPolarity = tempPolarity + listOfValue[indexY]
            countTemp = countTemp +1
        indexY = indexY + 1
    indexY = 0
    avs = float(tempPolarity) / float(countTemp)
    mlDictionary[x] = float(avs)
    tempPolarity=0
    countTemp = 0
            
print(mlDictionary)
print(len(mlDictionary.values()))    


#przygotowanie danych pod uczenie
first = list(mlDictionary.keys())[0] 
last = list(mlDictionary.keys())[-1] 
print()
index = pandas.date_range(first, last)
index, len(index)
listOfKeys = mlDictionary.keys()
listOfValues = []
for x in mlDictionary.values():
    listOfValues.append([x])
print(len(listOfValues))

df = pandas.DataFrame(data = listOfValues, index =listOfKeys )
df.index.name = "date"
df.columns = ["value"]
df
daysZeroToMax = range(0,len(mlDictionary) )
print(daysZeroToMax)
df['days_from_start'] = daysZeroToMax

x = df['days_from_start'].values.reshape(-1, 1)
y = df['value'].values
print(y)
#regresja liniowa
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(x, y)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model.predict([[1], [7], [500], [501], [502],[503], [2000], [2001], [2002], [2003], [2010] ])
array = np.array(model.predict([[1], [7], [500], [501], [502],[503], [2000], [2001], [2002], [2003], [2010], [6000] ]))

xList = np.array([1,7,500,501,502,503,2000,2001,2002,2003,2010,6000])


#wizualizacja danych powstałych w wyniku regresji

plt.figure(figsize=(6,5))
plt.plot(xList, array, 'o')


m, b = np.polyfit(xList, array, 1)


plt.plot(xList, m*xList + b)

plt.show()

#obraz danych po zmianie datetime na date 

fig, ax = plt.subplots(figsize=(10, 10))

# Add x-axis and y-axis
ax.scatter(mlDictionary.keys(),
           mlDictionary.values(),
           color='purple')

# Set title and labels for axes
ax.set(xlabel="czas",
       ylabel="Polaryzacja",
       title="Polaryzacja na przestrzeni czasu")

plt.show()
