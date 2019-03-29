
# coding: utf-8

# In[1]:


import json
import codecs
import re
import pandas as pd


# In[2]:


import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import nltk
from num2words import num2words
from nltk.stem.snowball import PortugueseStemmer
import string


# In[3]:


from sklearn.utils import shuffle


# In[4]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression  # Logistic Regression
# from sklearn.cross_validation import train_test_split # Para dividir o conjunto de treinamento e teste
from sklearn.model_selection import train_test_split # Para dividir o conjunto de treinamento e teste
from sklearn.neighbors import KNeighborsClassifier  # K nearest neighbours
from sklearn import svm  # Para o algoritmo Support Vector Machine (SVM) Algorithm
from sklearn import metrics # Para verificar as métricas
from sklearn.tree import DecisionTreeClassifier # para o algoritmo de árvores de decisão
from sklearn.neural_network import MLPClassifier # Para as redes neurais
from sklearn import ensemble, naive_bayes, neighbors, svm, tree
from sklearn.preprocessing import MinMaxScaler

# adicionado
nltk.download('stopwords')


# In[ ]:


def beep():
    import winsound
    duration = 1000  # millisecond
    freq = 440  # Hz
    winsound.Beep(freq, duration)


# Metodo para ler o arquivo json recebido

# In[5]:


def read_json_file(path, enc='utf8'):
    # adicionado trecho
    myFile = open(path, 'r')
    myObject = myFile.read()
    u = myObject.encode().decode('utf-8-sig')
    myObject = u.encode('utf-8')
    myFile.encoding
    myFile.close()
    # alterado o valor de path
    path = myObject
    with codecs.open(path, encoding=enc) as j:
        data_json = json.load(j)
    return data_json


# Metodo para remover o index do documento (cabecalho)

# In[6]:


def remove_document_index(x):
    return re.sub('.*(?<=\\r1)\.', '', x)


# Método para filtrar as stopwords

# In[7]:


def filter_stopwords(tokens):
    """Docstring."""
    return [i.lower() for i in tokens if
            i.lower() not in stopwords] 


# Método para aplicar o stemming - Note que neste caso a pontuacao foi removida

# In[8]:


def stemming(x):
    return _stemmer.stem(x)


# Transforma os digitos em texto (e.g. 1 -> um, 2 -> dois)

# In[9]:


def number_to_word(word, language = 'pt_BR'):
    try:
        return num2words(float(word), to = 'cardinal', lang = language)
    except NotImplementedError:
        return word


# Extrai as metricas do modelo

# In[43]:


def extract_model_metrics(y_predicted, y_test, positive_label=1):
        model_metrics = dict()
        model_metrics['accuracy'] = metrics.accuracy_score(y_test, y_predicted)
        model_metrics['f1'] = metrics.f1_score(y_test, y_predicted, average='weighted', pos_label=1)
        model_metrics['precision'] = metrics.precision_score(y_test, y_predicted, average='weighted', pos_label=1)
        model_metrics['recall'] = metrics.recall_score(y_test, y_predicted, average='weighted', pos_label=1)

        fpr, tpr, threshold = metrics.roc_curve(
            y_test, 
            y_predicted.tolist()
        )

        # TODO: Rever isso, pois é sobreescrito em seguida
        model_metrics['true_positive'] = tpr
        model_metrics['false_positive'] = fpr
        model_metrics['auc'] = metrics.auc(fpr, tpr)
        model_metrics['kappa'] = metrics.cohen_kappa_score(y_test, y_predicted)
        model_metrics['log_loss'] = metrics.log_loss(y_test, y_predicted)

        confusion_matrix = pd.DataFrame(metrics.confusion_matrix(y_test, y_predicted))

        _tp = confusion_matrix.iloc[1,1]
        _tn = confusion_matrix.iloc[0,0]
        _fp = confusion_matrix.iloc[0,1]
        _fn = confusion_matrix.iloc[1,0]

        model_metrics['true_positive'] = confusion_matrix.iloc[1,1]
        model_metrics['true_negative'] = confusion_matrix.iloc[0,0]
        model_metrics['false_positive'] = confusion_matrix.iloc[0,1]
        model_metrics['false_negative'] = confusion_matrix.iloc[1,0]

        model_metrics['positive_pred_value'] = 0.0 if (_tp + _fp) == 0 else (_tp / (_tp + _fp))
        model_metrics['negative_pred_value'] = 0.0 if (_tn + _fn) == 0 else (_tn / (_tn + _fn))

        model_metrics['sensitivity'] = 0.0 if (_tp + _fn) == 0 else (_tp / (_tp + _fn))
        model_metrics['specificity'] = 0.0 if (_tn + _fp) == 0 else (_tn / (_tn + _fp))

        model_metrics['expected_no'] = Counter(y_test)[0]
        model_metrics['expected_yes'] = Counter(y_test)[1]
        model_metrics['diff_expected'] = abs(Counter(y_test)[1] - Counter(y_predicted)[1])

        for x in model_metrics.keys():
            model_metrics[x] = float(model_metrics[x])
        
        return model_metrics


# Definições globais

# In[11]:


_stemmer = PortugueseStemmer()
stopwords = nltk.corpus.stopwords.words('portuguese')
punct = string.punctuation


# In[12]:


# data_civil = read_json_file('C:/Users/allanbs/Documents/Git/RicardoDissertacao/dados/Formato Json/Civel.json')
data_civil = read_json_file('arquivos/json/civel.json')

# In[13]:


# data_crime = read_json_file('C:/Users/allanbs/Documents/Git/RicardoDissertacao/dados/Formato Json/Crime.json')
data_crime = read_json_file('arquivos/json/crime.json')

# In[14]:


data_crime = data_crime['Documentos']
k = list()
for x in data_crime:
    k.append(remove_document_index(x['EMENTA']))

data_crime = k


# In[15]:


data_civil = data_civil['Documentos']
k = list()
for x in data_civil:
    k.append(remove_document_index(x['EMENTA']))

data_civil = k


# Remove o cabeçalho e caracteres de controle

# In[16]:


for i in range(0, len(data_civil)):
    x = data_civil[i]
    x = re.sub('.*(?<=\\r1)\.', '', x)
    x = re.sub(r'[\t\n\r]', ' ', x)
    x = re.sub("\s\s+", " ", x)
    data_civil[i] = x


# In[17]:


for i in range(0, len(data_crime)):
    x = data_crime[i]
    x = re.sub('.*(?<=\\r1)\.', '', x)
    x = re.sub(r'[\t\n\r]', ' ', x)
    x = re.sub("\s\s+", " ", x)
    data_crime[i] = x


# Aplica o filtro por stopwords em todos os textos

# In[18]:


data_civil = [' '.join(filter_stopwords(x.split(' '))) for x in data_civil]
data_crime = [' '.join(filter_stopwords(x.split(' '))) for x in data_crime]


# Aplica o filtro o stemming em todos os textos

# In[19]:


data_civil = [stemming(x) for x in data_civil]
data_crime = [stemming(x) for x in data_crime]


# Cria o vectorizer para processar todos os textos de TF-IDF

# In[20]:


vectorizer = TfidfVectorizer()


# Repassa todos os textos para o TF-IDF

# In[21]:


texts = data_civil
texts.extend(data_crime)
vectorizer.fit(texts)
del texts


# Obtém o vetor TF-IDF para o primeiro elemento de todos os textos

# In[22]:


data_civil = vectorizer.transform(data_civil).toarray()
data_crime = vectorizer.transform(data_crime).toarray()


#  

# Gera os dataframes e atribui os labels dos dados

# In[23]:


data_civil = pd.DataFrame(data_civil,columns=vectorizer.get_feature_names())
data_crime = pd.DataFrame(data_crime,columns=vectorizer.get_feature_names())


# In[24]:


data_civil['CLASSE'] = 'civil'
data_crime['CLASSE'] = 'crime'


# In[25]:


all_data = pd.concat([data_civil, data_crime], axis=0)


# In[26]:


all_data = shuffle(all_data)


# In[27]:


beep()


# Divide o conjunto em treinamento (70%) e teste (30%)

# In[44]:


all_data['CLASSE'] = [1 if x == 'crime' else 0 for x in all_data['CLASSE']]
x_train, x_test, y_train, y_test = train_test_split(all_data.drop('CLASSE', axis=1, inplace=False), all_data['CLASSE'], test_size = 0.3, random_state = 42)


# Configuração da semente

# In[30]:


random_state_seed = 2567


# Aplica o SVM com kernel linear

# In[ ]:


model = svm.SVC(kernel='linear', gamma='auto', C=1, degree=0.1, probability=False, random_state=random_state_seed)
model.fit(x_train,y_train) # nós treinamos o algoritmo com os dados de treinamento e a saída de treinamento
y_predicted=model.predict(x_test) # agora passamos os dados de teste para o algoritmo treinado

# Para verificar o desempenho, é necessário passar a saída obtida pelo modelo e a esperada
extract_model_metrics(y_predicted,y_test) # agora nós verificamos a acurácia do algoritmo.


# Aplica o SVM com kernel radial

# In[60]:


model = svm.SVC(kernel='rbf', gamma='auto', C=1, degree=0.1, probability=False, random_state=random_state_seed)
model.fit(x_train,y_train) # nós treinamos o algoritmo com os dados de treinamento e a saída de treinamento
y_predicted=model.predict(x_test) # agora passamos os dados de teste para o algoritmo treinado

# Para verificar o desempenho, é necessário passar a saída obtida pelo modelo e a esperada
extract_model_metrics(y_predicted,y_test) # agora nós verificamos a acurácia do algoritmo.


# Aplica o SVM com kernel polinomial

# In[ ]:


model = svm.SVC(kernel='poly', gamma='auto', C=1, degree=0.1, probability=False, random_state=random_state_seed)
model.fit(x_train,y_train) # nós treinamos o algoritmo com os dados de treinamento e a saída de treinamento
y_predicted=model.predict(x_test) # agora passamos os dados de teste para o algoritmo treinado

# Para verificar o desempenho, é necessário passar a saída obtida pelo modelo e a esperada
extract_model_metrics(y_predicted,y_test) # agora nós verificamos a acurácia do algoritmo.


# Aplica a regressão logistica

# In[53]:


model = LogisticRegression(random_state=random_state_seed)
model.fit(x_train,y_train)
y_predicted=model.predict(x_test)
extract_model_metrics(y_predicted,y_test)


# Aplica árvores de decisão

# In[58]:


model=DecisionTreeClassifier(random_state=random_state_seed)
model.fit(x_train,y_train)
y_predicted=model.predict(x_test)
extract_model_metrics(y_predicted,y_test)


# Aplica o algoritmo do KNN

# In[59]:


model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
y_predicted=model.predict (x_test)
extract_model_metrics(y_predicted,y_test)


# Aplica o algoritmo Random Forest

# In[61]:


model=ensemble.RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None, n_jobs=-1, bootstrap=True, random_state=random_state_seed)
model.fit(x_train,y_train)
y_predicted=model.predict(x_test)
extract_model_metrics(y_predicted,y_test)


# Aplica o algoritmo Naive Bayes

# In[55]:


model=naive_bayes.GaussianNB()
model.fit(x_train,y_train)
y_predicted=model.predict(x_test)
extract_model_metrics(y_predicted,y_test)

