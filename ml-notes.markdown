---
layout: home
title: ML Notes
permalink: /ml-notes/
---

# 🧠 ML Insights: Exploring the Frontier

Welcome to my personal collection of notes and explorations in the world of Machine Learning, Deep Learning, and Artificial Intelligence.

---

## 🚀 Featured Topics
Browse through my latest thoughts and research summaries on:
- **Neural Networks & Optimization**
- **Computer Vision & GANs**
- **Large Language Models (LLMs)**
- **Reinforcement Learning**


# 9. NEURAL NETWORKS

Questa sezione raccoglie appunti e spiegazioni sulle reti neurali artificiali, con attenzione a chiarezza, esempi pratici e formule matematiche. Ideale per studenti e appassionati di ML.

---

## Cos'è una Rete Neurale?

Una rete neurale artificiale (ANN) è un modello di machine learning ispirato al cervello umano, composto da unità chiamate **neuroni artificiali**. Ogni neurone riceve input, applica pesi, una funzione di attivazione e un bias:

- **Input**: segnali provenienti da altri neuroni o dati grezzi
- **Pesi**: valori appresi durante il training
- **Funzione di attivazione**: introduce non linearità
- **Bias**: valore che regola la soglia di attivazione

<img src="/assets/img/Pasted%20image%2020260317220324.png" alt="neurone artificiale" width="350"/>

---

## Il Perceptron

Il **perceptron** è la più semplice rete neurale: un solo neurone con funzione di attivazione a soglia (threshold):

$$
y = \begin{cases} 0 & \text{se } w^T x < 0 \\ 1 & \text{se } w^T x \geq 0 \end{cases}
$$

**Algoritmo di apprendimento:**

1. Inizializza $w$ a caso
2. Finché $w$ non converge:
    - Prendi un esempio $x$ dal dataset
    - Se $x$ è positivo e $w^T x < 0$: aggiorna $w = w + x$
    - Se $x$ è negativo e $w^T x > 0$: aggiorna $w = w - x$

Se i dati sono linearmente separabili, il perceptron converge; altrimenti no.

---

## Artificial Neural Network (ANN)

Un'ANN è un insieme di perceptron organizzati in **layer**:

- **Feedforward**: l'informazione si propaga dagli input agli output
- **Recurrent**: l'informazione può tornare indietro (cicli)

### Multi Layer Perceptron (MLP)

Un MLP ha almeno un **hidden layer**. L'output dipende dagli output degli hidden layer, dai pesi e dalle funzioni di attivazione.

**Teorema dell'approssimazione universale:**
Una rete con un hidden layer può approssimare qualsiasi funzione continua con errore arbitrariamente piccolo.

<img src="/assets/img/Pasted%20image%2020260317221606.png" alt="MLP" width="400"/>

**Iperparametri:**
- Numero di neuroni e layer
- Troppi neuroni: rischio overfitting
- Troppo pochi: rischio underfitting

**Per la regressione:** l'ultimo layer ha funzione di attivazione lineare.

**Hidden neuron** con funzioni di attivazione non lineari permettono di apprendere funzioni non lineari.

---

## Training di una Rete Neurale

Il training consiste nel modificare iterativamente i pesi per minimizzare l'errore sull'output (loss). Si usa un **training set** etichettato $= \{(x^{(t)}, y^{(t)})\}_{t=1}^T$.

**Loss function (esempio: sum of squared errors):**

$$
E^{(i)} = \frac{1}{2} \sum_{k=0}^K (z_k - y_k)^2
$$

- $E^{(i)}$: errore sull'$i$-esimo esempio
- $K$: classi
- $z_k$: output atteso
- $y_k$: output generato

**Fasi:**
1. Calcola output e misura errore (**forward propagation**)
2. Propaga l'errore all'indietro e aggiorna i pesi (**backward propagation**)

---

## Delta Rule (Regola di Apprendimento)

Nella backpropagation, i pesi vengono aggiornati sottraendo un $\Delta$ calcolato:

$$w = w - \Delta w$$

**Per gli output neuron:**

$$
\Delta w_{jh} = - \eta \cdot \delta_j \cdot z_j
$$

- $\eta$: learning rate
- $\delta_j = z_j - y_j$: errore sulla classe $j$
- $z_j$: output del neurone $j$
- $h$: hidden neuron collegato

**Per gli hidden neuron:**

$$
\Delta w_{hi} = - \eta \cdot \sum_j (\delta_j \cdot w_{jh} \cdot (1 - y_j) \cdot y_j) \cdot x_i
$$

- $x_i$: input
- $w_{hi}$: peso tra hidden neuron e input

---

## Suggerimenti Pratici

### Learning Rate ($\eta$)
Scegli $\eta$ né troppo piccolo né troppo grande.

#### Adaptive Learning Rate ⭐
Fai variare $\eta$ durante il training con ottimizzatori:
- **Adagrad**: $\eta$ alto con gradienti bassi, basso con gradienti alti (può diventare troppo piccolo)
- **RMSProp**: adatta $\eta$ in base ai gradienti precedenti
- **SGD**: diminuisce $\eta$ ad ogni iterazione
- **Adam**: il più usato, combina i vantaggi di SGD e momentum

### Inizializzazione dei pesi ⭐
- Non inizializzare tutti i pesi uguali!
- Usa valori random vicini a zero

### Input normalizzato ⭐
Standardizza l'input: $\mu = 0$, $\sigma < 1$

$$z = \frac{x - \mu}{\sigma}$$

### Numero di hidden layer
Meglio troppi che troppo pochi (ma attenzione all'overfitting)

### Vanishing Gradients ⭐
Con molte moltiplicazioni di derivate $\in [0,1]$, i pesi degli strati lontani dall'output cambiano molto lentamente (problema accentuato con la sigmoid)

### Overfitting ⭐
Strategie:
1. **Early stopping**: interrompi il training quando l'errore sul validation set inizia a crescere
2. **Regularization**: aggiungi un termine alla loss per penalizzare pesi troppo grandi

$$L(w) = \underbrace{\frac{1}{N} \sum_{i=1}^{N} L_i(f(x_i; w), y_i)}_{\text{DATA LOSS}} + \underbrace{\lambda R(w)}_{\text{REGULARIZATION}}$$

Tipi:
- **L1**: $R(w) = \sum_{k,l} |w_{k,l}|$
- **L2** (più usato): $R(w) = \sum_{k,l} w_{k,l}^2$
- **Elastic Net**: $R(w) = \sum_{k,l} \beta w_{k,l}^2 + |w_{k,l}|$

Più $\lambda$ cresce, più i pesi saranno piccoli.

⭐ Esempio: L2 regularization impedisce ai pesi di diventare troppo grandi → meno overfitting.

3. **Dropout**: durante il training, spegni casualmente alcuni neuroni (es: con 64 neuroni e dropout 0.5, ne spegni 32 a caso ad ogni iterazione)

---

# 12. KNOWLEDGE GRAPHS

Grafo di dati molto grande, con l'obiettivo di rappresentare la conoscenza in modo strutturato. Composto da **nodi** (entità) e **archi** (relazioni).
Basato sull'OPEN WORLD ASSUMPTION: se una relazione non è presente, non significa che sia falsa, ma semplicemente che non è nota.

Un grafo è descritto da un set di STATEMENTS

G = { (s, p, t) } 

## Machine Learning per Knowledge Graphs
Migliorare la performance nei trask di ML sfruttando la conoscenza del KG. Esempi:
- **Node classification**: classificare i nodi in base alle loro caratteristiche e relazioni
- **Link prediction**: predire relazioni mancanti tra nodi
Però per poter fare ML su un KG, dobbiamo prima rappresentarlo in modo che un modello possa processarlo. Ci sono due approcci principali:
1. **Knowledge Graph Embeddings**: rappresentare nodi e relazioni come vettori in uno spazio continuo
2. **Graph Neural Networks (GNNs)**: estendere le reti neurali per lavorare direttamente su grafi, sfruttando la struttura del grafo per apprendere rappresentazioni più ricche.

## Knowledge Graph Embeddings

**EMBEDDING:** mappa nodi e relazioni in uno spazio vettoriale continuo, preservando la struttura del grafo.

Esempi di modelli:

Un modello di ML per rappresentare concetti e relazioni di un KG in uno spazio vettoriale continuo, preservando la struttura del grafo.

Cosa fa? Usa la funzione di scoring per valutare una nuova tripla:

$$
f(s, p, o) = \text{score}
$$

Valore alto se la tripla è **positiva**, basso se è **negativa**.

**Esempi di modelli:**

- **TransE:** rappresenta le relazioni come traslazioni nello spazio vettoriale

  $$
  f_{\text{TransE}}(s, p, o) = -\lVert \mathbf{e}_s + \mathbf{e}_p - \mathbf{e}_o \rVert_n
  $$

- **RotatE:** rappresenta le relazioni come rotazioni nello spazio vettoriale

  $$
  f_{\text{RotatE}}(s, p, o) = -\lVert \mathbf{e}_s \circ \mathbf{e}_p - \mathbf{e}_o \rVert_n
  $$

La **loss function** è la funzione da minimizzare durante il training, che penalizza le triplette negative e premia quelle positive.

**Pairwise Margin-based Hinge Loss:**

$$
L = \sum_{(s,p,o)\in G} \sum_{(s',p,o')\notin G} \left[\gamma + f(s', p, o') - f(s, p, o)\right]_+
$$

dove:
- $G$: insieme delle triplette positive
- $(s', p, o')$: triplette negative generate da negative sampling
- $[x]_+ = \max(0, x)$: funzione hinge che penalizza solo se la tripla negativa ha score più alto di quella positiva di almeno $\gamma$

**Cross-Entropy Loss:**

$$
L = - \sum_{(s,p,o)\in G} \log \sigma(f(s, p, o)) - \sum_{(s',p,o')\notin G} \log (1 - \sigma(f(s', p, o')))
$$

dove:
- $\sigma(x) = \frac{1}{1 + \exp(-x)}$: funzione sigmoid che mappa lo score in $[0, 1]$, interpretato come probabilità che la tripla sia positiva.

La loss penalizza le triplette positive con score basso e le triplette negative con score alto.

REGULARIZATION: aggiungere un termine alla loss per evitare overfitting, ad esempio:   
$$L_{\text{reg}} = L + \lambda \sum_{i} \lVert \mathbf{e}_i \rVert^2$$
dove $\lambda$ è il coefficiente di regularizzazione e $\mathbf{e}_i$ sono gli embedding dei nodi e delle relazioni.

L1,L2, L3, Dropout, etc.

Initialization: Random

Negative Generation: per ogni tripla positiva, genera triplette negative sostituendo casualmente il soggetto o l'oggetto con un'entità casuale. 
Usiamo la **LOCAL CLOSED WORLD ASSUMPTION**: assume che la KB è completa per le entità coinvolte, quindi se una tripla non è presente, è considerata negativa.

---

# 📚 Domande Frequenti d'Esame: Approfondimenti

## Variational Autoencoders (VAE)

I **Variational Autoencoders** sono una famiglia di autoencoder probabilistici. L'obiettivo è apprendere una rappresentazione compatta (embedding) dei dati che sia continua e interpretabile, utile per generare nuovi dati simili a quelli osservati.

- **Encoder:** mappa l'input $x$ in una distribuzione latente $q_\phi(z|x)$ (tipicamente gaussiana, con media e varianza apprese).
- **Decoder:** ricostruisce l'input a partire da un campione $z$ estratto dalla distribuzione latente.
- **Loss:** somma di due termini:
    - Ricostruzione: misura quanto il dato ricostruito è simile all'input.
    - Divergenza KL: penalizza la distanza tra la distribuzione latente appresa e una distribuzione prior (es. gaussiana standard).

$$
\mathcal{L}_{VAE} = \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
$$

---

## Denoising Autoencoder

Un **Denoising Autoencoder** è un autoencoder che impara a ricostruire l'input originale a partire da una versione "rumorosa" (noisy) dell'input.

- **Encoder:** comprime l'input rumoroso in una rappresentazione latente.
- **Decoder:** ricostruisce l'input pulito dalla rappresentazione latente.
- **Obiettivo:** imparare rappresentazioni robuste che catturino le caratteristiche essenziali dei dati.

---

## RNN (Recurrent Neural Network)

Le **RNN** sono reti neurali progettate per sequenze di dati (testo, serie temporali, ecc.).

- Ogni output dipende sia dall'input corrente che dallo stato precedente (memoria).
- Architetture comuni: Simple RNN, LSTM, GRU.
- Utili per: traduzione automatica, analisi del sentiment, generazione di testo.

---

## Costi nella Classificazione

I **costi** nella classificazione rappresentano l'importanza relativa degli errori (falsi positivi, falsi negativi, ecc.).

- Si usano quando gli errori hanno impatti diversi (es. diagnosi medica: meglio un falso positivo che un falso negativo).
- Si implementano tramite una **matrice dei costi** o pesando la loss function.
- Esempio: in una loss pesata, si può dare più peso agli errori su una classe minoritaria.

---

## Negative Sampling nei KGE

Il **negative sampling** è una tecnica per generare esempi negativi (triplette false) durante il training dei modelli di Knowledge Graph Embedding.

- Per ogni tripla positiva $(s, p, o)$, si genera una tripla negativa sostituendo $s$ o $o$ con un'entità casuale.
- Serve per insegnare al modello a distinguere relazioni plausibili da quelle implausibili.
- Si basa spesso sulla **Local Closed World Assumption**: se una tripla non è presente, si assume negativa.

---

## Architettura base dei KGE models

- **Input:** triplette $(s, p, o)$ (soggetto, predicato, oggetto)
- **Embedding Layer:** ogni entità e relazione è rappresentata da un vettore (embedding)
- **Scoring Function:** calcola uno score per la tripla (es. TransE, RotatE)
- **Loss Function:** penalizza le triplette negative e premia le positive
- **Negative Sampling:** genera esempi negativi per l'addestramento

---

## Random Forest

- Insieme di **alberi decisionali** addestrati su sottoinsiemi diversi dei dati e delle feature.
- Ogni albero vota per la classe; la classe finale è quella più votata (classificazione) o la media (regressione).
- Vantaggi: robustezza, riduzione dell'overfitting, gestione di dati rumorosi.

---

## Decision Tree

- Modello predittivo che suddivide ricorsivamente i dati in base alle feature.
- Ogni nodo interno rappresenta una decisione su una feature; ogni foglia una classe o valore.
- Algoritmi comuni: CART, ID3, C4.5.
- Criteri di split: **Gini**, entropia, varianza.

---

## Logistic Regression

- Modello lineare per la classificazione binaria.
- Calcola la probabilità che un input appartenga alla classe positiva tramite la funzione sigmoid:

$$
P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}
$$

- Loss: **log-loss** (cross-entropy)
- Può essere regolarizzata (L1, L2)

---

## Che problema risolve Gini

L'**indice di Gini** misura la "purezza" di un nodo in un albero decisionale.
- Serve a scegliere lo split migliore: più basso è il Gini, più puro è il nodo.
- Formula:

$$
Gini = 1 - \sum_{i=1}^C p_i^2
$$

dove $p_i$ è la proporzione di elementi della classe $i$ nel nodo.

---

## Regolarizzazione

- Tecnica per evitare l'overfitting penalizzando pesi troppo grandi.
- Si aggiunge un termine alla loss:
    - **L1** (lasso): penalizza la somma dei valori assoluti dei pesi
    - **L2** (ridge): penalizza la somma dei quadrati dei pesi
- Esempio (L2):

$$
L = L_{data} + \lambda \sum_j w_j^2
$$

---

## SVM (Support Vector Machine)

- Modello per classificazione che cerca l'**iperpiano** che massimizza il margine tra le classi.
- Può usare kernel per separare dati non linearmente separabili.
- Loss: **hinge loss**

---

## Feedforward Neural Network

- Rete neurale in cui l'informazione fluisce solo in avanti (dagli input agli output).
- Composta da uno o più layer (input, hidden, output).
- Ogni neurone calcola una combinazione pesata degli input, applica una funzione di attivazione e passa il risultato al layer successivo.
- Non ha memoria (a differenza delle RNN).

---