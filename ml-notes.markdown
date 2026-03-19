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

Modello di ML con un numero di layer nascosti tipicamente compreso tra 1 e 3
Neurone artificiale -> singolo unità di elaborazione delle informazioni

Formato da:
- Input provenienti da altri neuroni
- Pesi imparati
- Activation Function
- Bias(valore esterno che permette di aumentare o diminuire l'effetto dell'input nell'activation function)
![[Pasted image 20260317220324.png]]


## PERCEPTRON
la più semplice rete neurale che si può descirvere, Ha un SOLO NEURONE
che ha come activation function una funzione di threshold
$$y = \begin{cases} 0 & \text{se } w^T x < 0 \\ 1 & \text{se } w^T x \geq 0 \end{cases}$$

---

## **LEARNING**

- inizializza $w$ a caso
    
- finché $w$ non converge:
    
    - prendi a caso $x$ dal dataset
        
    - se $x$ è positivo e $w^T x < 0$: $\leftarrow$ _cioè se $x$ è positivo ma sarebbe classificato come negativo_
        
        $$w = w + x$$
        
    - se $x$ è negativo e $w^T x > 0$: $\leftarrow$ _cioè se $x$ è negativo ma sarebbe classificato come positivo_
        
        $$w = w - x$$

se i dati sono LINEARLY separable, convergerà ma se non lo sono, non converge

## ARTIFICIAL NEURAL NETWORK
Un insieme di perceptron

**ARCHITETTURA**
FEEDFORWARD ARCH. -> L'informazione si propaga dai neuroni di input a quelli di output

-> SINGLE LAYER PERCEPTRON: ![[Pasted image 20260317221258.png]]
L'output dipende da: INPUT, PESI e FUNZIONE DI ATTIVAZIONE

RECURRENT ARCH -> L'informazioen può tornare indietro.

MULTI LAYER PERCEPTRON
L'output dipende da output dell'hidden layer, pesi e funzione di attivazione
può essere espresso come funzione parametrizzata che ho come parametri i pesi.

TEOREMA DELL'APPROSSIMAZIONE UNIVERSALE
Una rete neural con un hidden layer può approssimare cn un errore piccolo o a piacere QUALSIASI FUNZIONE CONTINUA.

![[Pasted image 20260317221606.png]]
Il numero di neuroni e il numero di layer sono **IPERPARAMETRI**
- Troppi HIDDEN NEURON portano a minimizzare l'input, ciò causa OVERFITTING
- Pochi HIDDEN LAYER portano a un eccessiva generalizzazione, ciò causa UNDERFITTING

MLP per 
REGRESSIONE -> L'ultimo layer ha la funzione lineare come funzione di attivazione

Gli HIDDEN NEURON con activation function NON LINEARI permettono al MULTI-LAYER PERCEPTRON di imparare FUNZIONI NON LINEARI

**TRAINING**

La fase di training consiste in un algoritmo iterativo che modifica i pesi con l'obiettivo di minimizzare l'errore dell'output. => Learning => minimizzare l'errore sul training set
Uso un insieme di dati etichettati chiamato **TRAINING SET** $= \{(x^{(t)}, y^{(t)})\}_{t=1}^T$

Possiamo misurare l'errore della rete (ovvero la **LOSS FUNCTION**) ad esempio attraverso la **SUM OF SQUARED ERRORS**:

$$E^{(i)} = \frac{1}{2} \sum_{k=0}^K (z_k - y_k)^2$$

- $E^{(i)}$: errore sull' $i$-esimo esempio.
    
- $K$: classi.
    
- $z_k$: output atteso dal neurone $k$.
    
- $y_k$: output generato dal neurone $k$.
    

### **Quindi ad ogni iterazione:**

1. Calcolo l'output della rete e misuro l'errore $\rightarrow$ **FORWARD PROPAGATION**
    
2. Propago l'errore all'indietro in modo da modificare i pesi della rete $\rightarrow$ **BACKWARD PROPAGATION**
    

---

## **DELTA RULE**

Learning rule adottata nella **BACK PROPAGATION**, consiste nell'aggiornare i pesi di un neurone sottraendo un $\Delta$ calcolato:

$$w = w - \Delta w$$

### **Per gli OUTPUT NEURON:**

$$\Delta w_{jh} = - \eta \cdot \delta_j \cdot z_j$$

- $\eta$: learning rate.
    
- $\delta_j = z_j - y_j$ (errore fatto sulla classe $j$).
    
- $z_j$: output del neurone $j$.
    
- $j$: indice per intendere il $j$-esimo neurone di **OUTPUT**.
    
- $h$: indice per intendere l' $h$-esimo **HIDDEN NEURON** da cui deriva il valore in input al $j$-esimo output neuron.
    

### **Per gli HIDDEN NEURON:**

$$\Delta w_{hi} = - \eta \cdot \sum_j (\delta_j \cdot w_{jh} \cdot (1 - y_j) \cdot y_j) \cdot x_i$$

- $\eta$: learning rate.
    
- $\sum_j$: sommatoria su ogni classe.
    
- $x_i$: input di $i$.
    
- $w_{hi}$: peso in entrata a $h$ e uscita a $i$, entrambi **HIDDEN NEURON**.

# **SUGGERIMENTI PRATICI**

## **LEARNING RATE ($\eta$)**

Scegli $\eta$ in modo che non sia né troppo piccolo né troppo grande.

### **ADAPTIVE LEARNING RATE** ⭐

Far variare $\eta$ durante il **TRAINING**, ad esempio con gli **OPTIMIZER**:

- **ADAGRAD:** (Adaptive Gradient Algorithm) $\rightarrow \eta$ alto con gradienti bassi, $\eta$ basso con gradienti alti $\Rightarrow$ metodo scarso perché i gradienti alti portano $\eta$ ad essere troppo piccolo.
    
- **RMSPROP:** (Root Mean Square Propagation) $\Rightarrow$ adatta $\eta$ in base ai gradienti precedenti.
    
- **SDG:** (Stochastic Gradient Descent) $\rightarrow$ diminuisce $\eta$ ad ogni iterazione per ottimizzare la discesa del gradiente.
    
- **ADAM:** (Adaptive Momentum Estimation) $\Rightarrow$ migliore, basato sull'ottimizzare SDG.
    

---

## **INIZIALIZZAZIONE DEI PESI** ⭐

- ⚠️ Se inizializzassi tutti i pesi con lo stesso valore, la **BACKPROPAGATION** li farà rimanere identici.
    
- ✅ Assegna un valore random vicino a "0".
    

---

## **INPUT NORMALIZZATO** ⭐

Meglio standardizzare l'input con $\mu = 0$ e $\sigma < 1$:

$$z = \frac{x - \mu}{\sigma}$$

---

## **NUMERO DI HIDDEN LAYERS**

Meglio averne **TROPPI** che troppo pochi.

---

## **VANISHING GRADIENTS** ⭐

Le derivate assumono valori $[0, 1]$, quindi sui neuroni molto lontani dall'output nelle MLP i pesi cambiano molto lentamente, dato che la derivata è moltiplicata molte volte.

Questo caso è particolarmente incidente sulla **sigmoid function** dove il gradiente è $\approx 0$ sia su input **TROPPO POSITIVI** che su input **TROPPO NEGATIVI**.

---

## **OVERFITTING** ⭐

Strategie per evitarlo:

### **1. EARLY STOPPING**

Facciamo training del modello fin quando non siamo vicini al **minimo globale**, cioè fin quando l'errore sul **VALIDATION SET** non inizia ad aumentare.

### **2. REGULARIZATION**

Nella fase di learning può succedere che i valori di $W$ diventino inutilmente complessi. Ciò **appesantisce i calcoli** e porta all'**OVERFIT**. Inseriamo nella loss function una funzione **REGULARIZER** che forza il modello a usare pesi semplici:

$$L(w) = \underbrace{\frac{1}{N} \sum_{i=1}^{N} L_i(f(x_i; w), y_i)}_{\text{DATA LOSS}} + \underbrace{\lambda R(w)}_{\text{REGULARIZATION}}$$

#### **Tipi di regularization:**

- **L1 REGULARIZATION:** $R(w) = \sum_{k,l} |w_{k,l}|$
    
- **L2 REGULARIZATION (più usato):** $R(w) = \sum_{k,l} w_{k,l}^2$
    
- **ELASTIC NET:** $R(w) = \sum_{k,l} \beta w_{k,l}^2 + |w_{k,l}|$
    

#### **Regularization Strength ($\lambda$):**

- $\lambda = 0 \Rightarrow$ NO REGULARIZATION.
    
- Più $\lambda$ cresce e più piccoli saranno i pesi cercati dal modello.
    
⭐ Example:  
L2 regularization prevents weights from getting too large → avoids overfitting.  
If λ=1λ=1, and true optimal weight is 5, penalized weight becomes ~4.7 (roughly).
### **3. DROPOUT**

Durante il training spengo dei neuroni a caso.

⭐ Example:

In a hidden layer with 64 neurons, set dropout rate = 0.5.

During training, randomly set 32 neurons to 0 (50% chance each).

Validation/test use full network.

---

## ✍️ Latest Notes
Below you will find my most recent posts and research logs.

