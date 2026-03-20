---
layout: math
mathjax: true
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

# 1. INTRODUCTION

Nella pPROGARAMMAZIONE NORMALE, presi in input DATI e REGOLE generiamo un OUTPUT. Nella PROGRAMMAZIONE LOGICA, presi in input DATI e OUTPUT generiamo REGOLE. Nella MACHINE LEARNING, presi in input DATI e OUTPUT generiamo un MODELLO. I modelli di ML sono sistemi in grado di sviluppare loro il modo di risolvere un problema, a partire da esempi di input e output. Il modello impara a generalizzare, cioè a fare previsioni su dati nuovi, non visti durante il training, basandosi sui pattern appresi dai dati di training e sul loro comportamento.

## PARADIGMI DI MACHINE LEARNING

┌─────────────────────────────────────────────────────────────────────────────┐
│                         PARADIGMI DI MACHINE LEARNING                        │
└──────┬───────────────────────────────┬───────────────────────────────┬──────┘
       │                               │                               │
┌──────▼─────────────────┐     ┌───────▼───────────────┐     ┌─────────▼─────────────┐
│   SUPERVISED LEARNING  │     │ REINFORCEMENT LEARNING│     │ UNSUPERVISED LEARNING │
│ (Conosce l'output per  │     │ (Cerca di massimiz-   │     │ (Cerca pattern        │
│  ogni esempio input)   │     │  zare un PAYOFF)      │     │  nell'input)          │
└──────┬─────────────────┘     └───────────────────────┘     └─────────┬─────────────┘
       │                                                               │
       ├─► CLASSIFICATION                                              ├─► CLUSTERING
       │   (Output [1,N], label)                                       │   (Gruppi relazionati)
       │                                                               │
       ├─► REGRESSION                                                  ├─► ANOMALY DETECTION
       │   (Output in ℝ)                                               │   (Differenze tra dati)
       │                                                               │
       └─► ORDINAL REGRESSION                                          └─► ASSOCIATION
           (Output ordinato,                                               (Es: consigli acquisti)
            es: stelle TripAdvisor)

<br/>
<details>
<summary><strong>Tabella paradigmi (Markdown)</strong></summary>

| Paradigma                | Descrizione                        | Esempi principali           |
|--------------------------|------------------------------------|----------------------------|
| Supervised Learning      | Conosce l'output per ogni input    | Classification, Regression, Ordinal Regression |
| Unsupervised Learning    | Cerca pattern nell'input           | Clustering, Anomaly Detection, Association     |
| Reinforcement Learning   | Massimizza un payoff               | -                          |

</details>

## CLASSIFICATION

L'OBIETTIVO è trovare un modo per mappare ogni Xi del training set a una classe Yi. 
Con yi ∈ {1, ...., C} (C classi) e xi ∈ ℝ^d (d feature).
se C = 2 → CLASSIFICAZIONE BINARIA
se C > 2 → CLASSIFICAZIONE MULTI-CLASSE
se le CLASSI NON SONO MUTUAMENTE ESCLUSIVE → CLASSIFICAZIONE MULTI-LABEL (es: classificare un film in più generi)

L'obiettivo è fare una predizione corretta su un nuovo input. Per disambiguare meglio, il modello deve ritornare una distribuzione di probabilità sulle classi possibili. 
Quindi consideriamo come predizione $$\hat{y} = \text{ARGMAX}_{c=1}^{C} \ P(y=c \mid x, D)$$
D is the training data, x is the new input

Il SUPERVISED LEARNING p una CONDITIONAL DENSITY ESTIMATION

UNSUPERVISED LEARNING:
Cerco un modelo nella forma P(X_i| theta) che spiega i dati osservati. quindi si tratta di una UNCONDITIONAL DENSITY ESTIMATION.

Un esempio è il CLUSTERING
Se k denota il numero di cluster:
1° Obiettivo (fase di training): stimare la distribuzione degli elementi di training sul numero di cluster k
2° Obiettivo (uso del modello): stimare a quale cluster appartiene un nuovo elemento x, cioè stimare P(k|x)

K = ARGMAX_{k} P(k|D) 
z_i = argmax_{k} P(Z_i = k | x_i, D) k identifica il cluster in particolare.

CONCETTI DI BASE:

MODELLI PARAMETRICI o NON-PARAMETRICI
Se un modello ha un numero finito di parametri, allora è un modello PARAMETRICO(più veloci). Se invece ha un numero di parametri che cresce con la quantità di dati, allora è NON-PARAMETRICO(più flessibili).

Esempio di NON-PARAMETRIC MODEL: KNN (K-NEAREST NEIGHBORS CLASSSIFIER)
- Non ha parametri da stimare, ma memorizza tutto il training set
Classifcatore che guarda ai K punti nel vicinato di X.
La probabilità che la classe di output sia C noto l'input x è data da:
P(y=C|x,D,K) = (1/K) * sum_{i=1}^{K} 1(y_i = C)

Esempio di PARAMETRIC MODEL: LINEAR REGRESSION
Funzione lineare sull'input del tipo Y(X) = w^Tx + ε, w è la matrice dei parametri

OVERFITTING: quando un modello è troppo complesso e si adatta troppo bene ai dati di training, perdendo la capacità di generalizzare a nuovi dati.
UNDERFITTING: quando un modello è troppo semplice e non riesce a catturare
OPTIMAL MODEL COMPLEXITY: il punto in cui il modello è abbastanza complesso da catturare i pattern nei dati, ma non così complesso da adattarsi al rumore.

# 2. MODEL EVALUATION

## EVALUATION
Utile a capire quale modello funziona meglio sul task

Validare l'errore sul TRAINING SET non è un buon indicatore sui dati FUTURI, questo è detto RESUBSTITUION ERROR, non ci dà info sull'OVERFIT ma ce le dà sull'UNDERFIT. Quindi è comunque utile conoscerlo quindi ci serve un TEST-SET indipendente -> instanze simili al training-set, ma che non compaiono nel learning process.

Le procedure migliori sono TRE SETS:
- TRAINING SET: usato per addestrare il modello
- VALIDATION SET: usato per scegliere il modello migliore tra quelli addestrati
- TEST SET: usato per stimare l'errore del modello scelto sui dati futuri

Presi tutti i dati che abbiamo, per dividerli tra training e test possiamo fare: 
HOLDOUT: dividere il set in 2/3 per il training e 1/3 per il test
STRATIFIED SAMPLING: dividere il set in modo che la distribuzione delle classi sia simile tra training e test
REPEATED HOLDOUT METHOD: facciamo più cicli dove dividiamo a caso 2/3 e 1/3, addestriamo il modello e valutiamo l'errore sul test, alla fine facciamo la media degli errori.

K-FOLD CROSS VALIDATION: dividiamo il set in K fold mantenendo la distribuzione delle classi, addestriamo il modello su K-1 fold e valutiamo l'errore sul fold rimanente, ripetiamo per tutti i fold e alla fine facciamo la media degli errori.

LEAVE-ONE-OUT CROSS VALIDATION: è un caso particolare di K-FOLD con K = N (numero di istanze), addestriamo il modello su tutte le istanze tranne una e valutiamo l'errore su quella rimanente, ripetiamo per tutte le istanze e alla fine facciamo la media degli errori. Considerando N elemetni nel dataset, consiste nel fare N iterazioni usando 1 elemneto come TEST (sempre diverso) e N-1 elementi per il train. E' computazionalmente costoso e non è possibile stratificare i fold, ma è utile quando abbiamo pochi dati.

BOOTRAP o BOOTSTRAP O,632

Presi N elementi del dataset, faccio N pesche con rimessa.
Ad ogni pesca, se l'elemento non è nel train-set, lo aggiungo, altrimenti non faccio niente. Statisticamente dopo N pesche dovremmo avere il 65,2% degli elementi del training-set, gli altri nel test-set. La procuedura va ripetuta più volte e alla fine si fa la media dei risultati.
Dato che il train è effettuato su pochi elementi, lo consideriamo nel calcolo dell'errore:
err=0.632 * err_test + 0.368 * err_train

## SELEZIONE DEGLI IPERPARAMETRI
Gli iperparametri sono parametri che fanno della fase di learning e quindi devono essere ottimizzati usando i dati di TRAIN, Per cui facciamo un'ulteriore divisione del training-set, in train e validation.
Es: per stimare K ottimale nella K-NN,
1) dopo aver diviso in TRAIN e VAL, PER OGNI K, addestriamo il modello sul TRAIN e valutiamo l'errore sul VAL
2) Trainiamo il modello con K ottimale su tutto il TRAIN e valutiamo l'errore sul TEST

Se il dataset è troppo piccolo da poter essere splittato un'altra volta, si può considerare una nested cross validation, quindi uno sula suddivisione TEST-TRAIN, e uno sulla suddivisione TRAIN-VAL. (Computazionalmente costososissimo)

## INTERVALLO DI CONFIDENZA
Ci serve stimare l'intervallo di confidenza dell'errore stimato rispetto a quello effettivo, Ovviamente questo intervallo dipende dal numero di elementi nel test-set, presi N provem di cui S successi, ci interessa definire l'intervallo a cui appartiene il true success rate p invece il success rate f = S/N <- f è una variabile aleatoria con media p e varianza p(1-p)/N per N grande a piacere F segue una Distribuzione Normale, All'aumentare di N, si rimpicciolisce l'intervallo, quindi possiamo stimare l'intervallo di confidenza al 95% con la formula:

All'aumentare di $N$, si rimpicciolisce l'intervallo di confidenza.Definito una confidenza desiderata $c$ (di solito 90/98 %), trovo il valore di $z$ associato da qui:
| $\text{Pr}[X \ge z]$ | $z$ | $c$ |
| :--- | :--- | :--- |
| 0.1% | 3.09 | 99.8% |
| 0.5% | 2.58 | 99% |
| 1% | 2.33 | 98% |
| **5%** | **1.65** | **90%** |
| 10% | 1.28 | 80% |
| 20% | 0.84 | 60% |
| 40% | 0.25 | 20% |
E applico questa formula:
$$p = \frac{\left( f + \frac{z^2}{2N} \pm z \sqrt{\frac{f}{N} - \frac{f^2}{N} + \frac{z^2}{4N^2}} \right)}{\left( 1 + \frac{z^2}{N} \right)}$$
$p$ apparterrà a questo intervallo.

## CONFRONTO DI SCHEMI DI LEARNING
E' difficile definire quale di due LEARNING SCHEMERS perfoma meglio.
Anche usando la 10-fold cross validation, comunque non sappiamo se i risultati sono affidabili
Abbiamo bisogno di un SIGNIFICANCE TEST, test che misura quanta confindenza c'è nel dire che non c'è differenza nei due LEARNING SCHEMAS, o che uno è migliore dell'altro.

PAIRED T-TEST

Confronto due learning scheme confrontando le MEAN AVERAGE ACCURACY, presi {X_i}_i=0^k e {Y_i}_i=0^k, output dei due modelli, calcoliamo mx e my: le loro medie.
Come visto prima, per K sufficientemente garande, la differenza tra le medie segue una distribuzione normale, Calcoliamo m_d = mx - my, e la varianza della differenza tra le medie, m_d standardizzato è chiamato t-statistic: t= m_d / sqrt(sigma_d^2 / K) 
Presa z dalla tabella di prima, considerando la confidenza che vogliono Se t <= -z v t >= z: la differenza è significativa tra i due learning scheme, altrimenti non è significativa.

## COME CALCOLARE LE PERFORMANCE DI UN CLASSIFICATORE
Considreiamo p1, ..., pK le probabilità generate su una classificazione rispetto alle K classi

*QUADRATIC LOSS* Considera tutte le probabilità stimate.
$$\sum_{j \neq c} p_j^2 + (1 - p_c)^2 \in [0, 2]$$

- $\sum_{j \neq c} p_j^2$: Contributo di tutte le predizioni sbagliate.
- $(1 - p_c)^2$: Contributo della predizione corretta.

**INFORMATIONAL LOSS**: Si basa solo sulla stima di probabilità della classe corretta.
$$-\log_2(p_c) \in [0, +\infty]$$
$p_c$: Probabilità predetta sulla classe corretta.

## Criteri di valutazione (Classificazione)

### Confusion Matrix

|                   | Classe Predetta: Vero | Classe Predetta: Falso |
|-------------------|:--------------------:|:---------------------:|
| **Classe Effettiva: Vero**  | <span style="color:green">True Positive (TP)</span>  | <span style="color:red">False Negative (FN)</span>  |
| **Classe Effettiva: Falso** | <span style="color:red">False Positive (FP)</span>   | <span style="color:green">True Negative (TN)</span>  |

---

### Valori importanti

- **True Positive Rate (TPR):** $TPR = \frac{TP}{TP + FN}$
- **False Negative Rate (FNR):** $FNR = \frac{FN}{TP + FN}$
- **True Negative Rate (TNR):** $TNR = \frac{TN}{TN + FP}$
- **False Positive Rate (FPR):** $FPR = \frac{FP}{TN + FP}$
- **Overall Success Rate:** $SuccR = \frac{TP + TN}{TP + TN + FP + FN}$
- **Error Rate:** $1 - SuccR$

---

### Metriche principali

- **Precision:**
  $$
  	ext{Precision} = \frac{TP}{TP + FP}
  $$
  Elementi predetti come positivi, tra tutti quelli effettivamente positivi.

- **Recall:**
  $$
  	ext{Recall} = \frac{TP}{TP + FN}
  $$
  Elementi predetti come positivi, tra tutte le predizioni.

- **$F_1$-measure:**
  $$
  F_1 = \frac{2 \cdot \text{Recall} \cdot \text{Precision}}{\text{Recall} + \text{Precision}}
  $$
  Se alto, Precision e Recall sono alti.

- **$F_\beta$-measure:**
  $$
  F_\beta = \frac{(1 + \beta^2) \cdot \text{Recall} \cdot \text{Precision}}{\text{Recall} + \beta^2 \cdot \text{Precision}}
  $$
  Generalizza $F_1$-measure.

---

# MATRICE DI COSTO

**es.**

| CLASSE EFFETTIVA \ CLASSE PREDETTA | VERO | FALSO |
| :--- | :---: | :---: |
| **FALSO** | 0 | 1 |
| **VERO** | 2 | 0 |

> *Nota:* dò un costo più alto alle FP rispetto alle FN. SE CORRETTA IL COSTO È 0.

Ciò ci porta a prendere le predizioni che minimizzano il costo.
In realtà raramente si ha idea di quali siano i costi da assegnare.
Per stimarli usiamo i:

## LIFT CHART
Ho lo scopo di definire un sottoinsieme del test-set con la proporzione di istanze positive maggiore possibile.

![alt text](image.png)
*   **Asse Y:** Number of respondents (0 to 1000)
*   **Asse X:** Sample size (0% to 100%)
*   *Annotazioni sul grafico:*
    *   Punto (100%, 1000) $\leftarrow$ **ideale**
    *   Punto sulla curva $\leftarrow$ **giusto TRADE-OFF**

## ROC CURVE
...

---

# VALUTAZIONE SU PREDIZIONI NUMERICHE
*(es. REGRESSION)*

In questo caso gli errori non sono PRESENTI o ASSENTI, ma hanno una misura.
Considerando $\{a_i\}_{i=0}^n$ come **GROUND TRUTH** e $\{p_i\}_{i=0}^n$ come **VALORI PREDETTI**, misuriamo gli errori con:

> *Nota:* $\bar{a}$ e $p$ del training set — PARLIAMO DI LOSS

## ERRORI ASSOLUTI
$\leftarrow$ che non considerano se sono state fatte 2 o 2000 predizioni

### MEAN-SQUARED ERROR or MEAN-SQUARED LOSS
$$\frac{(p_1 - a_1)^2 + \dots + (p_n - a_n)^2}{n}$$

### ROOT MEAN-SQUARED ERROR
$$\sqrt{\frac{(p_1 - a_1)^2 + \dots + (p_n - a_n)^2}{n}}$$

### MEAN ABSOLUTE ERROR
$$\frac{|p_1 - a_1| + \dots + |p_n - a_n|}{n}$$

---

## ERRORI RELATIVI
$\bar{a}$ è il valore medio sui dati di train

### RELATIVE SQUARED ERROR
$$\frac{(p_1 - a_1)^2 + \dots + (p_n - a_n)^2}{(a_1 - \bar{a})^2 + \dots + (a_n - \bar{a})^2}$$

### ROOT RELATIVE SQUARED ERROR
$$\sqrt{\frac{(p_1 - a_1)^2 + \dots + (p_n - a_n)^2}{(a_1 - \bar{a})^2 + \dots + (a_n - \bar{a})^2}}$$

### RELATIVE ABSOLUTE ERROR
$$\frac{|p_1 - a_1| + \dots + |p_n - a_n|}{|a_1 - \bar{a}| + \dots + |a_n - \bar{a}|}$$

# 3. REGRESSION
(QUI VEDIAMO LA LEAST SQUARE REGRESSION)

I Task che consistono nel predirre un output continuo, anche se limitato

Si tratta di SUPERVISED LEARNING, per cui consideriamo un set di input X^(i) di cui conosciamo gli f^(i) output reali detti TRAINING EXAMPLES

Vogliono approssimare una funzione che rappresenti la relazione tra x e t,  (x è un vettore (features) t scalare)
Per farlo ci serve una LOSS che ci dia quanto bene il modello approssima i dati di training, e un OPTIMIZER che ci dica come modificare i parametri del modello per minimizzare la loss.
t(x) = f(x) + ε, ε è un errore casuale con media 0 e varianza σ^2

la funzione f(x) sarà del tipo f(x) = w_0 + w_1 x dove w_0 è bias e w_1 è il peso associato alla feature x, e w = (w_0, w_1) è il vettore dei parametri da stimare.
Un esempio di loss è la SUM OF SQUARED  VERTICAL ERROR:

$$l(w) = \sum_{n=1}^{N} [t^{(n)} - (w_0 + w_1 x^{(n)})]^2$$

cioè la somma di tutte le linee verdi al quadrato del grafico, che rappresentano la distanza verticale tra i punti di training e la retta di regressione.
![alt text](image-1.png)

Noi cerchiamo w che minimizza questa loss, e per farlo usiamo un OPTIMIZER, ad esempio il GRADIENT DESCENT, che ci dice di modificare w in direzione opposta al gradiente della loss:
## GRADIENT DESCENT METHOD
$$w \leftarrow w - \lambda \frac{\partial l}{\partial w}$$
Lambda è il learning rate, deve esse basso per convergere piano piano

Possiamo applicare questo metodo in due modi:
1. BATCH UPDATES
Somma tutti gli updates su tutti gli elementi di train e poi cambia il valore di w
$$
w \leftarrow w + 2\lambda \sum_{n=1}^{N} (t^{(n)} - y(x^{(n)}))x^{(n)}
$$
Comuptazionalemente costoso, ma generalizza meglio

2. STOCHASTIC UPDATES
Aggiorna w ogni volta che vede un elemento di train

$$
	ext{for } i \in \text{range}(N): \quad w \leftarrow w + 2\lambda (t^{(i)} - y(x^{(i)}))x^{(i)}
$$

MODELLO LINEARE CON FUNZIONE POLINOMIALE
$$
y(x, w) = w_0 + \sum_{j=1}^{M} w_j x^j
$$
$\leftarrow$ funzione non lineare su $X$ ma lineare su $w$ $\Rightarrow$ modello lineare.

## GENERALIZATION
da applicare per evitare l'overfitting, nel caso di primacon M=9 il modello non generalizza bene, mentre con M=2 generalizza meglio, anche se non approssima bene i dati di train, ma è più robusto su dati nuovi.

Quando un modello OVERFIT, i suoi pesi tendono ad essere MOLTO GRANDI, quindi per evitarlo, dobbiamo incoraggiare il modello a usare pesi piccoli,
 ciò è chiamato
## REGULARIZATION
Valore che cresce al crescere di dimensione dei pesi, da aggiungere alla loss, in modo che il modello tenda a usare pesi piccoli per ridurre la loss.

$$
l(w) = \sum_{n=1}^{N} [t^{(n)} - (w_0 + w_1 x^{(n)})]^2 + \alpha \|w\|^2
$$

$\sum_{n=1}^{N} [\ldots ]^2$: Somma degli errori quadratici verticali.
$\alpha \|w\|^2$: REGOLARIZZATORE (Regolarizzazione $L_2$ o Ridge).
$\alpha$: è l'iperparametro che decidiamo noi per decidere l'importanza del regularizer.

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