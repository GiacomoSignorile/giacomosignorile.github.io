---
layout: math
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

![Lift Chart](/assets/img/image.png)
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
![Regression Error](/assets/img/image-1.png)

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

# 4. CLASSIFICATION

Consiste nell'assegnare ad un input vector uno di un set finito di LABELS 
Tipi di clasisficazione:
- Binary classification: due classi, output 0 o 1 yes or no
- Multiclass classification: più di due classi, output in {1, 2, ..., C}
- Multilabel classification: più classi, ma non mutuamente esclusive, output in {0, 1}^C
Approcci alla classificazione:
1. APPROCCIO DISCRIMINATIVO: stima direttamente P(y|x) o una funzione di decisione f(x) che mappa x a y
2. APPROCCIO GENERATIVO: stima P(x|y) e P(y), poi usa Bayes per calcolare P(y|x)

## CLASSIFICAZIONE BINARIA(LINEARE)
$$y(x) = f(w^T x + w_0)$$
- $w^T$: pesi
- $x$: input
- $w_0$: bias
- $f$: ACTIVATION FUNCTION (non lineare)
Nota: Al contrario dei modelli di regressione, non è lineare su $W$ a causa dell'ACTIVATION FUNCTION.

se X è espresso da 1 feature, il classificatore sarà un THRESHOLD
2 feature, una linea
3 feature, un piano
/> 4 un iperpiano
Una volta definito l'activation function, ci serve trovare w e w_0, che distinguono bene le due classi, qundi ci serve una
LOSS FUNCTION
Cioè una funzione che dice l'errore commesso prevedendo y qyando la classe corretta è t, esempi: 
1. ZERO/ONE LOSS $$L_{0-1}(y(x), t) = \begin{cases} 0 & \text{if } y(x) = t \\ 1 & \text{if } y(x) \neq t \end{cases}$$Difficile da minimizzare.
2. ASYMMETRIC BINARY LOSS $$L_{ABL}(y(x), t) = \begin{cases} \alpha & \text{if } y(x)=1 \land t=0 \\ \beta & \text{if } y(x)=0 \land t=1 \\ 0 & \text{if } y(x)=t \end{cases}$$Dà valori diversi a seconda dell'errore commesso.
3. SQUARED QUADRATIC LOSS $$L_{\text{squared}}(y(x), t) = (t - y(x))^2$$4. ABSOLUTE ERROR $$L_{\text{absolute}}(y(x), t) = |t - y(x)|$$  

Il problema si dice LINEARLY SEPARABLE se riusciamo a separare le classi con un classificatore lineare.
Se non riusciamo a separare le classi le cause sono:
- Non ci basta un classificatore lineare, ci serve un classificatore non lineare
- L'input è troppo rumoroso, quindi non è possibile separare le classi, in questo caso ci serve un classificatore che sia robusto al rumore.
- L'output della GROUND TRUTH è errato, quindi non è possibile separare le classi, in questo caso ci serve un classificatore che sia robusto al rumore.
- Ci servono più feature in input

## METRICHE DI CLASSIFICAZIONE
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
Similar to Regression, anche per la classificazione è importante evitare l'overfitting, quindi dobbiamo regolarizzare il modello, ad esempio con la REGOLARIZZAZIONE L2, che aggiunge alla loss un termine che penalizza i pesi grandi:

# 5. NON-PARAMETRIC MODELS: KNN
Modelli la cui dimensione dipende dalla quantità di dati, più dati abbiamo, più parametri ha il modello. KNN è un esempio di modello non parametrico, che memorizza tutto il training set e classifica un nuovo input x guardando i K punti più vicini a x nel training set e prendendo la classe più frequente tra questi K punti.
Si chiamano LAZY LEARNERS perché non fanno nulla durante la fase di training, ma solo durante la fase di test, quando devono classificare un nuovo input.

## 1-NN
Trova la classe di X cercando l'esempio più vicino a X nel training set e prendendo la sua classe. Non è molto robusto al rumore, perché se l'esempio più vicino è un outlier, la classificazione sarà errata.
ad esempio con la DISTANZA EUCLIDEA, la distanza tra due punti x e y è data da:
$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

x* = argmin_{x_i} d(x, x_i)

x* è l'esempio più vicino a x, e la sua classe è la classe predetta per x.
deduciamo che y = t*, dove t* è la classe dell'esempio più vicino x*.
VORONOI DIAGRAM
Ci permettte di visualizzare la decision boundary del classificatore 1-NN, che è data dai punti che sono equidistanti da due o più esempi del training set. Questi punti formano una linea (o un iperpiano) che separa le classi.

# 9. NEURAL NETWORKS

Questa sezione raccoglie appunti e spiegazioni sulle reti neurali artificiali, con attenzione a chiarezza, esempi pratici e formule matematiche. Ideale per studenti e appassionati di ML.

---

## Cos'è una Rete Neurale?

Una rete neurale artificiale (ANN) è un modello di machine learning ispirato al cervello umano, composto da unità chiamate **neuroni artificiali**. Ogni neurone riceve input, applica pesi, una funzione di attivazione e un bias:

- **Input**: segnali provenienti da altri neuroni o dati grezzi
- **Pesi**: valori appresi durante il training
- **Funzione di attivazione**: introduce non linearità
- **Bias**: valore che regola la soglia di attivazione

![Neurone artificiale](/assets/img/Pasted%20image%2020260317220324.png)

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

![MLP](/assets/img/Pasted%20image%2020260317221606.png)

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
A **Random Forest** is a highly effective **ensemble learning algorithm** that builds upon the concepts of bagging (bootstrap aggregating) and randomization. 

**How it Works**
The algorithm constructs a "forest" by building a large ensemble of randomized decision trees. To create this ensemble, it explicitly tries to decorrelate the individual trees (the base learners) by introducing randomness in two key ways:
*   **Data Randomization (Bagging):** Each decision tree is trained on a randomly chosen subset of the available data cases. A randomized decision tree is built in each iteration of the bagging algorithm.
*   **Feature Randomization (Random Subspaces):** Instead of evaluating all possible features to find the best split at each node, the algorithm learns trees based on a **randomly chosen subset of input variables**. This approach is directly related to the random subspaces method for constructing an ensemble of classifiers.
* Performance 
By combining these randomization techniques, Random Forests generate diverse models whose noise tends to cancel out, resulting in **very good predictive accuracy** and excellent overall predictors. However, because it builds many complex trees, one minor drawback is that the algorithm can be fairly slow to train compared to simpler models.
---

## Decision Tree

- Modello predittivo che suddivide ricorsivamente i dati in base alle feature.
- Ogni nodo interno rappresenta una decisione su una feature; ogni foglia una classe o valore.
- Algoritmi comuni: CART, ID3, C4.5.
- Criteri di split: **Gini**, entropia, varianza.
A **decision tree** is a machine learning representation created through a "divide-and-conquer" approach to learning from a set of independent instances. It maps an input to a predicted output by applying a sequence of simple tests, making them easy to interpret as they correspond to a sequence of decisions applied to individual input variables.

**Structure and Classification Process**
*   **Nodes and Tests:** Each internal node in a decision tree involves testing a particular attribute. For a nominal attribute, the node typically branches for each possible value. For a numeric attribute, the test usually determines whether the value is greater or less than a predetermined constant, resulting in a two-way split, though three-way or multi-way splits are also possible. 
*   **Leaves:** The terminal nodes, or leaf nodes, assign a classification that applies to all instances reaching that leaf, or occasionally a probability distribution over possible classifications. If the tree is designed to predict numeric quantities rather than categories, it is called a **regression tree** (where leaves contain average numeric values) or a **model tree** (where leaves contain linear regression models).
*   **Routing:** To classify an unknown instance, it is routed down the tree from the root, following the branches that correspond to the instance's attribute values, until a leaf is reached and the classification is assigned.

**Constructing the Tree**
Decision trees are usually built using a top-down, recursive divide-and-conquer algorithm:
1.  **Select an Attribute:** An attribute is chosen for the root node, and branches are made for its possible values. To determine the best attribute, the algorithm evaluates which choice will produce the "purest" daughter nodes. This is typically calculated using an entropy-based measure called **information gain**. Because information gain naturally biases the model toward highly branching attributes (like unique ID codes), a modified metric called the **gain ratio** is often used to compensate by taking the size and number of branches into account.
2.  **Split the Data:** The training examples are divided into subsets according to their values for the chosen attribute.
3.  **Recurse:** The process repeats recursively for each branch, using only the subset of instances that actually reach it. The process terminates when all instances at a node have the exact same classification (a completely pure node), or when the data cannot be split any further.

**Handling Missing Values**
If an instance has an unknown value for a tested attribute, the algorithm can notionally split the instance into pieces. Each piece is sent down a different branch and is assigned a numeric weight proportional to the number of training instances that took that specific branch.

**Pruning to Prevent Overfitting**
A fully expanded decision tree will often overfit the training data, meaning it follows the data too slavishly and learns noise, resulting in poor generalization to independent test sets. To resolve this, trees undergo a simplification process called **pruning**. While prepruning attempts to stop the tree from growing early, most systems rely on postpruning. Common postpruning operations include:
*   **Subtree replacement:** Selecting a subtree and replacing it entirely with a single leaf node.
*   **Subtree raising:** Replacing an internal node with an entire subtree that was located below it.
---

## Logistic Regression

- Modello lineare per la classificazione binaria.
- Calcola la probabilità che un input appartenga alla classe positiva tramite la funzione sigmoid:

$$
P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}
$$

- Loss: **log-loss** (cross-entropy)
- Può essere regolarizzata (L1, L2)

**Logistic regression** is a highly popular statistical and machine learning technique that, despite its name, is specifically designed for **classification rather than regression** tasks. It is predominantly used for binary classification, where the goal is to categorize an input into one of two distinct classes (e.g., 0 or 1) by estimating the probability of class membership.

**The Logit Transform and Sigmoid Function**
Standard linear regression is poorly suited for predicting probabilities because a linear function can easily output values below 0 or above 1. Logistic regression solves this issue by building a linear model based on a transformed target variable. 
*   It uses the **logit transformation**, which calculates the log-odds ratio: $\log(p / (1 - p))$,. The resulting values can span from negative infinity to positive infinity, allowing them to be approximated accurately by a linear combination of the input attributes.
*   Equivalently, the linear output is passed through a non-linear **logistic sigmoid function** (often called a "squashing function"), mathematically defined as $\sigma(x) = 1 / (1 + \exp(-x))$,. This transforms the raw linear score into a value strictly confined to the (0, 1) interval, which can be seamlessly interpreted as a valid **conditional probability**,.

**Linear Decision Boundary**
In a two-class logistic regression model, the decision boundary lies exactly where the predicted probability is 0.5. This 0.5 threshold occurs exactly when the underlying linear sum of the attributes equals zero ($w_0 + w_1a_1 + ... + w_ka_k = 0$),. Because this relies on a linear equality, the resulting decision surface separating the classes is a **flat, linear hyperplane** in the instance space,.

**Training and Optimization**
Unlike simple linear regression, logistic regression **does not have a closed-form mathematical solution** to instantly calculate the optimal parameter weights,. 
*   **Loss Function:** Instead of minimizing the squared error, the algorithm determines the optimal weights by **maximizing the log-likelihood** of the training data, which is equivalent to minimizing the negative log-likelihood or **cross-entropy error function**,,. 
*   **Iterative Algorithms:** Fortunately, the cross-entropy error function is convex and possesses a unique global minimum,. Thus, the optimal weights can be found reliably using iterative numerical optimization procedures. Common techniques include **gradient descent**, or second-order methods like **Iteratively Reweighted Least Squares (IRLS)**, which is based on the Newton-Raphson scheme,.

**Preventing Overfitting**
Logistic regression can exhibit severe overfitting if the training data is perfectly linearly separable. Under standard maximum likelihood estimation, the algorithm will attempt to drive the weights to infinity to create a perfectly steep "step function," resulting in a brittle model,. To mitigate this, practitioners usually employ **regularization (such as an $L_2$ weight decay penalty)** or adopt a Bayesian approach (MAP estimation) that applies a prior distribution to encourage smaller, more stable coefficients.

* Multiclass Extensions
Although fundamentally a binary classifier, logistic regression can be generalized to handle multiple classes. This can be achieved by training separate classifiers for each pair of classes (pairwise classification) and combining their votes, or by using a mathematical generalization known as the **softmax function** (or normalized exponential) to yield a unified multinomial logistic regression model,.
---

## Che problema risolve Gini

L'**indice di Gini** misura la "purezza" di un nodo in un albero decisionale.
- Serve a scegliere lo split migliore: più basso è il Gini, più puro è il nodo.
- Formula:

$$
Gini = 1 - \sum_{i=1}^C p_i^2
$$

dove $p_i$ è la proporzione di elementi della classe $i$ nel nodo.

The **Gini index** (often called Gini impurity) is a metric used by decision tree learning algorithms, such as CART, to evaluate and select the best attribute for splitting data. 

Specifically, the Gini index resolves the limitations of using a simple misclassification rate when growing decision trees by providing a superior measure of **node impurity**. It resolves these limitations in two primary ways:

*   **Sensitivity to Node Purity:** The Gini index is much more sensitive to changes in class probabilities than the raw misclassification rate. It actively encourages the formation of pure regions—nodes where a high proportion of the data points belong to a single class. For example, if two different dataset splits yield the exact same overall misclassification rate, but one split successfully creates a perfectly "pure" node (containing only one class), the Gini index will correctly favor the split that isolates the pure node.
*   **Differentiability:** Unlike the misclassification rate, the Gini index is a differentiable function, which makes it far better suited for gradient-based optimization methods used to evaluate splits during tree construction.

**How it Works**
Mathematically, the Gini index is calculated as $1 - \sum_c \hat{\pi}_c^2$ (or equivalently $\sum_{c=1}^C \hat{\pi}_c(1-\hat{\pi}_c)$), where $\hat{\pi}_c$ represents the proportion of instances at that node that belong to class $c$. 

*   This formula effectively calculates the **expected error rate** at a given node: $\hat{\pi}_c$ is the probability that a random entry in the leaf belongs to class $c$, and $(1-\hat{\pi}_c)$ is the probability that it would be misclassified.
*   The index evaluates to $0$ (vanishes) when a node is perfectly pure (meaning the probability of a class is either $0$ or $1$), indicating zero impurity.
*   It reaches its maximum value (e.g., $0.5$ in a two-class problem) when the classes are perfectly evenly mixed, indicating maximum impurity.
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
Regularization refers to any modification made to a machine learning algorithm intended to reduce its generalization error (its performance on unseen data) without necessarily reducing its training error,. Its primary goal is to prevent **overfitting**—a scenario where a high-capacity model memorizes the noise and specific details of the training data but fails to generalize to new inputs—by controlling the model's effective complexity,,,. 

**How it Works**
Regularization generally operates by trading a slight increase in model bias for a significant reduction in variance. Rather than strictly limiting the model's mathematical capacity (such as restricting a neural network to fewer layers), regularization incorporates a preference for certain types of solutions, such as simpler functions with smaller parameter weights,. 

From a probabilistic perspective, many regularization techniques are equivalent to Maximum A Posteriori (MAP) Bayesian inference. In this view, the regularization penalty corresponds to a prior probability distribution over the model's parameters, restricting the learning algorithm based on prior beliefs before any data is even observed,,.

**Common Regularization Strategies**
There are many forms of regularization available to practitioners, including:

*   **Parameter Norm Penalties:** This involves adding a mathematical penalty term to the objective or cost function to discourage the model's coefficients from reaching large values,.
    *   **L2 Regularization (Weight Decay or Ridge Regression):** Adds a penalty based on the squared magnitude of the weights. This shrinks weights closer to the origin and is mathematically equivalent to assuming a Gaussian prior on the parameters,,,.
    *   **L1 Regularization (Lasso):** Adds a penalty based on the sum of the absolute values of the weights. Unlike L2, L1 regularization naturally induces **sparse** solutions where many parameters are driven exactly to zero, making it an effective mechanism for automatic feature selection,,,. This corresponds to applying a Laplace prior,.
*   **Early Stopping:** During iterative training, the model's error on an independent validation set is continuously monitored. Training is halted when the validation error stops improving and begins to rise,. This unobtrusive method restricts the optimization to a smaller volume of the parameter space and has been shown to behave mathematically similarly to L2 regularization.
*   **Dataset Augmentation:** Expanding the training set by generating synthetic data through transformations (e.g., randomly translating, rotating, or flipping images). This forces the model to generalize to variations it hasn't explicitly seen,.
*   **Noise Injection:** Adding noise to the input data, the hidden units, or the model weights during training. It can also be applied to the output targets (a technique known as label smoothing) to prevent the model from pursuing extremely confident, hard probabilities,.
*   **Dropout:** A technique specifically for neural networks where neurons are randomly turned off (set to zero) during training. This prevents neurons from co-adapting too strongly to specific training cases and effectively forces the network to behave like an ensemble of exponentially many smaller, robust subnetworks,.
*   **Parameter Sharing and Tying:** Forcing sets of parameters to be equal across different parts of a model. This is famously used in Convolutional Neural Networks (CNNs) to dramatically lower the number of unique parameters and reduce the memory footprint without sacrificing expressive power.
*   **Adversarial Training:** Training the model on intentionally constructed, perturbed "adversarial" examples. This discourages the network from being overly sensitive to tiny, linear changes in the input, encouraging the model to be locally constant in the neighborhood of the training data,.
*   **Representational Sparsity:** Placing a penalty on the *activations* of the units in a neural network rather than its weight parameters, encouraging the internal representations of the data to be sparse (containing mostly zeros),.

---

## SVM (Support Vector Machine)

- Modello per classificazione che cerca l'**iperpiano** che massimizza il margine tra le classi.
- Può usare kernel per separare dati non linearmente separabili.
- Loss: **hinge loss**

A **Support Vector Machine (SVM)** is a highly influential supervised learning algorithm primarily used for classification, though it can also be adapted for regression tasks. Despite its name, it is an algorithm rather than a physical machine.

Its defining feature is its ability to **use simple linear models to implement complex, nonlinear class boundaries**. It achieves this by mathematically transforming the input data into a new, often higher-dimensional space where a straight line (or flat hyperplane) can separate the different classes.

Here are the core mechanisms that make the SVM work:

*   **The Maximum Margin Hyperplane:** When data is linearly separable, there might be multiple lines that can divide the classes. The SVM algorithm specifically searches for the maximum margin hyperplane—the boundary that gives the greatest possible separation (or margin) between the classes, coming no closer to either group than it absolutely has to.
*   **Support Vectors and Sparsity:** The training data points that lie closest to this maximum margin hyperplane are called **support vectors**. These specific points uniquely define the decision boundary; in fact, all other training instances are irrelevant and could be deleted without changing the model's outcome. Because the final model relies on only this small subset of the data, SVMs are known for producing highly **sparse solutions**.
*   **The Kernel Trick:** Transforming data into high-dimensional feature spaces to find boundaries can be incredibly computationally expensive. SVMs bypass this using a mathematical shortcut known as the **kernel trick**. This technique allows the algorithm to implicitly calculate relationships (dot products) in the high-dimensional space while actually performing the computations in the original, low-dimensional space. Common kernel functions include the polynomial kernel, the radial basis function (RBF) or Gaussian kernel, and the sigmoid kernel.
*   **Soft Margins for Overlapping Data:** In most real-world scenarios, classes overlap, meaning a perfect separation would force the model to overfit to noise. SVMs resolve this by relaxing the hard boundary into a **"soft margin"**. By introducing "slack variables," the model allows some training points to fall inside the margin or even be misclassified. A user-specified regularization parameter (usually denoted as $C$) controls the trade-off between minimizing these training errors and keeping the margin as wide as possible.
*   **Support Vector Regression:** When applied to numeric prediction, the algorithm is called Support Vector Regression. Instead of minimizing squared error like standard linear regression, it creates an **$\epsilon$-insensitive "tube"** around the regression function. The model completely ignores any prediction errors that fall within this tube's width ($2\epsilon$), only penalizing points that lie outside of it, which once again results in a sparse model. 

* In practice, training an SVM is a constrained quadratic optimization problem, which guarantees that any local solution found by the algorithm is also a global optimum.
---

## Feedforward Neural Network

- Rete neurale in cui l'informazione fluisce solo in avanti (dagli input agli output).
- Composta da uno o più layer (input, hidden, output).
- Ogni neurone calcola una combinazione pesata degli input, applica una funzione di attivazione e passa il risultato al layer successivo.
- Non ha memoria (a differenza delle RNN).

A **feedforward neural network**, also widely known as a **multilayer perceptron (MLP)**, is the quintessential deep learning model designed to approximate a target mathematical function. 

**Core Characteristics:**
*   **Feedforward Information Flow:** These models are called "feedforward" because information flows in strictly one direction. It starts at the input $x$, passes through intermediate computations, and finally reaches the output $y$. Unlike recurrent neural networks, feedforward networks have no feedback connections that loop the model's outputs back into itself.
*   **Network Composition:** They are called "networks" because they are formed by composing many different, simpler mathematical functions together. This is typically represented as a directed acyclic graph in a chain structure. For instance, a network with three functions would be chained together to form $f(x) = f^{(3)}(f^{(2)}(f^{(1)}(x)))$.
*   **Neural Inspiration:** The term "neural" reflects their loose inspiration from biological neuroscience. The network's layers are vector-valued, and each individual element within a vector can be thought of as an artificial "neuron". Each neuron receives inputs from many other units, computes its own activation value—typically by applying an affine (linear) transformation followed by a fixed, non-linear activation function like a rectified linear unit (ReLU)—and passes the signal forward.

**Layer Architecture:**
*   **Input and Output Layers:** The first function in the chain is the first layer (which receives the raw input), and the final function in the chain forms the **output layer** (which produces the final prediction).
*   **Hidden Layers:** The intermediate functions between the input and output are called **hidden layers**. They are "hidden" because the training data does not explicitly tell the network what the output of these specific layers should be. Instead, the learning algorithm must automatically determine how to use and adapt these hidden units to extract increasingly complex features from the data to best implement the final prediction.

---

## CNN

A **Convolutional Neural Network (CNN)** is a specialized type of feedforward neural network designed for processing data that has a known, grid-like topology, such as 1D time-series data or 2D image data. It distinguishes itself from standard neural networks by using a mathematical operation called **convolution** in place of general matrix multiplication in at least one of its layers.

CNNs are highly effective because they leverage three core architectural ideas:
*   **Sparse Interactions:** In a traditional fully connected neural network, every output unit interacts with every input unit. In a CNN, the network uses filters (or kernels) that are much smaller than the overall input. This means neurons only process a small, localized spatial region (a "receptive field"), which drastically reduces memory requirements and improves computational efficiency.
*   **Parameter Sharing (Tied Weights):** Rather than learning a separate set of parameters for every single location in an input, a CNN uses the exact same kernel across all positions of the input. This allows the network to learn a feature (like an edge or a corner) once and detect it anywhere in the image, significantly reducing the number of unique parameters the model must store.
*   **Equivariant Representations:** Because the same weights are scanned across the entire input, the convolutional layer is inherently equivariant to translation. If an object shifts in the input image, its corresponding feature representation will shift by the exact same amount in the output feature map.

**Typical CNN Architecture**
A standard layer within a CNN generally consists of three sequential stages:
1.  **Convolution Stage:** The layer applies multiple learnable filters in parallel to the input to produce a set of linear activations, often called feature maps.
2.  **Detector Stage:** A nonlinear activation function, commonly a Rectified Linear Unit (ReLU), is applied element-wise to these linear activations.
3.  **Pooling Stage:** The output is modified by a pooling function, which replaces the network's output at a certain location with a summary statistic of the nearby outputs (such as the maximum value in a rectangular neighborhood, known as **max-pooling**, or the average value). Pooling downsamples the spatial resolution and makes the model's representation **approximately invariant to small translations**. This means the network learns to care more about *whether* a feature is present rather than its exact pixel-perfect location.

In a complete CNN, the input is usually subjected to multiple repeating phases of these convolution, nonlinearity, and pooling operations. As the data moves deeper into the network, the receptive fields become larger, allowing the network to combine simple features (like edges) into highly complex, abstract features (like entire object parts). Finally, the resulting spatial feature maps are typically flattened into a vector and fed into a fully connected feedforward network (multilayer perceptron) to make the final classification or prediction.

* Notably, this design is heavily inspired by neuroscientific models of the mammalian primary visual cortex (V1), mapping closely to how biological "simple cells" detect local features and "complex cells" provide invariance to small shifts.
---