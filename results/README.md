Models can be found in code/src/LDA/models
### Full models
1)

Gamma=0.8, window_size=1, sequence_size=3

                precision    recall  f1-score   support

         LOC       0.10      0.12      0.11      1925
        MISC       0.08      0.08      0.08       918
           O       0.91      0.69      0.78     38323
         ORG       0.12      0.45      0.19      2496
         PER       0.18      0.27      0.22      2773
    accuracy                           0.62     46435
    weighted avg   0.77      0.62      0.68    46435

2)

with masking

                precision    recall  f1-score   support
         LOC       0.06      0.15      0.09      1925
        MISC       0.04      0.17      0.06       918
           O       0.92      0.68      0.78     38323
         ORG       0.13      0.28      0.18      2496
         PER       0.20      0.30      0.24      2773

    accuracy                           0.60     46435
    weighted avg   0.78      0.60      0.67     46435

3)

with adversarial noise (without merging)

                precision    recall  f1-score   support
       B-LOC       0.12      0.44      0.19      1668
      B-MISC       0.03      0.03      0.03       702
       B-ORG       0.09      0.13      0.10      1661
       B-PER       0.25      0.20      0.22      1617
       I-LOC       0.01      0.02      0.01       257
      I-MISC       0.00      0.06      0.01       216
       I-ORG       0.06      0.14      0.09       835
       I-PER       0.06      0.02      0.03      1156
           O       0.93      0.69      0.79     38323
    accuracy                           0.60     46435
    weighted avg       0.79      0.60      0.67     46435
    

### Probabilistic only
The following models were trained on only the stored probabilistic values. If no part of the sequence could be found as a context at all, it was not added to the table and ended up as an "O"

1)

Gamma=0.8, window_size=1, sequence_size=4

                precision    recall  f1-score   support
       B-LOC       0.04      0.05      0.05      1668
      B-MISC       0.04      0.05      0.04       702
       B-ORG       0.05      0.67      0.09      1661
       B-PER       0.19      0.12      0.14      1617
       I-LOC       0.00      0.07      0.01       257
      I-MISC       0.01      0.28      0.02       216
       I-ORG       0.01      0.01      0.01       835
       I-PER       0.03      0.01      0.01      1156
           O       1.00      0.21      0.35     38323

    accuracy                           0.21     46435
    weighted avg   0.83      0.21      0.30     46435

2)

sequence_size=3, merged I and B for same category

                precision    recall  f1-score   support

         LOC       0.09      0.32      0.14      1925
        MISC       0.05      0.29      0.08       918
           O       1.00      0.24      0.38     38323
         ORG       0.07      0.65      0.12      2496
         PER       0.32      0.15      0.21      2773
    accuracy                           0.26     46435
    weighted avg   0.85      0.26      0.34     46435
    
3)

Clubbing all entities as type ENT in post-processing, sequence_size=4

                precision    recall  f1-score   support
      ENTITY       0.21      1.00      0.35      8112
           O       1.00      0.21      0.35     38323

    accuracy                           0.35     46435
    weighted avg   0.86      0.35      0.35     46435
    
4)

window_size=2

                precision    recall  f1-score   support
         LOC       0.07      0.23      0.11      1925
        MISC       0.04      0.26      0.07       918
           O       1.00      0.16      0.28     38323
         ORG       0.07      0.69      0.13      2496
         PER       0.14      0.25      0.18      2773

    accuracy                           0.20     46435
    weighted avg   0.84      0.20      0.25     46435

sequence_size=5

                precision    recall  f1-score   support
         LOC       0.04      0.15      0.06      1925
        MISC       0.03      0.22      0.06       918
           O       1.00      0.16      0.28     38323
         ORG       0.07      0.72      0.13      2496
         PER       0.29      0.15      0.20      2773

    accuracy                           0.19     46435
    weighted avg   0.85      0.19      0.25     46435

5)

gamma=0.4

                precision    recall  f1-score   support
         LOC       0.06      0.34      0.10      1925
        MISC       0.04      0.39      0.07       918
           O       1.00      0.15      0.26     38323
         ORG       0.08      0.57      0.14      2496
         PER       0.18      0.08      0.11      2773
    accuracy                           0.18     46435
    weighted avg   0.84      0.18      0.23     46435