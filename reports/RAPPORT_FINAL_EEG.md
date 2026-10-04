# Détection automatique des crises d'absence à partir de l'EEG

Rapport final — 4 octobre 2026. Les nombres proviennent du notebook exécuté `notebooks/mini_projet_eeg_final.ipynb`.

## Introduction

Les crises d'absence (CE) correspondent à des épisodes annotés dans les enregistrements EEG du projet. Nous avons construit une chaîne de classification des fenêtres, comparé quatre modèles et reconstruit les périodes prédites. Le déséquilibre a été traité comme une variable expérimentale, en conservant un test composé de données réelles.

## Problématique

Peut-on détecter automatiquement les CE dans des enregistrements réservés à l'évaluation, et quel modèle et quelle stratégie d'équilibrage obtiennent le meilleur compromis entre rappel et précision ?

## Méthodologie

Le pipeline a suivi : chargement → audit → segmentation → étiquetage → extraction des caractéristiques → séparation → normalisation/rééchantillonnage de l'apprentissage → entraînement → évaluation → reconstruction → analyse des erreurs.

```mermaid
flowchart LR
 A[EEG et horaires] --> B[Audit et contrôles source]
 B --> C[Fenêtres de 2 s, recouvrement 50 %]
 C --> D[12 caractéristiques et classe]
 D --> E[Séparation par enregistrement]
 E --> F[Apprentissage et validation interne]
 F --> G[DT, RF, KNN, SVM]
 G --> H[Test naturel réservé]
 H --> I[Intervalles et erreurs]
```

Nous avons retenu le canal commun EEGF7 avant l'entraînement. La fréquence mesurée est de 256 Hz sur les tracés retenus : chaque bloc original contient 256 échantillons et les lignes temporelles sources sont espacées d'une seconde. Les exports ont été comparés aux cellules numériques des fichiers MATLAB, avec égalité des échantillons après conversion float32, des noms ordonnés et des temps. Ce contrôle technique ne constitue pas une validation clinique.

Les horaires Excel ont été convertis par `(DebCE - DebutTrace) × 86400` et `(FinCE - DebutTrace) × 86400`. Aucune durée de crise n'a été inventée. Le masque suit l'intervalle fermé [début, fin] imprimé dans le PDF ; la différence avec la convention ouverte à gauche de la référence locale est explicitée dans le notebook.

Chaque fenêtre comporte 512 échantillons et le pas est de 256. Elle est positive si au moins 50 % de ses échantillons sont en crise. Les huit caractéristiques temporelles sont moyenne, écart-type, variance, minimum, maximum, amplitude, RMS et énergie (somme des carrés). Les quatre puissances FFT unilatérales couvrent Delta [0,5;4[, Theta [4;8[, Alpha [8;13[ et Beta [13;30[ Hz.

Nous avons réservé 4925 fenêtres (21,01 %) de 4 enregistrements pour le test et utilisé 18519 fenêtres de 12 enregistrements pour l'apprentissage. Une sélection déterministe de partitions groupées a approché le ratio 80/20 et les proportions des classes, sans comparer les scores des modèles. Aucun enregistrement ne figure dans les deux ensembles. Trois plis de validation stratifiée groupée, uniquement dans l'apprentissage, ont servi à sélectionner le meilleur F1 moyen ; le rappel puis la précision départagent les égalités.

Les quatre modèles ont été entraînés avec les mêmes caractéristiques et partitions pour quatre stratégies : déséquilibre initial, sous-échantillonnage aléatoire, suréchantillonnage par duplication et SMOTE. Les transformations sont dans des pipelines ajustés sur chaque pli d'apprentissage. SMOTE crée des caractéristiques interpolées, pas des EEG ou des crises réelles. Les hyperparamètres fixés et la graine 42 sont conservés dans le notebook et le module du projet.

## Données

Le classeur mis à jour comporte 21 lignes d'identifiants et 19 identifiants uniques. Nous avons exclu les deux doublons conflictuels `230406B-A` et `230515B-G`, ainsi que `210504B-C`, dont la première crise dépasse la durée du tracé. Trois exports sans ligne dans le classeur mis à jour n'ont pas été assimilés à des enregistrements normaux. L'écart de `NombreCE` pour `210914B-A` a été documenté et les onze paires explicites ont été conservées.

La cohorte finale comprend 16 enregistrements et 65 CE, de durée moyenne 13,46 s, médiane 11 s, minimale 3 s et maximale 49 s. L'intervalle libre moyen entre deux crises successives du même tracé est de 185,86 s. La segmentation fournit 23444 fenêtres, dont 940 positives (4,01 %). La proportion d'échantillons en crise est 3,73 % ; elle est distincte de celle des fenêtres.

Figure 1 — EEG et annotations.

![Figure 1 — EEG et annotations](../artifacts/final/figures/signal_annote.png)

Figure 2 — Durées des crises.

![Figure 2 — Durées des crises](../artifacts/final/figures/durees_annotations.png)

## Résultats

Tableau 1 — Quatre modèles sans équilibrage, sur le test réservé.

| Stratégie | Modèle | Accuracy | Précision | Rappel | F1 | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sans équilibrage | Arbre de décision | 0.979 | 0.728 | 0.810 | 0.767 | 171 | 64 | 40 | 4650 |
| Sans équilibrage | Forêt aléatoire | 0.986 | 0.843 | 0.815 | 0.829 | 172 | 32 | 39 | 4682 |
| Sans équilibrage | KNN | 0.959 | 0.551 | 0.232 | 0.327 | 49 | 40 | 162 | 4674 |
| Sans équilibrage | SVM | 0.969 | 0.889 | 0.303 | 0.452 | 64 | 8 | 147 | 4706 |

Tableau 2 — Comparaison complète des 16 configurations, sur le même test.

| Stratégie | Modèle | Accuracy | Précision | Rappel | F1 | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sans équilibrage | Arbre de décision | 0.979 | 0.728 | 0.810 | 0.767 | 171 | 64 | 40 | 4650 |
| Sans équilibrage | Forêt aléatoire | 0.986 | 0.843 | 0.815 | 0.829 | 172 | 32 | 39 | 4682 |
| Sans équilibrage | KNN | 0.959 | 0.551 | 0.232 | 0.327 | 49 | 40 | 162 | 4674 |
| Sans équilibrage | SVM | 0.969 | 0.889 | 0.303 | 0.452 | 64 | 8 | 147 | 4706 |
| Sous-échantillonnage | Arbre de décision | 0.895 | 0.268 | 0.829 | 0.405 | 175 | 479 | 36 | 4235 |
| Sous-échantillonnage | Forêt aléatoire | 0.915 | 0.321 | 0.891 | 0.472 | 188 | 398 | 23 | 4316 |
| Sous-échantillonnage | KNN | 0.625 | 0.093 | 0.882 | 0.168 | 186 | 1821 | 25 | 2893 |
| Sous-échantillonnage | SVM | 0.755 | 0.135 | 0.877 | 0.234 | 185 | 1183 | 26 | 3531 |
| Suréchantillonnage | Arbre de décision | 0.942 | 0.416 | 0.863 | 0.562 | 182 | 255 | 29 | 4459 |
| Suréchantillonnage | Forêt aléatoire | 0.982 | 0.791 | 0.791 | 0.791 | 167 | 44 | 44 | 4670 |
| Suréchantillonnage | KNN | 0.895 | 0.216 | 0.550 | 0.310 | 116 | 422 | 95 | 4292 |
| Suréchantillonnage | SVM | 0.681 | 0.110 | 0.905 | 0.196 | 191 | 1549 | 20 | 3165 |
| SMOTE | Arbre de décision | 0.924 | 0.342 | 0.848 | 0.488 | 179 | 344 | 32 | 4370 |
| SMOTE | Forêt aléatoire | 0.966 | 0.568 | 0.834 | 0.676 | 176 | 134 | 35 | 4580 |
| SMOTE | KNN | 0.869 | 0.191 | 0.640 | 0.294 | 135 | 571 | 76 | 4143 |
| SMOTE | SVM | 0.687 | 0.111 | 0.900 | 0.198 | 190 | 1519 | 21 | 3195 |

Figure 3 — Caractéristiques par classe.

![Figure 3 — Caractéristiques par classe](../artifacts/final/figures/caracteristiques.png)

Figure 4 — Corrélations.

![Figure 4 — Corrélations](../artifacts/final/figures/correlations.png)

Figure 5 — Comparaison de la précision, du rappel et du F1.

![Figure 5 — Comparaison de la précision, du rappel et du F1](../artifacts/final/figures/comparaison_modeles.png)

La variance est le carré de l'écart-type ; l'énergie vaut 512 fois le carré du RMS. Ces identités expliquent une redondance structurelle, sans imposer une corrélation de Pearson exactement égale à un. Les moyennes et écarts par classe ainsi que les corrélations effectivement mesurées sont affichés en Q10–Q11 ; les douze caractéristiques de référence ont été conservées pour isoler l'effet de l'équilibrage.

## Discussion et analyse des erreurs

Le choix fixé en validation interne est **Arbre de décision avec smote**. Sur le test réservé, ce modèle atteint une précision de 0,342, un rappel de 0,848 et un F1 de 0,488, avec TP=179, FP=344, FN=32 et TN=4370 au niveau des fenêtres. Le choix n'a pas été modifié en fonction du classement sur le test.

Le plus grand F1 observé sur ce test est 0,829, obtenu par Forêt aléatoire / Sans équilibrage. Ce résultat descriptif ne remplace pas la sélection antérieure sur les plis internes. L'écart entre ces classements souligne la variabilité de la petite cohorte.

La baseline « toujours hors crise » obtient 0,957 d'accuracy, avec rappel et F1 nuls. Une accuracy élevée peut donc masquer une absence de détection. Le rappel rend visibles les crises manquées ; la précision et le F1 complètent l'analyse pour éviter de favoriser un modèle qui déclencherait partout. La comparaison avant/après n'affirme pas que l'équilibrage améliore toutes les métriques : elle expose les résultats de chaque stratégie.

Les fenêtres positives successives ont été reconstruites par leur support temporel. Un appariement un à un avec seuil IoU ≥ 0,3, fixé avant l'évaluation, donne 17 événements correctement appariés, 185 événements prédits non appariés et 0 crises non appariées. Cette définition est plus exigeante qu'une simple intersection. Les tables des bornes réelles/prédites et les figures par tracé sont dans le notebook ; les événements ne doivent pas être confondus avec les fenêtres.

## Conclusion

Le projet fournit une comparaison exécutée de quatre familles de modèles et de quatre stratégies sur des annotations horaires corrigées, avec une séparation stricte par enregistrement. Le modèle retenu est celui choisi en validation interne. Ses scores et ses erreurs temporelles, présentés ci-dessus, constituent le résultat de cette expérience ; ils ne démontrent pas une performance clinique sur de nouveaux patients.

## Limites et perspectives

La cohorte de 16 tracés demeure petite et le test ne comprend que 4 enregistrements. Le choix conservateur d'exclure les annotations conflictuelles réduit l'effectif et peut modifier la population étudiée. La correspondance des exports aux valeurs sources a été contrôlée, mais aucune validation clinique indépendante des annotations ou des unités d'amplitude n'a été réalisée. Les fenêtres se recouvrent ; les plis groupés empêchent leur partage entre entraînement et évaluation, sans faire des fenêtres d'un même tracé des observations indépendantes. Aucun identifiant patient fiable ne permet de garantir une séparation par patient. Le canal unique, le montage, les artefacts et la généralisation à d'autres acquisitions restent des limites. Les interpolations SMOTE ne créent aucune nouvelle diversité de patients et peuvent rompre les identités entre caractéristiques. Une validation externe et la résolution des annotations exclues sont des prolongements prioritaires.

## Bibliographie

1. Énoncé local, `mini-prjet-data-mining.pdf`, parties I–XII et questions 1–14.
2. Message de l'enseignant fourni par l'utilisateur : interprétation des débuts et fins comme horaires et exemple `190304A-E`.
3. Classeur source, `Annotations MAJ.xlsx`, feuille `Feuil1`.
4. imbalanced-learn, [Pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html) et [erreurs fréquentes](https://imbalanced-learn.org/stable/common_pitfalls.html).
5. scikit-learn, [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
6. Chawla et al., [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813), 2002.