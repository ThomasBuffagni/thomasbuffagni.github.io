---
layout: post
title: Le Deep Learning en douceur
date_fr: 7 juin 2020
---

Le Deep Learning est un domaine d'application de l'intelligence artificielle, c'est celui qui produit actuellement les meilleurs résultats. C'est justement grâce à ses performances qu'il a attiré l'attention des chercheurs et qu'il est maintenant utilisé dans de nombreux systèmes intelligents.

On entend souvent que l'accès à un nombre de données de plus en plus important et l'augmentation de notre capacité de calculs ces dernières sont les deux facteurs qui ont lancé le Deep Learning sur le devant de la scène. Mais il est important de noter que les chercheurs n'avaient pas fois en ces techniques avant l'année 2012. Personne n'avait jamais réussi à construire un réseau de neurones artificiels qui fonctionne au moins aussi bien que d'autres techniques plus traditionnelles. Mais le 30 septembre 2012, le réseau de neurones prénommé AlexNet a gagné le concours ImageNet avec un taux d'erreur 10% plus faible que son meilleur compétiteur. A partir de ce jour, la communauté à changé de point de vue sur les réseaux de neurones artificiel.

Dans cet article vous allez apprendre les fondements du Deep Learning de manière progressive. Plus vous avancerez et plus la difficulté augmentera. Les curieux pourront se satisfaire des généralités, et les plus tenaces pourront coder un réseau de neurones de A à Z en Python. N'ayez pas peur et lancez-vous, vous irez peut-être plus loin que ce que vous imaginez !

# Première approche

Les voitures complètement autonomes n'existent pas encore aujourd'hui (06/2020), mais toutes les tentatives de construction de voitures autonomes utilisent des réseaux de neurones (ANN) pour piloter la voiture. On peut voir le ANN comme le cerveau de la voiture, il analyse les différentes données en provenance des capteurs situés sur la voiture, et choisis la marche à suivre en fonction de ces informations. Ces données peuvent être des vidéos enregistrées par des caméras embarquées, ou même une carte 3D de son environnement construite grâce à un LiDAR par exemple. Il peut décider de freiner brutalement lorsqu'un vélo traverse devant la voiture ou bien de doubler une voiture lente située devant la voiture.

Cette situation correspond à la phase de prédiction d'un ANN. Durant cette phase, le ANN est fonctionnel et les personnes qui l'ont développé ont évalué qu'il est suffisamment performant pour pouvoir être utilisé en situation réelle.

Le ANN est en fait un modèle mathématique, composé de beaucoup de paramètres. Concrètement ces paramètres sont des nombres à virgules, en générales assez petit, entre 0 et 1 par exemple. La valeur de ses paramètres agit directement sur les décisions que le modèle va prendre durant la phase de prédiction. Et ces valeurs ne sont pas choisies au hasard, elles sont déterminées durant la phase d'entraînement du modèle. C'est la phase où le ANN va "apprendre à conduire".

Au début de la phase d'entraînement, les paramètres du modèle ont des valeurs aléatoires. A ce moment là il ne sait pas conduire du tout, il réagit de manière complètement aléatoire face aux données qu'il reçoit. Pour trouver les valeurs optimales de chaque paramètres, le modèle suit un processus très simple, composé de 2 étapes: prédiction, correction. Nous allons lui proposer des données pour lesquelles nous savons quelle prédiction nous voulons qu'il fasse. Comme ses paramètres sont initialement aléatoires, il est presque certain que sa prédiction ne corresponde pas à nos attentes. Nous passons alors à l'étape de correction où nous mesurons l'erreur de sa prédiction, et nous modifions les valeurs des paramètres du modèle pour que sa prédiction soit juste la prochaine fois qu'il verra ses données.

Maintenant, si on lui montre de nouvelles données, mêmes si les paramètres ont été ajustés pour les données précédentes, sa prédiction ne conviendra sans doute toujours pas. Il faut donc encore une fois les ajuster. On répète donc ce processus avec beaucoup, beaucoup, beaucoup de données. Cela peut paraître simple. Le challenge se trouve dans le fait de déterminer une unique valeur pour chaque paramètre qui permette de faire la meilleure prédiction possible dans toutes les situations d'entraînement, mêmes si celles-ci ne se ressemblent pas du tout. Heureusement nous avons développé des algorithmes permettant d'automatiser ce processus d'apprentissage.

__Image panneau stop baissé__

Prenons l'exemple d'un panneau stop. S'il y a un chantier sur la route, il est possible que la voiture rencontre un humain portant un panneau stop. Il faut donc s'arrêter. Mais il est également possible que ce panneau soit porté vers le bas, ce qui signifie qu'il ne faut pas s'arrêter. Dans les deux cas, le modèle "voit" un panneau stop sur la vidéo provenant de la caméra, mais ses prédictions doivent être différentes.

Il est important de noter ici la différence entre algorithme et modèle. Le modèle est créé grâce à un algorithme. Et c'est le modèle qui est ensuite utilisé durant la phase de prédiction.

# Qu'est ce qu'un réseau de neurones artificiel

Les modèles mathématiques dans le cadre du Deep Learning sont composés de neurones. Le schéma suivant représente un ANN. Chaque cercle représente un neurone. Les neurones sont regroupés par couches. Elles sont ici représentées verticalement. Cet ANN est donc composé de 4 couches contenant 3, 5, 4, et 1 neurones.

La couche la plus à gauche est appelée la couche d'entrée. C'est par celle-ci que les données entrent dans le réseau. La couche la plus à droite est la couche de sortie. C'est par cette couche qu'une prédiction sort du réseau. Les couches intermédiaires sont appelées couches cachées.

__Schéma ANN__

La couche d'entrée étant le point d'entrée des données, elle comporte autant de neurones qu'il y a de données. Pour analyser une image et déchiffrer un chiffre écrit à main levée sur cette image, la couche d'entrée contient autant de neurones qu'il y a de pixels dans l'image. Ainsi chaque neurone prendra la valeur d'un pixel. Ensuite, ces valeurs vont naviguer de neurone en neurone dans lesquels elles vont subir différentes opérations mathématiques jusqu'à la couche de sortie.

__Image MNIST__

Dans tous les autres neurones du réseau, des opérations ont lieu. C'est le résultat final de ces opérations qui définit la valeur d'un neurone. Et c'est également cette valeur qui est transmise aux neurones de la couche suivante. Une fois arrivé dans la couche de sortie, il n'y a plus de neurone auquel transmettre une valeur. Alors la valeur des neurones de la couche de sortie correspond à la prédiction du modèle.

Lorsque la couche de sortie ne contient qu'un seul neurone, c'est en générale une prédiction binaire : vrai ou faux, 0 ou 1. Par exemple, on peut donner au neurone l'image ci-dessus, et lui demander si le chiffre écrit sur cette image est un 5. Si la valeur du dernier neurone est un 1, cela signifie que le modèle prédit que le chiffre est bien un 5. En revanche si cette valeur est 0, alors le modèle prédit que cette image ne représente pas un 5. Il est possible qu'il se trompe. Il prend des décisions en fonction de ce qu'il a apprit durant la phase d'entraînement. Cependant il est impossible de couvrir tous les cas possibles à ce moment là en un temps fini.

__Lien vers notebook. ANN sur MNIST au niveau d'abstraction__

# A l'intérieur d'un neurone

Pour aller plus loin et comprendre les opérations qui sont réalisées dans un ANN pour faire une prédiction à partir de données. Il est nécessaire de plonger à l'intérieur d'un neurone pour en étudier le fonctionnement.

On peut voir sur le schéma précédent à plusieurs liaisons d'entrée. Il y en a une pour chaque neurone de la couche précédente. Pour déterminer sa valeur finale, un neurone procède en 2 étapes. Tout d'abord il y a la phase de pré-activation. C'est à ce moment là que toutes les valeurs en entrées sont regroupées en une seule. Tout simplement en les sommant. Ensuite c'est la phase d'activation qui consiste à ramener cette valeur dans un intervalle plus petit, entre -1 et 1 par exemple. Cela permet d'uniformiser les valeurs des différents neurones d'une même couche et donc de faciliter l'apprentissage des couches suivantes.
