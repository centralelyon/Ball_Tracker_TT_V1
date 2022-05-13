<h1> Tracker de ball de tennis de table </h1>

<h2> Pré-traitement </h2>

Le premier programme est appliqué aux frames extraites de la video (extraite avec opencv .read(video))

Ce programme permet de retirer la moyenne de l'arrière plan d'une vidéo avec l'aide de OpenCV, pour cela on crée un BackgroundSusbstractor (MOG2 ou KNN) qui va venir apprendre et modifier chaque image de la vidéo. On parcourt les frames de la vidéo avec cv2.VideoCapture et cv2.read

<h2> Premier réseau de neurones : Classification </h2>

En effet on apprend tout d'abord à notre algorithme à reconnaitre les images qui contiennent une balle de tennis de table et celles qui n'en contiennent pas afin que notre programme ne calcule pas sur des images inutiles. Problème classique de classification avec 2 sorties (type chien/chat)

Pour entrainer cet algorithme il faut séparer les 2 manuellement, j'utilise le fichier CSV qui comporte les frames annotées (les frames non annotées ne contiennent pas la balle), cet algo est dans Tri_Train_Test.py
