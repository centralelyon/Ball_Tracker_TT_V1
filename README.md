<h1> Tracker de ball de tennis de table </h1>

<h2> Pré-traitement </h2>

Le premier programme est appliqué aux frames extraites de la video (extraite avec opencv .read(video))

Ce programme permet de retirer la moyenne de l'arrière plan d'une vidéo avec l'aide de OpenCV, pour cela on crée un BackgroundSusbstractor (MOG2 ou KNN) qui va venir apprendre et modifier chaque image de la vidéo. On parcourt les frames de la vidéo avec cv2.VideoCapture et cv2.read
