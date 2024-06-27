# BotRecognitionTDA

Plan:
stage1.py handelt das extrahieren von ego netzwerken pro nutzer und nutzt die adjazenzmatrizen um den die Persistenzdiagramme zu berechnen.
Output ist hier dann eine datei mit persistenzdiagrammen

(, eine mit persistenzbildern und eine mit persistenzlandschaften) <-- evtl nur persistenzdiagramme, die anderen kann man vielleicht auch locally computen

Dann können die Daten verwendet werden um Models zu trainieren.


TODO:
-  stage1.py ergänzen (ego network statistics und TDA)
-  WeightedRips oder Rips? (lieber Weighted weil anscheinend mehr Nuancen)
DONE:
- stage1.py wandelt jetzt erfolgreich ein directed ungewichtetes ego netzwerk in ein ungerichtetes gewichtetes ego netzwerk perfekt für TDA um. Gibt sogar eine distance matrix aus (cool)
- Dockerfile für Kubernetes Service auf AMD64 hat einwandfrei gebaut. Ich erwarte, dass es später damit noch Probleme geben wird, just because. Jetzt muss das nur noch irgendwie erfolgreich als Service im Cluster eingebunden werden. Joa.

Probleme:
- Dockerfile für Kubernetes Service erstellt, aber wie benutzen? Main Frage: Wie bekomme ich die Daten nachher wieder runter und wie erstelle ich den Service so dass er ein Persistent Volume benutzt?
- docker push failed, weil permissions fehlen



IDEE MAPPER:
Persistenzdiagramme von Ego Networks per MDS in point cloud umwandeln --> Mapper darauf anwenden
- einmal um outliers zu sehen (meh)
- oder splitten in bots und humans und für bot subset mapper anwenden, dann einzelne Botgruppen erkennen um Bot subklassen für statistical approach zu haben --> mehr explainable und fine-grained
