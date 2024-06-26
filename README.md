# BotRecognitionTDA

Plan:
stage1.py handelt das extrahieren von ego netzwerken pro nutzer und nutzt die adjazenzmatrizen um den die Persistenzdiagramme zu berechnen.
Output ist hier dann eine datei mit persistenzdiagrammen

(, eine mit persistenzbildern und eine mit persistenzlandschaften) <-- evtl nur persistenzdiagramme, die anderen kann man vielleicht auch locally computen

Dann können die Daten verwendet werden um Models zu trainieren.


TODO:
-  stage1.py ergänzen (ego network statistics und TDA)
-  WeightedRips oder Rips? (lieber Weighted weil anscheinend mehr Nuancen)
-  Oder lieber doch Weight ranked Clique? Das gibts aber nirgends für python anscheinend.
-  
