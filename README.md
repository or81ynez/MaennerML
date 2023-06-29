# MaennerML

# Eine Fahrzeug Erkennungsapp 

Schwerpunkt der App : #

   # Wichtige Artikel 

* A Deep Learning Model for Transportation Mode Detection Based on Smartphone Sensing Data von : Xiaoyuan Liang , Yuchuan Zhang, Guiling Wang, and Songhua Xu
* Inhalt des Atikels: 
  Der Artikel beschäftigt sich mit der Entwicklung eines Deep Learning Modells zur Erkennung des Transportmodus anhand von Sensordaten von Smartphones. Durch die Analyse von GPS, Beschleungigungsmesser und Gyroskop- Daten können charakteristische Muster verschiedener Transportmodi indentifiziert werden. Das entwickelte Modell wurde mit umfangreichen Testdaten evaluiert und erreichte hohe Genauigkeitswerte. Die Autoren sehen großes Potenzial für die Anwendung des Modells in der Mobilitätsanalyse und Verkehrsoptimierung, beispielweise für die Verbesserung der Routenplanung und Verkehrsprognosen. Insgesamt zeigt der Artikel, das die Kombination von Smartphone-Sensordaten und tiefem Lernen eine effektive Mehtode zur Erkennung des Transportmodus darstellt.


* Transportation Mode Detection Using Temporal Convolutional Networks Based on Sensors Integrated into Smartphones Von : Pu Wang and Yongguo Jiang
* Inhalt des Artikels: Der Artikel befasst sich mit der Nutzung von Sensoren, die in Smartphones integriert sind, zur Erkennung des Transportmodus. Die Autoren verwenden speziell zeitliche Faltungsnetzwerke( Temporal convolutional networks, TCN) um Muster in den Sensorinformationen zu erkennen. Durch die Analyse von Beschleunigungsmesser- und Gyroskop-Daten können charakteristische Merkmale der verschiedenen Transportmodi indentifiziert werden. Das entwickelte TCN-Modell wird mit umfangreichen Testdaten evaluiert und erzielt hohe Erkennungsraten für verschiedene Transportmodi wie Gehen,Laufen Fahrradfahren und Fahren mit dem Auto. Die Ergebnisse zeigen, das die TCN-Methode effektiv ist und eine vielversprechende Lösung für die Transportmoduserkennung auf Smartphones darstellt. Die Autoren schließen daraus, dass ihr Ansatz zur Verbesserung der Mobilitätsanalyse und Verkehrsplanung beitragen kann. Insgesamt zeigt der Artikel das Potenzial von TCN und Smartphone- Sensoren für die


* Custom Dual Transportation Mode Detection by Smartphone Devices Exploiting Sensor Diversity  von : Claudia Carpineti, Vincenzo Lomonaco, Luca Bedogni, Marco Di Felice, Luciano Bononi Department of Computer Science and Engineering, University of Bologna, Italy
* Inhalt des artikels: Der Artikel beschäftigt sich mit der Entwicklung einer Individuellen Erkennungsmethode für zwei Transportmodi mithilfe von Smartphones. Die Autoren nutzen die Vielfalt der Sensoren in Smartphones, wie Beschleunigungsmesser und Gyroskope, um charakteristische Merkmale der Transportmodi zu erfassen. Durch die Kombination verschiedener Sensoren werden präzisere und zuverlässigere Erkennungsergebnisse erzielt. Die entwickelte Methode wird anhand einer umfangreichen Testdatensammlung evaluiert und erreicht hohe Erkennungsgenauigkeiten für die zwei spezifischen Transportmodi. Die Ergebnisse zeigen das Potenzial der Sensorvielfalt in Smartphones für die maßgeschneiderte Erkennung von Transportmodi. Die Autoren betonen die Bedeutung dieser Methode für die Optimierung von Mobilitätsanwendung und die Verbesserung des individuellen Reiseverhaltens. Insgesamt zeigt der Artikel, wie Smartphone-Geräte durch die Ausnutzung der Sensorvielfalt eine maßgeschneiderte Erkennung von Transportmodi ermöglichen können.


# Zeitlicher Ablauf der Entwicklug

<img width="875" alt="Bildschirmfoto 2023-06-29 um 12 18 24" src="https://github.com/or81ynez/MaennerML/assets/131467070/a315bd50-36c8-4f95-9e14-9e4ca4eae83e">


# 1.Projektübersicht : Unsere App zielt drauf ab, das aktuell verwendete Transportmittel zu erkennen. 
* Das Ziel unserer App ist es, Nutzern in Echtzeit relevante Informationen und praktische Lösungen für ihre individuellen Transportbedürfnisse zu liefern
  
# 2.Technologien und Tools

* Sensor Logger App : Für Datenerfassung
* Google Drive : Teilen der Daten für andere Teammitglieder
* Google Colab : Für die Zusammenarbeit am Code
* Python : Zum Codieren
* Streamlit : App Veröffentlichung
* 



# 3.Datenerfassung 

* Daten erfassung unterschiedlichen Verkehrmittel mit der "Sensor Logger"  App
  
<img width="334" alt="Bildschirmfoto 2023-06-29 um 17 42 12" src="https://github.com/or81ynez/MaennerML/assets/131467070/81cc26c7-e098-40c4-b8c5-0a85e8417bec">
<img width="334" alt="Bildschirmfoto 2023-06-29 um 17 43 46" src="https://github.com/or81ynez/MaennerML/assets/131467070/53d734ac-d976-4ce1-ad5b-21aaff73d461">



# 4.Datenverarbeitung und Modellbildung
*
*

# 5.Entwicklungsphasen
* Recherche und Datenauswahl: Wir führten umfangreiche Recherchen durch, um herauszufinden, welche Daten sich am besten für unseren Zweck eignen. Wir untersuchten verschiedene Datenquellen, wie beispielsweise Sensoren oder APIs, um genaue Informationen über das aktuell genutzte Transportmittel zu erhalten.
* Datensammlung: Die Sammlung ausreichender Daten erwies sich als eine Herausforderung. Wir benötigten eine große Menge an qualitativ hochwertigen Daten, um unsere App effektiv zu trainieren. Dabei stellten wir fest, dass wir zu Beginn fehlerhafte Daten aufgenommen hatten. Dies erforderte, dass wir die Daten löschen und den Sammelprozess erneut starten mussten, um genaue und zuverlässige Informationen zu erhalten.
* Datenbereinigung und -verarbeitung: Nachdem wir die gesammelten Daten hatten, mussten wir sie bereinigen und aufbereiten, um sie für die spätere Verarbeitung optimal zu nutzen. Dies beinhaltete das Entfernen von Rauschen, Ausreißern oder inkonsistenten Datenpunkten. Wir verwendeten verschiedene Datenverarbeitungstechniken und Algorithmen, um sicherzustellen, dass die Daten qualitativ hochwertig und repräsentativ waren.
* Modellbildung und Algorithmusauswahl: Ein weiterer wichtiger Schritt bestand darin, ein Modell zu entwickeln, das in der Lage ist, das aktuelle Transportmittel zu erkennen. Wir haben verschiedene Algorithmen und Machine-Learning-Techniken evaluiert, um die beste Leistung und Genauigkeit zu erzielen. Dies erforderte eine sorgfältige Auswahl und Anpassung des Modells an unsere spezifischen Anforderungen.
* Validierung und Optimierung: Nach der Modellbildung haben wir das System umfangreich validiert, um sicherzustellen, dass es zuverlässig und genau funktioniert. Dies beinhaltete die Verwendung von Testdaten und die Überprüfung der Ergebnisse im Vergleich zu den tatsächlichen Transportmitteln. Wir haben das Modell iterativ optimiert und verbessert, um eine höhere Genauigkeit und Robustheit zu erreichen.
* Während des gesamten Entwicklungsprozesses waren wir mit Herausforderungen konfrontiert, die es erforderten, kreative Lösungen zu finden. Dazu gehörten auch die Zusammenarbeit im Team, das Überwinden technischer Hindernisse und das kontinuierliche Feedback von Testnutzern. Durch unsere Bemühungen und die iterative Entwicklung kon
# 6.Funktionalitäten

# 7. Testing und Validierung

# 8. Zukünfitge Erweiterungen : 

* Eine Möglichkeit besteht darin, den eigenen Status während der Autofahrt automatisch an Kontakte zu übermitteln, um sicherzustellen, dass man während der Fahrt nicht abgelenkt wird und keine Anrufe entgegennimmt.
* Des Weiteren kann die App die beste Route sowohl für Autofahrten als auch für Fahrradfahrten berechnen wodurch sie sich als praktischer begleiter im Straßenverkehr erweist.
* Auch bei der Nutzung öffentlicher Verkehrsmittel erweist sich die App nützlich, indem sie die besten Verbindungsmöglichkeiten und Routenempfehlungen bietet. 


# 
