# Weniger Krumme Touren

❔ A2 👤 64712 🧑 Leonhard Masche 📆 15.04.2023

## Inhaltsverzeichnis

1. [Lösungsidee](#lösungsidee)
2. [Umsetzung](#umsetzung)
    1. [Verbesserungen](#verbesserungen)
    2. [Laufzeit](#laufzeit)
    3. [Komplexität](#komplexität)
3. [Beispiele](#beispiele)
4. [Quellcode](#quellcode)

## Lösungsidee

Die Aufgabe wird als ein kompletter Graph $G(V, E)$ dargestellt. Hierbei sind die $|V|$ Scheiben die Knoten im Graph, und die Kanten $E$ repräsentieren ein Aufeinanderfolgen dieser Scheiben/Knoten. Nun gilt es einen Hamiltonpfad in diesem Graphen zu finden, der die geometrischen Bedingungen der orthogonalen Schnitte erfüllt. Existiert dieser, gibt es für diese Käsescheiben eine Lösung und der Hamilton-Pfad (startend von der End-Scheibe mit der kleineren Fläche) ist die Reihenfolge, in der die Scheiben wieder zusammengefügt werden können.

Um zu sehen, ob zwei Schieben zusammengefügt werden können, wird die folgende Beobachtung verwendet:

**Lemma 1**: Nach einer Scheibe kann eine andere nur hinzugefügt werden, wenn sie in mindestens einer der beiden Größen mit der vorherigen übereinstimmt.

*Beweis*: Wenn eine Scheibe nach einer anderen hinzugefügt wird, muss sie sich mit ihr mindestens eine Kante von gleicher Länge teilen.

Eine Scheibe, die diese Bedingungen erfüllt passt aber nicht immer auf den Quader. Zusätzlich muss während dem Aufbauen also geprüft werden, ob die Scheibe wirklich die gleichen Dimensionen wie eine Seite des Quaders hat.

## Umsetzung

Da es von dem vorherigen Pfad abhängt, ob eine Kante ausgewählt werden kann oder nicht, hilft in diesem Fall nur schlaues ausprobieren.

Zuerst werden die Scheiben in eine Liste geladen. Aus dieser  wird nun eine Lookup-Tabelle von Seitenlänge zum Index in der Liste erstellt, um schnell auf potentiell anfügbare Scheiben zugreifen zu können. Nun wird ein Backtracking-Algorithmus angewendet. Es werden immer weiter passende Scheiben hinzugefügt, und falls keine weitere Lösung möglich ist, wird der Pfad zurückvefolgt, bis es weitere mögliche Nachbarn gibt und dieser Pfad wird genauso weiterverfolgt.

Das Programm (`program.py`) ist in Python geschrieben und mit einer Umgebung ab der Version `3.8` ausführbar. Es werden nur Standard-Bibliotheken verwendet. Wird das Program aufgerufen, fragt es nach der Zahl des Beispiels und berechnet die Lösung für dieses Anschließend. Zusätzlich wird diese Lösung für die BWINF-Beispiele in Textform in dem Ordner `output` gespeichert. Jede Zeile beschreibt eine Scheibe aus dem Beispiel in der Reihenfolge, in der sie hinzugefügt werden.

### Verbesserungen

#### Deduplizierung

Nachbarn mit gleicher Größe werden dedupliziert. So wird ein unnötiges mehrfaches Besuchen dieser Nachbarn verhindert, welches garantiert nicht zu einer Lösung führt.  TODO elaborate

#### Aufgegessen

Da hatte Antje doch zu viel Hunger und hat einige Scheiben aufgegessen! Eine modifizierte Version des Programmes kann auch Beispiele lösen, in denen Scheiben fehlen. Dazu überprüft es nicht nur Nachbarn mit den passenden Dimensionen, sondern auch Nachbarn, die in jeweils einer Dimension um $1$ größer sind. Tritt ein solcher Fall ein, wird dem Pfad eine 'virtuelle' Scheibe hinzugefügt, und weiter iteriert. So kann das Programm einzelne Scheiben die im Stapel fehlen wiederherstellen. Sollten zwei oder mehr Scheiben in Folge fehlen, werden die Scheiben auf mehrere Quader verteilt. Da nun statt maximal drei Möglichkeiten, eine Scheibe anzufügen, $12$ Möglichkeiten betrachtet werden, steigt der Rechenaufwand auch sehr schnell mit der Länge des Beispiels.

#### Mehr Käse

Auch wenn Antje an einem Tag mehrfach telefoniert, und alle Käsescheiben vermischt hat, kann auch dieses Problem gelöst werden. Dazu werden von jedem Startknoten aus alle Pfade mit maximaler Länge generiert. Diese müssen aber nicht vollständig sein. Nun wird jede Kombination aus $n$ Pfaden überprüft, wobei $n$ von $1$ bis hin zur Länge des Käsestapels erhöht wird. Sind in einer Kombination zu viele Knoten enthalten, werden sie vom Ende der Pfade entfernt. Wurden Lösungen gefunden, wird die mit den wenigsten 'aufgegessenen' Scheiben zurückgegeben.

### Laufzeit

Bei solchen Problemen liegt es nahe, einfach alle Kombinationen auszuprobieren, was eine Laufzeit von $\mathcal O(n!)$ bedeuten würde.

Nun kann man aber feststellen, dass an einen Quader von beliebiger Größe $a \times b\times c$ nur maximal drei Scheiben ($a \times b$, $b\times c$ und $a\times c$) angefügt werden können. Weitere Scheiben mit den gleichen Maßen können vernachlässigt werden, da diese logischerweise zu derselben Lösung führen würden. Wenn nun also für jede mögliche Start-Scheibe alle Kombinationen ausprobiert werden, ergibt sich eine Worst-Case Zeitkomplexität von $\mathcal O(n*3^{n-1})$,  wobei die Basis $3$ die maximale Anzahl der Nachbaren ist.

Das ist nun aber die Worst-Case Laufzeit des Programmes. In Wirklichkeit liegt die (experimetell ermittelte) durchnittliche Anzahl an Nachbarn während dem Lösungsvorgang zwischen $1.00$ und $1.04$. TODO proof

Somit befindet sich auch die Zeitkomplexität im Bereich zwischen $\mathcal O(n*1.00^{n-1})$ und $\mathcal O(n*1.04^{n-1})$.  TODO proof

-> berechnung kompexitäts-basis

### Komplexität

shortest hamiltonian path

Die Aufgabe kann auf das Hamiltonian-Path-Problem reduziert werden und ist als solches NP-Komplett.

## Beispiele

## Quellcode
