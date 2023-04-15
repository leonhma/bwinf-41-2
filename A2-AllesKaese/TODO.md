# TODO

(a, b) ist in aufsteigender reihenfolge sortiert
nach dem hinzufügen von scheibe (a, b) hat quader die form (a, b, n).
d. h. es kann eine scheibe mit den maßen (a, b), (a, n) oder (b, n) hinzgefügt werden

a, b = (x1, y1), (x2, y2)
->
len(a.intersection(b)) >= 1

---

hashmap of (a, b) (sorted) -> number of slices
stack is sorted by (a, b) (sorted)

---

muss in zwei dimensionenx üebreinstimmen, aber da dx_3 immer 1 ist, kann die dritte vernachlässigt werden

---

bsp7 "langer anruf"
"wohl bekommts"
"guten appetit"

---

verify solutions

---

reihenfolge is unique

---

fully connected graph - obviously illegal connections are pruned
IDEAS

- for every node calculate the largest number of nodes that can be connected from it in one path,
  check if that number is >= the number of nodes remaining

## NP-Complex or hard?

---

weniger blöcke werden bevorzugt, danach werden weniger aufgegessene scheiben bevorzugt

---

kaese8 testet beschränkung der beispiellänge und aufgegessene scheiben
9 testet 5 blöcke
10 testet aufgegessene schieben + mehrere blöcke

---

runnable since `3.8`

---

unnötige ineffizienz entfernen
