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

[X] verify solutions

---

"Hashmap hilft"

---

fully connected graph - obviously illegal connections are pruned
IDEAS

- for every node calculate the largest number of nodes that can be connected from it in one path,
  check if that number is >= the number of nodes remaining

## NP-Complex

---

run pipreqs

start node can't be missing

---

fix: full solutions are duplicate because backtracking is also triggered

---

prefer solutions that don't add slices

---

filter first nodes to be unique in size

---

ignore new paths using virtual nodes if a path without can be found

TODO add tracking of 'virtual' nodes
