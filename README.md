## bigO

`bigO` automatically measures empirical computational complexity (in both time and space) of functions.
To use `bigO`, you just need to add a `@bigO.track` decorator to your function together with a "length" function (the "n").
You can then run your function as usual, ideally along a wide range of inputs, and `bigO` will report the computational
complexity of that function.

`bigO` accumulates its results in a file named `bigO_data.json` in the local directory;
you can then generate a graph of time and space complexity for each tracked function by running `python3 -m bigO.graph`.

### Demonstration

The file `test/facts.py` is a small program that demonstrates `bigO`.

```python
import bigO

def fact(x: int) -> int:
  v = 1
  for i in range(x):
    v *= i
  return v

@bigO.track(lambda xs: len(xs))
def factorialize(xs: list[int]) -> list[int]:
  new_list = [fact(x) for x in xs]
  return new_list

# Exercise the function - more inputs are better!
for i in range(30):
    factorialize([i for i in range(i * 100)])
```

Now run the program as usual:

```bash
python3 test/facts.py
```

Now you can easily generate a graph of all tracked functions. Just run the following command in the same directory.

```bash
python3 -m bigO.graph
```

This command creates the file `bigO.pdf` that contains graphs like this:

![bigO](https://github.com/user-attachments/assets/8428180b-a454-4fc7-822c-7a130f9ba54e)

### Technical Details

#### Curve-fitting

`bigO`'s curve-fitting based approach is directly inspired by
["Measuring Empirical Computational
Complexity"](https://theory.stanford.edu/~aiken/publications/papers/fse07.pdf)
by Goldsmith et al., FSE 2007, using log-log plots to fit a power-law distribution.

Unlike that work, `bigO` also measures space complexity by
tracking memory allocations during function execution. In addition,
`bigO` uses the [AIC](https://en.wikipedia.org/wiki/Akaike_information_criterion) to
select the best model. 