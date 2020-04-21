# How to Run

preprocess(already done)
```
python preprocess.py
```

train model
```
python3 alined.py bpe_hindi_train_3.pt 1 hindi_1
```

predict
```
python3 predictor.py bpe_hindi_train_3.pt 1 hindi_1 predict.txt
```

Predictor Output:
//The output of the predictor is in the format :
list of (chunk_type, chunk_start, chunk_end)
//Example:
seq = [4, 5, 0, 3]
tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
result = [("PER", 0, 2), ("LOC", 3, 4)]
