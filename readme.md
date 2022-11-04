# Loan Application Status App
A series of machine learning models that were trained on data provided by LendingClub. Each model was trained with a certain task. These tasks include:

* Application Status Prediction
* Loan Grade Prediction
* Loan Subgrade Prediction
* Interest Rate Prediction

All models are deployed in containers on cloud and can be reached using an API. 

Data analysis and model prototyping steps can be found in the notebooks ahead.

## Application Status Prediction

```
response = requests.post("https://loancontainer-zqnmu75hoa-uc.a.run.app/predict-application-status", json=data_example)
```

```
response.text
```

```
>>> 0, 0.89
```
## Loan Grade Prediction

```
response = requests.post("https://loancontainer-zqnmu75hoa-uc.a.run.app/predict-grade", json=data_example)
```

```
response.text
```

```
>>> A, 0.59
```

## Loan Subgrade Prediction

```
response = requests.post("https://loancontainer-zqnmu75hoa-uc.a.run.app/predict-subgrade", json=data_example)
```

```
response.text
```

```
>>> A2, 0.77
```

## Interest Rate Prediction

```
response = requests.post("https://loancontainer-zqnmu75hoa-uc.a.run.app/predict-intrate", json=data_example)
```

```
response.text
```

```
>>> 12.1
```

## Notebooks

Part 1: https://colab.research.google.com/drive/1wnf2686_EPoQ_RQT32IGaveegzteGy8B?usp=sharing

Part 2: https://colab.research.google.com/drive/1rDMiln6gjN3P5TcOZ6lmJlXs3QXfCChb?usp=sharing

## Examples

https://colab.research.google.com/drive/1GDGuwehKSlz1MAXKQqC6ZUPGSYhV0_1f?usp=sharing