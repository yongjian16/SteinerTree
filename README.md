# End-to-End Risk-aware Reinforcement Learning to Detect Asymptomatic Cases in Epidemic Outbreaks

## Data
Graph and features are from UIHC database and they are in `data/` directory.
Each instance is a day of a visit of a patient in UIHC.

### `G500.graphml`: The time-expanded graph with synthetic ground truth data (503 nodes and 1539 edges)
Node attributes:
- `terminal`: indicator of whether the node is terminal
- `ASYMP`: indicator of whether the node is asymptomatic case

Edge attributes:
- `weight`: edge weight
### `features.csv`: 17640 instances
Columns are: `[vid,pid,day,los,age,gender,prev_visit,ABX1,ABX2,ABX3,ABX4,ABX5,GAS1,GAS2,cdiff,ABX1_cum,ABX2_cum,ABX3_cum,ABX4_cum,ABX5_cum,GAS1_cum,GAS2_cum,SCP_unit,SCP_0,MCP_unit,MCP_0,ABX1_mean,ABX2_mean,ABX3_mean,ABX4_mean,ABX5_mean,GAS1_mean,GAS2_mean]`
- `vid`: visit id of the patient
- `pid`: patient id
- `day`: day of the visit
- `los`: length of the stay starting from the admission time until the current date
- `age`: age of the patient at admission time
- `gender`: gender of the patient
- `prev_visit`: indicator of whether there was a previous visit wihtin 2 month of the current visit
- `ABX1`: Amoxicillin
- `ABX2`: Clindamycin
- `ABX3`: Third gen Cephalosporin
- `ABX4`: Fourth gen Cephalosporin
- `ABX5`: Flouroquinolone
- `GAS1`: H2 blocker (gastric acid suppersor)
- `GAS2`: Proton pump inhibitor
- `cdiff`: indicator of whether the patient got CDI positive test in the current visit
- `ABX1_cum`: cumulative medication prescription of Amoxicillin
- `ABX2_cum`: cumulative medication prescription of Clindamycin
- `ABX3_cum`: cumulative medication prescription of Third gen Cephalosporin
- `ABX4_cum`: cumulative medication prescription of Fourth gen Cephalosporin
- `ABX5_cum`: cumulative medication prescription of Flouroquinolone
- `GAS1_cum`: cumulative medication prescription of H2 blocker (gastric acid suppersor)
- `GAS2_cum`: cumulative medication prescription of Proton pump inhibitor
- `SCP_unit`: sum CDI pressures in the same unit
- `SCP_0`: sum CDI pressures in the same room
- `MCP_unit`: mean CDI pressures in the same unit
- `MCP_0`: mean CDI pressures in the same room
- `ABX1_mean`: mean medication prescription of Amoxicillin 
- `ABX2_mean`: mean medication prescription of Clindamycin
- `ABX3_mean`: mean medication prescription of Third gen Cephalosporin
- `ABX4_mean`: mean medication prescription of Fourth gen Cephalosporin
- `ABX5_mean`: mean medication prescription of Flouroquinolone
- `GAS1_mean`: mean medication prescription of H2 blocker (gastric acid suppersor)
- `GAS2_mean`: mean medication prescription of Proton pump inhibitor

## Requirements

We use Python3. Please install the Python libraries:
```setup
networkx
sklearn
pandas
numpy
matplotlib
torch
dgl
collections
```

## Training
Our algorithm will run for several days, so it's important to checkpoint the models.
The models will be saved in `models/` directory. If you want to start from a model, 
you can use `-modelname` to specify it.
To train the models, run this command:
```train
python train.py
```
You can set the parameters to tune your model:
```
--gamma: edge weight between dummy root with first occurrence of all the nodes
--alpha: node importance
--graph: input graph name
--features: feature file name
--gae: pretrain gae epochs
--ffn: pretrain ffn epochs
--pretrain: pretrain gae and ffn together epochs
--rl: train rl epochs
--modelname: model name
--batchsize: batch size
--frequency: how often do you want to save the model
```

## Evaluation
To evaluate the model and get the solution graph 
(the solution graph will be saved in `results/` directory), 
we need to load the model from `models/` directory by including `-modelname` and the 
name of the latest model
and run:
```eval
python test.py -modelname models/19000.tar
```
You can set the parameter to tune your model(testing)
```
--gamma: edge weight between dummy root with first occurrence of all the nodes
--alpha: node importance
--graph: input graph name
--features: feature file name
--modelname: model name
```

## Pre-trained Models

The pre-trained model is in `pretrained/` directory. You can load the pre-trained
model, get the solution graph, and test the accuracy by running:
```
python test.py -modelname pretrained/19900.tar
```

## Results

Our model achieves the following performance on:

| Model name | f1 score  |
|------------|-----------|
| Our model  | 0.5862    |