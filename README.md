# README.md
In this repo, we build a model to predict the delivery time of an order for Doordash.

The attached file `data/input/historical_data.csv` contains a subset of deliveries received at DoorDash in early 2015 in a subset of the cities. Each row in this file corresponds to one unique delivery. Each column corresponds to a feature as explained below. Note all money (dollar) values given in the data are in cents and all time duration values given are in seconds 
The target value to predict here is the total seconds value between `created_at` and `actual_delivery_time`. 

The Python code can be found in the directory `src/Python/`. The final report is at `docs/report.pdf`.
