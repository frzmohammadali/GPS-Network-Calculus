## without any "seed":

```json
{
  "number_of_flows": 10,
  "weights_mode": "WeightsMode.EQUAL",
  "seed": "None",
  "run": "util"
}
```
=> when weights are equal, difference of delay bounds are subtle no matter what utilization. (pattern stable)

```json
{
  "number_of_flows": 10,
  "weights_mode": "WeightsMode.RANDOM",
  "seed": "None",
  "run": "weights"
}
```
=> for totally random weight scenario, the higher the weight of foi, the more subtle the difference in delay bounds 
(pattern stable)

```json
{
  "number_of_flows": 10,
  "weights_mode": "WeightsMode.RANDOM",
  "seed": "None",
  "run": "burst"
}
```
=> for totally random weights scenario, the lower the burst of foi, the lower the difference in delay bounds (pattern
 stable)

```json
{
  "number_of_flows": 10,
  "weights_mode": "WeightsMode.RPPS",
  "seed": "None",
  "run": "burst" or "util"
}
```
=> given the range of foi burst or util, in RPPS weights scenario, chang and BL and PG are almost exact match and 
Bouillard is not crazy bad (pattern stable)

## Now let's play with "seed":
```json
{
  "number_of_flows": 10,
  "weights_mode": "WeightsMode.RANDOM",
  "seed": "10",
  "run": "util"
}
```
=> in a random weights scenario with whatever seed, the higher the utilization, the higher the delay bounds