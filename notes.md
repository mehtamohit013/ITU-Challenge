# Inferences
*   Lowering the discount function to zero leds to smoothed (at 0.99) value of -0.1507 and average value of -0.1405
*   Increasing the target update to 100 increases the smoothed to -0.1601 and average to -0.1583
*   Reducing the weight decay from 200 to 50  leads to smoothed valud of -0.1587 and average of -0.15874
*   Reducing the weight decay to 10 leads to smmothed value of -0.1547 and average of -0.15506 (tensorboard_result.py not working)
*   wd to 0.1 -> smoothed = -0.152  and average of -0.1533
*   Target update : 1000 -> smoothed = -0.1551 and average of -0.1523
*   Increasing the EPS_END to 0.3 -> smoothed = -0.1537 and average of -0.1536    
*   Adding channel_mag to reward and running it with best hyperparams leds to -0.1420
*   Increasing weight decay from 200 to 1000 leads to smoothed value of -0.158 and avg of  -0.1576
*   Increasing weight decay from 200 to 10000 leads to smoothed value of -0.1551 and avg of  -0.1552
*   Increasing weight decay from 200 to 10K leads to smoothed value of -0.1509 and avg of  -0.1489
*   Increasing weight decay from 200 to 80K leads to smoothed value of -0.1594 and avg of  -0.1581

# Inference with val
*   Standard : Validation Overall reward = -10556.43.  Validation Average Reward = -0.1537
*   With eps_decay = *0.1 : Validation Overall reward = -9452.27.  Validation Average Reward = -0.1376 
*   With gamma = 0 : Validation Overall reward = -10605.35.  Validation Average Reward = -0.1544
*   With target update = 10 : Validation Overall reward = -9785.65.  Validation Average Reward = -0.1424
*   With eps_end=0.5 :s

# Meeting
*   Reward function discussion and buffer packets; observation space??
*   Observation space
*   Preprocessing data