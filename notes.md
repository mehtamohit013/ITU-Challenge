# Inferences
*   Lowering the discount function to zero leds to smoothed (at 0.99) value of -0.1507 and average value of -0.1405
*   Increasing the target update to 100 increases the smoothed to -0.1601 and average to -0.1583
*   Reducing the weight decay from 200 to 50  leads to smoothed valud of -0.1587 and average of -0.15874
*   Reducing the weight decay to 10 leads to smmothed value of -0.1547 and average of -0.15506 (tensorboard_result.py not working)
*   wd to 0.1 -> smoothed = -0.152  and average of -0.1533
*   Target update : 1000 -> smoothed = -0.1551 and average of -0.1523
*   Increasing the EPS_END to 0.3 -> smoothed = -0.1537 and average of -0.1536    