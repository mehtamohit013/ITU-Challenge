from tensorflow.python.summary.summary_iterator import summary_iterator
import tensorflow as tf

val = 0
cnt = 0
for e in summary_iterator("/home/sundesh/Documents/git/ITU-Challenge/tb_logs/11_obs_space/events.out.tfevents.1630354729.sundesh-Aspire-A715-75G.24531.0"):
    for v in e.summary.value:
        if (v.tag == 'Reward'):
            val += v.simple_value
            cnt += 1
            # print(v.simple_value, v.tag)
    # if v.tag == 'loss' or v.tag == 'accuracy':
    # print(v.simple_value)
    # break

# -11038.178193164465 68700 -0.1606721716617826 for 7 observations basic dqn events.out.tfevents.1630352988.sundesh-Aspire-A715-75G.22460.0
# -11386.800883056156 68700 -0.16574673774463108 for 11 observations(orien xyzw added) basic dqn events.out.tfevents.1630352988.sundesh-Aspire-A715-75G.22460.0
print(val, cnt, val/cnt)