from tensorflow.python.summary.summary_iterator import summary_iterator
import tensorflow as tf

val = 0
cnt = 0
for e in summary_iterator("/home/mohit/ITU-Challenge/ITU-Challenge/log_tensorboard/11_obs_space/events.out.tfevents.1630402713.mehta.18782.2"):
    for v in e.summary.value:
        if (v.tag == 'episode_reward'):
            val += v.simple_value
            cnt += 1
            # print(v.simple_value, v.tag)
    # if v.tag == 'loss' or v.tag == 'accuracy':
    # print(v.simple_value)
    # break

# -11038.178193164465 68700 -0.1606721716617826 for 7 observations basic dqn events.out.tfevents.1630352988.sundesh-Aspire-A715-75G.22460.0
# -11386.800883056156 68700 -0.16574673774463108 for 11 observations(orien xyzw added) basic dqn events.out.tfevents.1630352988.sundesh-Aspire-A715-75G.22460.0
# -9801.601558819453 69728 -0.14056909073570809 " " " " with discount=0
# -10881.018337693667 68700 -0.15838454640019894 " " " " with target_update = 100
print(val, cnt, val/cnt)