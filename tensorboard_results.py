from tensorflow.python.summary.summary_iterator import summary_iterator
import tensorflow as tf

val = 0
cnt = 0
for e in summary_iterator("/home/sundesh/Documents/git/ITU-Challenge/log_tensorboard/A2C_6/events.out.tfevents.1630254698.ff191160447d"):
    for v in e.summary.value:
        if (v.tag == 'episode_reward'):
            val += v.simple_value
            cnt += 1
            # print(v.simple_value, v.tag)
    # if v.tag == 'loss' or v.tag == 'accuracy':
    # print(v.simple_value)
    # break

print(val, cnt)