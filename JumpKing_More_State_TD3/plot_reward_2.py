import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

step_1 = np.loadtxt('/home/chendh/Course_Projects/EECS_598_003/Final_Project/Code/JumpKing_More_State_TD3/step.txt')
step_2 = np.loadtxt('/home/chendh/Course_Projects/EECS_598_003/Final_Project/Code/JumpKing_More_State_TD3/step2.txt')
with open('Reward_hist/steps_hist.data', 'rb') as filehandle:
    step_3 = pickle.load(filehandle)

print(step_1[99])
print(len(step_3))

step_1_sum = 0
avg_step_1 = []
step_2_sum = 0
avg_step_2 = []
step_3_sum = 0
avg_step_3 = []

avg_step_hist_mean = []

for i in range(100):
    step_1_sum += step_1[i]/35
    avg_step_1.append(step_1_sum/(i+1))

for i in range(100):
    step_2_sum += step_2[i]/35
    avg_step_2.append(step_2_sum / (i + 1))

# for i in range(100):
#     step_3_sum += step_3[i]/40
#     avg_step_3.append(step_3_sum/(i+1))

for i in range(100):
    temp_sum = step_1[i]/35 + step_2[i]/35
    avg_step_hist_mean.append(temp_sum/2)

x_label = list(range(0, 100))

plt.figure()
# plt.plot(x_label, avg_reward_hist_list_2, 'red')
# plt.plot(x_label, avg_reward_hist_list_3)
# plt.plot(x_label, avg_reward_hist_list_hang)
plt.plot(x_label, avg_step_hist_mean, 'blue', marker=',')


# Shade the area between y1 and line y=0
# plt.fill_between(x_label, avg_step_1, avg_step_2,
#                  facecolor="orange",  # The fill color
#                  # color='blue',  # The outline color
#                  alpha=0.1)  # Transparency of the fill
# # Shade the area between y1 and line y=0
# plt.fill_between(x_label, avg_step_1, avg_step_3,
#                  facecolor="orange",  # The fill color
#                  # color='blue',  # The outline color
#                  alpha=0.05)  # Transparency of the fill
# # Shade the area between y1 and line y=0
# plt.fill_between(x_label, avg_step_2, avg_step_3,
#                  facecolor="orange",  # The fill color
#                  # color='blue',  # The outline color
#                  alpha=0.05)  # Transparency of the fill

plt.legend(['Mean', 'Shaded region by trials'])

plt.xlabel('Episode', fontsize=12)
plt.ylabel('Averaged Step', fontsize=12)
plt.xlim(0, 101)
plt.title('Mean Average Step', fontsize=15)
plt.grid()
plt.savefig("Avg_Step.jpg")
print("Import Successfully!")