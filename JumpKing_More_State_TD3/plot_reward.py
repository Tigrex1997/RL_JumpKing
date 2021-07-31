import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


print("Import Successfully!")

reward_1=np.loadtxt('Reward_hist\\reward_4_128_256.txt')[0:100]
reward_2=np.loadtxt('Reward_hist\\reward_4_256_256.txt')[0:100]
reward_3=np.loadtxt('Reward_hist\\reward_4_64_256_v2_modified (2).txt')[0:100]

avg_reward1 = [-5000]
avg_reward2 = [-5000]
avg_reward3 = [-5000]

for i in range(1,101):
    avg_r1 = sum(reward_1[0:i])/i
    avg_r2 = sum(reward_2[0:i])/i
    avg_r3 = sum(reward_3[0:i])/i
    avg_reward1.append(avg_r1)
    avg_reward2.append(avg_r2)
    avg_reward3.append(avg_r3)

x_label = list(range(0,101))

plt.figure()
plt.plot(x_label,avg_reward3)
plt.plot(x_label,avg_reward1)
plt.plot(x_label,avg_reward2)

plt.legend(['Hidden_dim=64', 'Hidden_dim=128', 'Hidden_dim=256'])

plt.xlabel('Episode',fontsize=12)
plt.ylabel('Averaged Reward',fontsize=12)
plt.xlim(0, 100)
plt.title('Comparision of Number of Hidden Dimension',fontsize=15)
plt.grid()
plt.savefig("Avg_Reward.jpg")

# reward_TD3=np.loadtxt('TD3136STATESNOFLAGSTUCKFLAG1.txt')[0:100]
# reward_DDQN=np.loadtxt('DDQN236STATESNOSTUCKFLAG.txt')[0:100]

# avg_reward_DDQN = [-5000]

# for i in range(1,100):
#     avg_r1 = sum(reward_DDQN[0:i])/i
#     avg_reward_DDQN.append(avg_r1)




# x_label = list(range(0,100))
# x_label_DDQN = list(range(0,100))
# plt.figure()
# plt.plot(x_label,reward_TD3)
# plt.plot(x_label_DDQN,avg_reward_DDQN)

# plt.legend(['TD3','DDQN'])
# plt.xlabel('Episode',fontsize=12)
# plt.ylabel('Averaged Reward',fontsize=12)
# plt.xlim(0, 100)
# plt.title('Comparision of DDQN and TD3',fontsize=15)
# plt.grid()
# plt.savefig("TD3vsDDQN.jpg")