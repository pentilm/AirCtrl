import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

f = open('../../weights/history.txt', 'r')
losses = f.readlines()
f.close()

train, valid = [], []

for loss in losses:
    loss = yaml.load(loss)
    train = train + loss.get('loss')
    valid = valid + loss.get('val_loss')


epoch = range(1, len(train) + 1)

fig1 = plt.figure(1)
plt.plot(epoch, np.log(train), 'C2', marker='X')
plt.plot(epoch, np.log(valid), '--', marker='>')
plt.legend(['Training Total Loss', 'Validation Total Loss'], loc=1, prop={'size': 18})
plt.xlabel('Epochs', fontsize=20)
plt.ylabel(r'$\mathit{log}_{e}\:(Total \;\: Loss \;\: \mathcal{L})$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('loss_curve.jpg')
plt.show()
