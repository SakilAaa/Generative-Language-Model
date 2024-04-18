import matplotlib.pyplot as plt
import os
import time 

def visualize_loss(train_loss_list, train_interval, val_loss_list, val_interval, dataset, out_dir):
    ### visualize loss of training & validation and save to [out_dir]/loss.png
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), tight_layout=True)
    axs[0].plot(range(0, len(train_loss_list), train_interval), train_loss_list[::train_interval], label="train")
    axs[0].set_title('Training Loss (Dataset: %s)' % dataset)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(range(0, len(val_loss_list), val_interval), val_loss_list[::val_interval], label="validation")
    axs[1].set_title('Validation Loss (Dataset: %s)' % dataset)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    t = int(time.time())
    plt.savefig(os.path.join(out_dir, '%dloss.png' % t))
    plt.close()