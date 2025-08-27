import torch
import matplotlib.pyplot as plt
import os
import re

def extract_delay_lr(filename):
    delay_match = re.search(r'delay(\d+)', filename)
    lr_match = re.search(r'lr(\d+)', filename)
    delay = int(delay_match.group(1)) if delay_match else None
    lr = int(lr_match.group(1)) if lr_match else None
    return delay, lr

def load_record_files(record_dir):
    files = [f for f in os.listdir(record_dir) if f.endswith('.pt')]
    records = []
    for f in files:
        delay, lr = extract_delay_lr(f)
        if delay is not None and lr == 4:
            data = torch.load(os.path.join(record_dir, f))
            records.append((delay, lr, data))
    records.sort(key=lambda x: (x[0])) 
    return records

def plot_curves(records):
    plt.figure(figsize=(18, 5))

    # Estimation error
    plt.subplot(1, 3, 1)
    for delay, lr, data in records:
        plt.plot(data['iteration'], [te - tr for te, tr in zip(data['test_loss'], data['train_loss'])], label=f'delay={delay}')
    plt.xlabel('Iterations')
    plt.ylabel('Estimation Error')
    plt.title('Estimation Error Curve')
    # plt.ylim(0.05,0.25)
    plt.legend()

    # Test Loss
    plt.subplot(1, 3, 2)
    for delay, lr, data in records:
        plt.plot(data['iteration'], data['test_loss'], label=f'delay={delay}, lr={lr}')
    plt.xlabel('Iterations')
    plt.ylabel('Test Loss')
    plt.ylim(0.05,0.25)
    plt.title('Test Loss Curve')
    plt.legend()

    # Stability
    plt.subplot(1, 3, 3)
    for delay, lr, data in records:
        plt.plot(data['iteration'], data['stability'], label=f'delay={delay}, lr={lr}')
    plt.xlabel('Iterations')
    plt.ylabel('Stability')
    plt.ylim(0,0.04)
    plt.title('Stability Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    record_dir = './pair_2/records'
    records = load_record_files(record_dir)

    plot_curves(records)
