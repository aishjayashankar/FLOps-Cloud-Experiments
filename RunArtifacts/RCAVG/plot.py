import re
import matplotlib.pyplot as plt
import numpy as np

# Function to extract loss per round from log file
def extract_loss(file_path):
    rounds = []
    losses = []
    pattern = re.compile(r"round (\d+): ([\d\.]+)")

    with open(file_path, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                rounds.append(int(match.group(1)))
                losses.append(float(match.group(2)))

    return rounds, losses

# Function to extract accuracy per round from log file
def extract_accuracy(file_path):
    rounds = []
    accuracies = []
    pattern = re.compile(r"\( *(\d+), ([\d\.]+)\)")  # Pattern for accuracy data in (round, value) format

    with open(file_path, "r") as file:
        in_metrics_section = False
        for line in file:
            if "History (metrics, distributed, evaluate):" in line:
                in_metrics_section = True
                continue
            if in_metrics_section:
                match = pattern.findall(line)
                for round_num, acc in match:
                    rounds.append(int(round_num))
                    accuracies.append(float(acc))

    return rounds, accuracies

# File paths
logs = {
 #   "IID Baseline": "baseline.log",
    "Non-IID Baseline": "niid_baseline.log",
 #   "Node Disconnect": "nodeDisconnect.log",
 #   "Timeout": "timeout40s.log",
 #   "Non-IID Disconnect": "noniid_s42.log",
    "Non-IID Client 4 Disconnect": "client4Disconnect_niid.log"
    #"Multi Epochs": "multi_epoch.log"
}

# Colors and markers for plotting
plot_styles = {
 #   "IID Baseline": ("blue", "o"),
    "Non-IID Baseline": ("purple", "x"),
 #   "Node Disconnect": ("red", "s"),
 #   "Timeout": ("green", "D"),
 #   "Non-IID Disconnect": ("orange", "^"),
    "Non-IID Client 4 Disconnect": ("cyan", "v")
    #"Multi Epochs": ("purple", "x")
}

# Extract loss and accuracy data for all experiments
loss_data = {}
accuracy_data = {}

for label, log_file in logs.items():
    loss_data[label] = extract_loss(log_file)
    accuracy_data[label] = extract_accuracy(log_file)

# Plot and save accuracy vs. rounds with a focus on 50% to 70% accuracy
plt.figure(figsize=(10, 12))
for label, (rounds, accuracies) in accuracy_data.items():
    # Filter accuracies and corresponding rounds to focus on 0.5 to 0.7
    # filtered_rounds = [r for r, a in zip(rounds, accuracies) if 0.5 <= a <= 0.7]
    # filtered_accuracies = [a for a in accuracies if 0.5 <= a <= 0.7]
    # plt.plot(filtered_rounds, filtered_accuracies, marker=plot_styles[label][1], color=plot_styles[label][0], label=f"{label} Accuracy")
    plt.plot(rounds, accuracies, marker=plot_styles[label][1], color=plot_styles[label][0], label=f"{label} Accuracy")

plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.ylim(0.1,0.8)  # Set y-axis limits to 0.5 to 0.7
plt.title("Accuracy vs. Rounds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_vs_rounds_big.png")
plt.close()

print("Focused accuracy plot saved as accuracy_vs_rounds_50_to_70.png")

"""
# Plot and save loss vs. rounds with overlay
plt.figure(figsize=(10, 5))
for label, (rounds, losses) in loss_data.items():
    plt.plot(rounds, losses, marker=plot_styles[label][1], color=plot_styles[label][0], label=f"{label} Loss")
plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.title("Loss vs. Rounds")
plt.legend()
plt.grid(True)
plt.savefig("loss_vs_rounds_overlay_2_expts.png")
plt.close()

# Plot and save accuracy vs. rounds with overlay
#plt.figure(figsize=(10, 5))
for label, (rounds, accuracies) in accuracy_data.items():
    plt.plot(rounds, accuracies, marker=plot_styles[label][1], color=plot_styles[label][0], label=f"{label} Accuracy")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.title("Accuracy vs. Rounds")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_vs_rounds_overlay_2_expts.png")
plt.close()

print("Overlay plots saved as loss_vs_rounds_overlay.png and accuracy_vs_rounds_overlay.png")
"""