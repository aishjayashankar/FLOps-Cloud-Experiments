import re
import matplotlib.pyplot as plt

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
    "Baseline": "baseline.log",
    "Node Disconnect": "nodeDisconnect.log",
    "Timeout": "timeout40s.log"
}

# Colors and markers for plotting
plot_styles = {
    "Baseline": ("blue", "o"),
    "Node Disconnect": ("red", "s"),
    "Timeout": ("green", "D")
}

# Extract loss and accuracy data for all experiments
loss_data = {}
accuracy_data = {}

for label, log_file in logs.items():
    loss_data[label] = extract_loss(log_file)
    accuracy_data[label] = extract_accuracy(log_file)

# Plot and save loss vs. rounds with overlay
plt.figure(figsize=(10, 5))
for label, (rounds, losses) in loss_data.items():
    plt.plot(rounds, losses, marker=plot_styles[label][1], color=plot_styles[label][0], label=f"{label} Loss")
plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.title("Loss vs. Rounds")
plt.legend()
plt.grid(True)
plt.savefig("loss_vs_rounds_overlay.png")
plt.close()

# Plot and save accuracy vs. rounds with overlay
plt.figure(figsize=(10, 5))
for label, (rounds, accuracies) in accuracy_data.items():
    plt.plot(rounds, accuracies, marker=plot_styles[label][1], color=plot_styles[label][0], label=f"{label} Accuracy")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Rounds")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_vs_rounds_overlay.png")
plt.close()

print("Overlay plots saved as loss_vs_rounds_overlay.png and accuracy_vs_rounds_overlay.png")