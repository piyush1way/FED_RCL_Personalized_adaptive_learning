import torch
import pickle
import numpy as np

# Function to load CIFAR-10 data from file
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Load test data
test_data = unpickle('/kaggle/input/cifar-10-batches/cifar-10-batches-py/test_batch')
test_images = test_data[b'data']
test_labels = test_data[b'labels']

# Reshape and normalize images
test_images = test_images.reshape(-1, 3, 32, 32).astype('float32') / 255.0
# Normalize with CIFAR-10 mean and std
mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2023, 0.1994, 0.2010])
for i in range(3):
    test_images[:, i, :, :] = (test_images[:, i, :, :] - mean[i]) / std[i]

# Convert to torch tensors
X_test = torch.FloatTensor(test_images)
y_test = torch.LongTensor(test_labels)

# Create dataset and dataloader
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# Load the model - try with TorchScript model first
try:
    model = torch.jit.load('/kaggle/input/your-model-checkpoint/res18.pt.jit')
    print("Loaded TorchScript model successfully")
except Exception as e:
    print(f"Error loading TorchScript model: {e}")
    # Fall back to regular PyTorch model
    try:
        checkpoint = torch.load('/kaggle/input/your-model-checkpoint/res18.pt')
        print("Loaded PyTorch checkpoint with keys:", checkpoint.keys())
        # If you have access to model definition
        # from your_model_file import ResNet18
        # model = ResNet18()
        # model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e2:
        print(f"Error loading PyTorch model: {e2}")

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Evaluate
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# Print overall accuracy
print(f'Overall Accuracy on CIFAR-10 test set: {100 * correct / total:.2f}%')

# Load class names
meta_data = unpickle('/kaggle/input/cifar-10-batches/cifar-10-batches-py/batches.meta')
class_names = [label.decode('utf-8') for label in meta_data[b'label_names']]

# Print per-class accuracy
print("\nPer-class Accuracy:")
for i in range(10):
    print(f'Accuracy of {class_names[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')

# Save results to file for later reference
with open('cifar10_accuracy_results.txt', 'w') as f:
    f.write(f'Overall Accuracy: {100 * correct / total:.2f}%\n\n')
    f.write('Per-class Accuracy:\n')
    for i in range(10):
        f.write(f'Accuracy of {class_names[i]}: {100 * class_correct[i] / class_total[i]:.2f}%\n')

print("\nResults saved to 'cifar10_accuracy_results.txt'")