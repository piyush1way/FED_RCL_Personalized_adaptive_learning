# import copy

# import torch
# from torchmetrics import Metric
# __all__ = ['evaluate']

# class AccumTensor(Metric):
#     def __init__(self, default_value: torch.Tensor):
#         super().__init__()

#         self.add_state("val", default=default_value, dist_reduce_fx="sum")

#     def update(self, input_tensor: torch.Tensor):
#         self.val += input_tensor

#     def compute(self):
#         return self.val


# def evaluate(args, model, testloader, device) -> float:
#     '''
#     Return: accuracy of global test data
#     '''
#     eval_device = device if not args.multiprocessing else 'cuda:' + args.main_gpu
#     eval_model = copy.deepcopy(model)
#     eval_model.eval()
#     eval_model.to(eval_device)
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data[0].to(eval_device), data[1].to(eval_device)
#             outputs = eval_model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     acc = 100 * correct / float(total)
#     print('Accuracy of the network on the 10000 test images: %f %%' % (
#             100 * correct / float(total)))
#     eval_model.to('cpu')
#     return acc



import copy
import torch
from torchmetrics import Metric

__all__ = ['evaluate']

class AccumTensor(Metric):
    def __init__(self, default_value: torch.Tensor):
        super().__init__()
        self.add_state("val", default=default_value, dist_reduce_fx="sum")

    def update(self, input_tensor: torch.Tensor):
        self.val += input_tensor

    def compute(self):
        return self.val


def evaluate(args, model, testloader, device) -> dict:
    """
    Evaluate both global and personalized models on the test set.

    Args:
        args: Configuration arguments.
        model: Trained model.
        testloader: DataLoader for the test dataset.
        device: Device for evaluation.

    Returns:
        dict: Accuracy of both global and personalized models.
    """
    eval_device = device if not args.multiprocessing else 'cuda:' + args.main_gpu
    eval_model = copy.deepcopy(model)
    eval_model.eval()
    eval_model.to(eval_device)

    correct_global, correct_personalized, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(eval_device), labels.to(eval_device)

            # Evaluate both global and personalized model outputs
            global_results = eval_model(images)
            personalized_results = eval_model.forward_classifier(global_results["feature"])

            _, predicted_global = torch.max(global_results["logit"], 1)
            _, predicted_personalized = torch.max(personalized_results, 1)

            total += labels.size(0)
            correct_global += (predicted_global == labels).sum().item()
            correct_personalized += (predicted_personalized == labels).sum().item()

    acc_global = 100 * correct_global / float(total)
    acc_personalized = 100 * correct_personalized / float(total)

    print(f'Global Model Accuracy: {acc_global:.2f}% | Personalized Model Accuracy: {acc_personalized:.2f}%')

    eval_model.to('cpu')
    return {"acc_global": acc_global, "acc_personalized": acc_personalized}
